"""
g2 Evaluator - Mission Completion Time Estimation

Centralized g2() function for MH-Dec-MCTS, used by both HLP and LLP planners.

Paper reference: "Multi-Horizon Multi-Agent Planning Using De-centralized Monte Carlo Tree Search"
Seiler et al., 2024

The reward decomposition is:
    g = g1(LL intents) + g2(all intents)

Available g2 modes (selectable via `mode` parameter):

  "stub"         - Original placeholder. Ignores hl_intents entirely.
                   Uses fixed coverage rate. Always returns near-zero.
                   Kept for baseline comparison.

  "hl_aware"     - HL-intent aware critical path (RECOMMENDED).
                   Simulates each agent walking its HL region sequence from
                   its LL end-position. Accounts for travel time + scan time
                   per region (scan_time = region_entropy / coverage_rate).
                   g2 = max over agents (critical path bottleneck).
                   Requires env_state["regions"] for full accuracy.

  "entropy_rate" - Fast entropy-rate estimate. Does not use HL intents.
                   g2 = remaining_entropy / (effective_rate * n_agents).
                   Effective rate penalised by LL footprint overlap.
                   Falls back gracefully when regions are unavailable.

  "eta_weighted" - Distance-weighted coverage deficit.
                   For each uncertain region, finds the agent that will cover
                   it soonest (via HL ETA sequence or estimated travel time).
                   g2 = sum(region_entropy * min_ETA_r) / total_entropy.
                   Falls back to entropy_rate when regions unavailable.

All modes scale output by 0.1 to stay comparable to g1 (IG units).
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Per-call heavy-parts cache
# ---------------------------------------------------------------------------
# During HLP MCTS (30 iterations × 2 g2 calls per agent), the belief and
# ll_intents are identical across all 60 calls.  This cache avoids rebuilding
# entropy_map, SAT, and covered-cell structures more than once per unique
# (belief, ll_intents) combination.
#
# Key  : (id(belief), belief.shape, ll_key)  where ll_key is a tuple of
#         (agent_id, id(intent)) sorted for stability.
# Value: (belief_ref, heavy_result_tuple)   belief_ref keeps the array alive
#         so that id() stays valid (Python GC cannot reuse the id).
_heavy_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_HEAVY_CACHE_MAX = 8  # one slot per active agent + a few spare


def _make_ll_key(ll_intents: Dict[int, Any]) -> tuple:
    """Cheap, stable fingerprint for an ll_intents dict within one planning cycle."""
    return tuple(sorted((aid, id(intent)) for aid, intent in ll_intents.items()))


def _compute_heavy_parts(
    belief: np.ndarray,
    ll_intents: Dict[int, Any],
    regions: Dict,
) -> Tuple:
    """
    Compute (and cache) the belief- and ll_intents-dependent quantities shared
    by all g2 modes:

        entropy_map, total_entropy,
        ll_covered_cells, ll_covered_entropy,
        ll_overlap_penalty, ll_execution_time, ll_end_positions,
        remaining_entropy_map, region_remaining_entropy

    All three non-stub g2 modes need exactly these values.  When called
    repeatedly with the same belief object and the same intent objects
    (as happens during HLP MCTS iterations) the result is returned from
    cache without any numpy work.
    """
    cache_key = (id(belief), belief.shape, _make_ll_key(ll_intents))
    if cache_key in _heavy_cache:
        cached_ref, result = _heavy_cache[cache_key]
        if cached_ref is belief:  # same array object → id still valid
            _heavy_cache.move_to_end(cache_key)
            return result

    # --- expensive numpy work (only runs on cache miss) ---
    p = _occupancy_prob(belief)
    entropy_map = _entropy(p)
    total_entropy = float(np.sum(entropy_map))

    ll_execution_time = compute_ll_execution_time(ll_intents)
    ll_covered_cells = compute_ll_covered_cells(ll_intents)
    ll_overlap_penalty = compute_ll_overlap_penalty(ll_intents)
    ll_end_positions = compute_ll_end_positions(ll_intents)

    # Build remaining_entropy_map with vectorised zeroing (no Python loop)
    remaining_entropy_map = entropy_map.copy()
    if ll_covered_cells:
        covered_arr = np.array(list(ll_covered_cells), dtype=np.intp)  # (N, 2)
        H, W = entropy_map.shape
        r, c = covered_arr[:, 0], covered_arr[:, 1]
        valid = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        # covered entropy (vectorised sum)
        ll_covered_entropy = float(np.sum(entropy_map[r[valid], c[valid]]))
        remaining_entropy_map[r[valid], c[valid]] = 0.0
    else:
        ll_covered_entropy = 0.0

    region_remaining_entropy = _region_entropy_sat(remaining_entropy_map, regions)

    # Per-region hotspot: grid (i, j) of the cell with maximum remaining entropy.
    # Used by hl_aware / eta_weighted to compute travel time to the *most unexplored*
    # part of each region rather than its geometric centre.
    region_hotspot_ij: Dict[int, Tuple[float, float]] = {}
    H_rem, W_rem = remaining_entropy_map.shape
    for rid, region in regions.items():
        (imin, imax), (jmin, jmax) = region["bounds"]
        r0 = max(0, min(imin, H_rem))
        r1 = max(0, min(imax, H_rem))
        c0 = max(0, min(jmin, W_rem))
        c1 = max(0, min(jmax, W_rem))
        sub = remaining_entropy_map[r0:r1, c0:c1]
        if sub.size > 0 and float(np.max(sub)) > 1e-6:
            flat_idx = int(np.argmax(sub))
            dr, dc = np.unravel_index(flat_idx, sub.shape)
            region_hotspot_ij[rid] = (float(r0 + dr), float(c0 + dc))
        else:
            ri, rj = region["center"]
            region_hotspot_ij[rid] = (float(ri), float(rj))

    result = (
        entropy_map,
        total_entropy,
        ll_covered_cells,
        ll_covered_entropy,
        ll_overlap_penalty,
        ll_execution_time,
        ll_end_positions,
        remaining_entropy_map,
        region_remaining_entropy,
        region_hotspot_ij,
    )

    if len(_heavy_cache) >= _HEAVY_CACHE_MAX:
        _heavy_cache.popitem(last=False)  # evict oldest
    _heavy_cache[cache_key] = (belief, result)  # keep belief alive
    return result


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

VALID_MODES = ("stub", "hl_aware", "entropy_rate", "eta_weighted")


def g2(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
    mode: str = "hl_aware",
) -> float:
    """
    Estimate mission completion time conditioned on LL+HL intents.

    Args:
        ll_intents : Dict[agent_id -> LLIntent]  short-horizon action plans
        hl_intents : Dict[agent_id -> HLIntent]  region allocation plans
        env_state  : Dict with keys:
                       "belief"  - np.ndarray belief map (required)
                       "regions" - Dict[region_id -> {bounds, center, area}]
                                   required for hl_aware / eta_weighted
                       "grid_info" - grid metadata (optional, unused currently)
        agent_id   : Optional calling agent ID (unused; kept for API compat)
        mode       : One of "stub" | "hl_aware" | "entropy_rate" | "eta_weighted"

    Returns:
        g2_value : Estimated mission time (lower is better).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"g2 mode '{mode}' unknown. Choose from {VALID_MODES}")

    if mode == "stub":
        return _g2_stub(ll_intents, hl_intents, env_state, agent_id)
    elif mode == "hl_aware":
        return _g2_hl_aware(ll_intents, hl_intents, env_state, agent_id)
    elif mode == "entropy_rate":
        return _g2_entropy_rate(ll_intents, hl_intents, env_state, agent_id)
    elif mode == "eta_weighted":
        return _g2_eta_weighted(ll_intents, hl_intents, env_state, agent_id)


# ---------------------------------------------------------------------------
# Mode 1 - stub (original broken implementation, kept for comparison)
# ---------------------------------------------------------------------------


def _g2_stub(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
) -> float:
    """
    Original stub implementation.
    Ignores hl_intents completely; uses a fixed nominal coverage rate.
    hl_intents has zero effect, so HLP MCTS sees no signal.
    Kept for baseline comparison only.
    """
    belief = env_state.get("belief")
    if belief is None:
        return 0.0

    ll_execution_time = compute_ll_execution_time(ll_intents)
    ll_covered_cells = compute_ll_covered_cells(ll_intents)
    ll_overlap_penalty = compute_ll_overlap_penalty(ll_intents)

    total_uncertain_area = compute_uncertain_area(belief, threshold=0.3)
    remaining_area = max(0, total_uncertain_area - len(ll_covered_cells))

    num_agents = max(1, len(ll_intents)) if ll_intents else 1
    nominal_coverage_rate = 50.0 * num_agents
    hl_completion_time = (
        remaining_area / nominal_coverage_rate if remaining_area > 0 else 0.0
    )

    total_time = ll_execution_time + hl_completion_time + ll_overlap_penalty
    return 0.1 * total_time


# ---------------------------------------------------------------------------
# Mode 2 - hl_aware (critical path over region sequences)
# ---------------------------------------------------------------------------


def _g2_hl_aware(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
) -> float:
    """
    HL-intent aware critical path estimation.

    Phase 1: Execute LL intents in parallel; record end-positions + covered cells.
    Phase 2: Each agent walks its HL region sequence from its LL end-position.
             per-region cost = travel_time + scan_time
             scan_time  = region_remaining_entropy / nominal_cells_per_step
             travel_time = euclidean_dist / xy_speed
    g2 = max over agents (critical path)

    Requires env_state["regions"] for full accuracy.
    Falls back gracefully to stub if regions are missing.
    """
    belief = env_state.get("belief")
    regions = env_state.get("regions", {})

    if belief is None:
        return 0.0

    # All heavy numpy work is cached across repeated calls with same belief+ll_intents
    (
        entropy_map,
        total_entropy,
        ll_covered_cells,
        ll_covered_entropy,
        ll_overlap_penalty,
        ll_execution_time,
        ll_end_positions,
        remaining_entropy_map,
        region_remaining_entropy,
        region_hotspot_ij,
    ) = _compute_heavy_parts(belief, ll_intents, regions)

    nominal_cells_per_step = 50.0
    # xy_speed in grid-cell units (agent moves ~1 cell per step at xy_step ≈ grid_length)
    xy_speed = 1.0

    # Grid-coord conversion helpers: world (x, y) → grid (i, j)
    grid_info_obj = env_state.get("grid_info")
    gl = getattr(grid_info_obj, "length", None)
    gs = getattr(grid_info_obj, "shape", None)
    gc = getattr(grid_info_obj, "center", True)
    use_grid_coords = gl is not None and gs is not None
    if use_grid_coords:
        _ci = gs[0] // 2 if gc else 0
        _cj = gs[1] // 2 if gc else 0

    def _to_grid(wx, wy):
        """Convert world (x, y) → grid (i, j)."""
        if use_grid_coords:
            return (-wy / gl + _ci, wx / gl + _cj)
        return (wx, wy)  # fallback (old mixed-coord behaviour)

    all_agent_ids = set(list(ll_intents.keys()) + list(hl_intents.keys()))

    if not all_agent_ids:
        total_remaining = float(np.sum(remaining_entropy_map))
        return 0.1 * (ll_execution_time + total_remaining / nominal_cells_per_step)

    agent_completion_times = []

    for aid in all_agent_ids:
        hl_intent = hl_intents.get(aid, None)
        start_pos = ll_end_positions.get(aid, None)

        if hl_intent is None or not hl_intent.region_sequence or not regions:
            agent_completion_times.append(ll_execution_time)
            continue

        agent_time = float(ll_execution_time)
        # Convert LL end-position (world coords) to grid coords for consistent distance
        pos = _to_grid(start_pos[0], start_pos[1]) if start_pos is not None else None

        for rid in hl_intent.region_sequence:
            if rid not in regions:
                continue
            # Target: most-unexplored cell (grid coords), fallback to region centre
            target = region_hotspot_ij.get(rid, regions[rid]["center"])

            if pos is not None:
                dist = np.sqrt((pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2)
                travel_steps = dist / xy_speed
            else:
                travel_steps = 0.0

            region_entropy = region_remaining_entropy.get(rid, 0.0)
            scan_steps = region_entropy / nominal_cells_per_step

            agent_time += travel_steps + scan_steps
            pos = target  # chain: next travel starts from this hotspot (grid coords)

        agent_completion_times.append(agent_time)

    critical_path = (
        max(agent_completion_times) if agent_completion_times else ll_execution_time
    )
    return 0.1 * (critical_path + ll_overlap_penalty)


# ---------------------------------------------------------------------------
# Mode 3 - entropy_rate (fast, HL intents unused)
# ---------------------------------------------------------------------------


def _g2_entropy_rate(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
) -> float:
    """
    Fast entropy-rate estimate. Does NOT use HL intents.

    g2 = ll_exec_time + remaining_entropy / effective_rate
    where effective_rate = nominal_rate * N_agents * (1 - overlap_fraction)

    Pro: very cheap, no per-region iteration.
    Con: HL intents have zero effect - only useful for LLP g2 where
         region info is unavailable.
    """
    belief = env_state.get("belief")
    if belief is None:
        return 0.0

    (
        entropy_map,
        total_entropy,
        ll_covered_cells,
        ll_covered_entropy,
        ll_overlap_penalty,
        ll_execution_time,
        _ll_end_positions,
        _remaining_entropy_map,
        _region_remaining_entropy,
        _region_hotspot_ij,
    ) = _compute_heavy_parts(belief, ll_intents, env_state.get("regions", {}))

    remaining_entropy = max(0.0, total_entropy - ll_covered_entropy)

    n_agents = max(1, len(ll_intents))
    nominal_rate = 50.0
    effective_rate = nominal_rate * n_agents * max(0.05, 1.0 - ll_overlap_penalty)

    hl_completion_time = remaining_entropy / effective_rate
    return 0.1 * (ll_execution_time + hl_completion_time)


# ---------------------------------------------------------------------------
# Mode 4 - eta_weighted (ETA-weighted coverage deficit)
# ---------------------------------------------------------------------------


def _g2_eta_weighted(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
) -> float:
    """
    ETA-weighted coverage deficit.

    For each uncertain region, find the agent covering it soonest
    (HL ETA sequence if available, else estimated travel time).

    g2 = sum_r( region_entropy_r * min_ETA_r ) / total_remaining_entropy

    Rewards HL plans that dispatch agents promptly to high-entropy areas.
    Uncovered regions receive worst-case ETA = total scan time.

    Requires env_state["regions"]. Falls back to entropy_rate if missing.
    """
    belief = env_state.get("belief")
    regions = env_state.get("regions", {})

    if belief is None:
        return 0.0

    if not regions:
        return _g2_entropy_rate(ll_intents, hl_intents, env_state, agent_id)

    (
        entropy_map,
        total_entropy,
        ll_covered_cells,
        ll_covered_entropy,
        ll_overlap_penalty,
        ll_execution_time,
        ll_end_positions,
        remaining_entropy_map,
        region_remaining_entropy,
        region_hotspot_ij,
    ) = _compute_heavy_parts(belief, ll_intents, regions)

    total_remaining = max(0.0, total_entropy - ll_covered_entropy)
    if total_remaining < 1e-6:
        return 0.0

    # xy_speed in grid-cell units (see hl_aware for coordinate system rationale)
    xy_speed = 1.0
    nominal_cells_per_step = 50.0
    worst_case_eta = ll_execution_time + total_remaining / nominal_cells_per_step

    # Grid-coord conversion: world (x, y) → grid (i, j)
    grid_info_obj = env_state.get("grid_info")
    gl = getattr(grid_info_obj, "length", None)
    gs = getattr(grid_info_obj, "shape", None)
    gc = getattr(grid_info_obj, "center", True)
    use_grid_coords = gl is not None and gs is not None
    if use_grid_coords:
        _ci = gs[0] // 2 if gc else 0
        _cj = gs[1] // 2 if gc else 0

    def _to_grid(wx, wy):
        if use_grid_coords:
            return (-wy / gl + _ci, wx / gl + _cj)
        return (wx, wy)

    weighted_sum = 0.0

    for rid, region in regions.items():
        region_entropy = region_remaining_entropy.get(rid, 0.0)
        if region_entropy < 1e-6:
            continue

        target = region_hotspot_ij.get(rid, region["center"])
        min_eta = float("inf")

        for aid, hl_intent in hl_intents.items():
            if hl_intent is None or not hl_intent.region_sequence:
                continue
            if rid not in hl_intent.region_sequence:
                continue

            seq_idx = hl_intent.region_sequence.index(rid)

            # Use stored ETA if available
            if hl_intent.eta_sequence and seq_idx < len(hl_intent.eta_sequence):
                eta = ll_execution_time + hl_intent.eta_sequence[seq_idx]
            else:
                # Estimate by walking region sequence to this region
                raw_pos = ll_end_positions.get(aid, None)
                pos = _to_grid(raw_pos[0], raw_pos[1]) if raw_pos is not None else None
                t = float(ll_execution_time)
                for r2 in hl_intent.region_sequence[: seq_idx + 1]:
                    if r2 not in regions:
                        break
                    t2 = region_hotspot_ij.get(r2, regions[r2]["center"])
                    if pos is not None:
                        dist = np.sqrt((pos[0] - t2[0]) ** 2 + (pos[1] - t2[1]) ** 2)
                        t += dist / xy_speed
                    if r2 != rid:
                        t += (
                            region_remaining_entropy.get(r2, 0.0)
                            / nominal_cells_per_step
                        )
                    pos = t2  # grid coords throughout
                eta = t

            min_eta = min(min_eta, eta)

        if min_eta == float("inf"):
            min_eta = worst_case_eta

        weighted_sum += region_entropy * min_eta

    g2_raw = weighted_sum / total_remaining
    return 0.1 * g2_raw


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------


def _occupancy_prob(belief: np.ndarray) -> np.ndarray:
    """Extract 2-D occupied-channel probability from belief array."""
    if belief.ndim == 3:
        return np.clip(belief[:, :, 1], 1e-10, 1 - 1e-10)
    return np.clip(belief, 1e-10, 1 - 1e-10)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy map (base-2)."""
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _region_entropy_sat(
    entropy_map: np.ndarray,
    regions: Dict[int, Any],
) -> Dict[int, float]:
    """
    Compute total entropy per region using a summed-area table (O(1) per query).
    Returns dict mapping region_id -> float entropy sum.
    """
    if not regions:
        return {}

    sat = np.pad(
        entropy_map.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    result = {}
    for rid, region in regions.items():
        (imin, imax), (jmin, jmax) = region["bounds"]
        total = float(
            sat[imax, jmax] - sat[imin, jmax] - sat[imax, jmin] + sat[imin, jmin]
        )
        result[rid] = max(0.0, total)

    return result


# ---------------------------------------------------------------------------
# Public utility functions (used by callers and tests)
# ---------------------------------------------------------------------------


def compute_uncertain_area(belief: np.ndarray, threshold: float = 0.3) -> float:
    """Count cells with binary entropy above threshold."""
    if belief is None:
        return 0.0
    p = _occupancy_prob(belief)
    ent = _entropy(p)
    return float(np.sum(ent > threshold))


def compute_ll_execution_time(ll_intents: Dict[int, Any]) -> float:
    """Maximum LL horizon across all agents (steps when last agent finishes LL)."""
    if not ll_intents:
        return 0.0
    max_horizon = 0.0
    for intent in ll_intents.values():
        if intent is None:
            continue
        if hasattr(intent, "action_sequence"):
            h = len(intent.action_sequence)
        elif hasattr(intent, "horizon"):
            h = intent.horizon or 5
        else:
            h = 5
        max_horizon = max(max_horizon, h)
    return float(max_horizon)


def compute_ll_covered_cells(ll_intents: Dict[int, Any]) -> set:
    """Set of (i, j) cells covered by any LL intent footprint.

    Uses numpy slicing instead of nested Python loops for speed.
    Returns an empty set quickly when there are no intents.
    """
    if not ll_intents:
        return set()

    # Collect all (imin, imax, jmin, jmax) rectangles first
    rects = []
    for intent in ll_intents.values():
        if intent is None or not hasattr(intent, "footprint_sequence"):
            continue
        for fp in intent.footprint_sequence:
            if fp is None or len(fp) != 4:
                continue
            rects.append(fp)

    if not rects:
        return set()

    # Build a boolean mask over a bounding box, then convert once to a set.
    # This avoids O(n_cells) Python-level iterations.
    imins, imaxs, jmins, jmaxs = zip(*rects)
    gimin = min(imins)
    gimax = max(imaxs)
    gjmin = min(jmins)
    gjmax = max(jmaxs)
    H = gimax - gimin
    W = gjmax - gjmin
    if H <= 0 or W <= 0:
        return set()

    mask = np.zeros((H, W), dtype=bool)
    for imin, imax, jmin, jmax in rects:
        r0 = imin - gimin
        r1 = imax - gimin
        c0 = jmin - gjmin
        c1 = jmax - gjmin
        mask[r0:r1, c0:c1] = True

    rows, cols = np.nonzero(mask)
    return set(zip((rows + gimin).tolist(), (cols + gjmin).tolist()))


def compute_ll_end_positions(
    ll_intents: Dict[int, Any],
) -> Dict[int, Tuple[float, float]]:
    """Agent end-positions (x, y) after LL intent execution."""
    end_positions = {}
    for aid, intent in ll_intents.items():
        if intent is None or not hasattr(intent, "state_sequence"):
            continue
        if intent.state_sequence:
            last = intent.state_sequence[-1]
            if len(last) >= 2:
                end_positions[aid] = (last[0], last[1])
    return end_positions


def compute_ll_overlap_penalty(ll_intents: Dict[int, Any]) -> float:
    """
    Fraction of LL-covered cells covered by more than one agent.
    Range [0, 1]. Higher = more redundant work.

    Uses numpy masks for each agent then counts overlap, avoiding nested Python loops.
    """
    if not ll_intents or len(ll_intents) < 2:
        return 0.0

    # Collect per-agent rectangles
    agent_rects: Dict[Any, list] = {}
    all_rects = []
    for aid, intent in ll_intents.items():
        if intent is None or not hasattr(intent, "footprint_sequence"):
            continue
        rects = [
            fp for fp in intent.footprint_sequence if fp is not None and len(fp) == 4
        ]
        if rects:
            agent_rects[aid] = rects
            all_rects.extend(rects)

    if not all_rects or len(agent_rects) < 2:
        return 0.0

    imins, imaxs, jmins, jmaxs = zip(*all_rects)
    gimin = min(imins)
    gimax = max(imaxs)
    gjmin = min(jmins)
    gjmax = max(jmaxs)
    H = gimax - gimin
    W = gjmax - gjmin
    if H <= 0 or W <= 0:
        return 0.0

    # Sum of per-agent masks = coverage count per cell
    coverage_count = np.zeros((H, W), dtype=np.int16)
    for aid, rects in agent_rects.items():
        agent_mask = np.zeros((H, W), dtype=bool)
        for imin, imax, jmin, jmax in rects:
            agent_mask[imin - gimin : imax - gimin, jmin - gjmin : jmax - gjmin] = True
        coverage_count += agent_mask  # bool is treated as 0/1 by numpy

    total_cells = int(np.count_nonzero(coverage_count))
    if total_cells == 0:
        return 0.0

    overlap_cells = int(np.count_nonzero(coverage_count > 1))
    return float(overlap_cells) / total_cells


# ---------------------------------------------------------------------------
# Legacy stubs (backward compatibility)
# ---------------------------------------------------------------------------


def evaluate_region_coverage(
    hl_intents: Dict[int, Any],
    belief: np.ndarray,
    grid_info: Any,
) -> float:
    """Legacy stub. Use _g2_hl_aware instead."""
    return 0.0


def compute_coordination_penalty(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
) -> float:
    """Legacy stub. Overlap handled in compute_ll_overlap_penalty."""
    return 0.0


def get_region_uncertainty(
    belief: np.ndarray,
    region_bounds: Tuple[int, int, int, int],
) -> float:
    """Total entropy in a rectangular region."""
    imin, imax, jmin, jmax = region_bounds
    region_belief = belief[imin:imax, jmin:jmax]
    if region_belief.ndim == 3:
        region_belief = region_belief[:, :, 1:]
    return compute_uncertain_area(region_belief)


def compute_region_overlap(
    region1: Tuple[int, int, int, int],
    region2: Tuple[int, int, int, int],
) -> float:
    """Intersection area between two (imin, imax, jmin, jmax) rectangles."""
    i1_min, i1_max, j1_min, j1_max = region1
    i2_min, i2_max, j2_min, j2_max = region2
    i_overlap = max(0, min(i1_max, i2_max) - max(i1_min, i2_min))
    j_overlap = max(0, min(j1_max, j2_max) - max(j1_min, j2_min))
    return float(i_overlap * j_overlap)

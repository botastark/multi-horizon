"""
g2 Evaluator - Mission Completion Time Estimation

Centralized g2() function for MH-Dec-MCTS, used by both HLP and LLP planners.

Paper reference: "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"
Seiler et al., 2024

The reward decomposition is:
    g = g1(LL intents) + g2(all intents)

Where:
- g1: Immediate task quality (IG from LLP) - computed in LowLevelPlanner
- g2: Long-horizon mission estimate (coordination cost) - computed here

g2 estimates the time to mission completion considering:
1. Remaining uncertainty in the environment
2. Distance to uncovered regions
3. Coordination costs (overlap, conflicts with teammates)
4. Region allocation efficiency

This module provides the shared g2() implementation that both planners can call.
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np


def g2(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
    env_state: Dict[str, Any],
    agent_id: Optional[int] = None,
) -> float:
    """
    Estimate mission completion time conditioned on LL intents.

    Paper-correct two-phase evaluation (Algorithm 2):

    Phase 1 - Execute LL intents (fixed):
        - Consume LL execution time
        - Update covered area (remove LL footprints from uncertain area)
        - Update agent end positions

    Phase 2 - Finish mission using HL intents:
        - Starting from LL end states
        - Estimate time to complete remaining regions

    Final value:
        g2 = LL_execution_time + HL_completion_time

    This ensures:
    - ✅ HL avoids regions already covered by LL
    - ✅ g2 decreases as LL progresses
    - ✅ Bottom-up information flow (LL → HL)
    - ✅ Alignment emerges naturally

    Args:
        ll_intents: Dict mapping agent_id -> LLIntent (short-horizon action plans)
        hl_intents: Dict mapping agent_id -> HLIntent (region allocation plans)
        env_state: Dictionary containing:
            - belief: Current belief map (np.ndarray)
            - grid_info: Grid metadata (shape, cell_size, etc.)
            - positions: Current agent positions (optional)
        agent_id: Optional agent ID (for agent-specific evaluation)

    Returns:
        g2_value: Estimated mission time (lower is better)
    """
    belief = env_state.get("belief")
    if belief is None:
        return 0.0

    # =========================================================================
    # PHASE 1: Execute LL intents
    # =========================================================================

    # 1a. Compute LL execution time
    ll_execution_time = compute_ll_execution_time(ll_intents)

    # 1b. Compute area covered by LL intents
    ll_covered_cells = compute_ll_covered_cells(ll_intents)

    # 1c. Compute overlap penalty (multiple agents covering same cells)
    ll_overlap_penalty = compute_ll_overlap_penalty(ll_intents)

    # 1d. Get agent end positions after LL execution
    ll_end_positions = compute_ll_end_positions(ll_intents)

    # =========================================================================
    # PHASE 2: Estimate HL completion time
    # =========================================================================

    # 2a. Compute total uncertain area
    total_uncertain_area = compute_uncertain_area(belief, threshold=0.3)

    # 2b. Remaining area after LL execution
    remaining_area = max(0, total_uncertain_area - len(ll_covered_cells))

    # 2c. Estimate time to complete remaining area using HL intents
    # Nominal coverage rate: ~50 cells per footprint
    num_agents = max(1, len(ll_intents)) if ll_intents else 1
    nominal_coverage_rate = 50.0 * num_agents

    hl_completion_time = (
        remaining_area / nominal_coverage_rate if remaining_area > 0 else 0.0
    )

    # =========================================================================
    # TOTAL TIME
    # =========================================================================

    # g2 = LL execution + HL completion + coordination cost
    total_time = ll_execution_time + hl_completion_time + ll_overlap_penalty

    # Scale to keep values reasonable relative to IG (~0-10)
    g2_value = 0.1 * total_time

    return g2_value


def compute_uncertain_area(belief: np.ndarray, threshold: float = 0.3) -> float:
    """
    Compute area (in cells) that needs coverage based on uncertainty.

    Args:
        belief: Belief map (shape: [H, W, 2] or [H, W, 3])
                belief[:,:,1] = p(occupied)
        threshold: Entropy threshold for considering a cell uncertain

    Returns:
        Number of uncertain cells needing coverage
    """
    if belief is None:
        return 0.0

    # Extract occupancy probabilities
    p_occupied = belief[:, :, 1]

    # Compute entropy: H = -p*log(p) - (1-p)*log(1-p)
    eps = 1e-10
    p_occupied = np.clip(p_occupied, eps, 1 - eps)
    entropy = -(
        p_occupied * np.log(p_occupied) + (1 - p_occupied) * np.log(1 - p_occupied)
    )

    # Count cells with entropy above threshold
    uncertain_cells = np.sum(entropy > threshold)

    return float(uncertain_cells)


def compute_ll_execution_time(ll_intents: Dict[int, Any]) -> float:
    """
    Compute time to execute all LL intents.

    Args:
        ll_intents: Dict mapping agent_id -> LLIntent

    Returns:
        Maximum horizon across all agents (time when all LL intents complete)
    """
    if not ll_intents:
        return 0.0

    max_horizon = 0.0
    for agent_id, ll_intent in ll_intents.items():
        if ll_intent is None or not hasattr(ll_intent, "horizon"):
            continue

        # Count actual steps in the plan
        if hasattr(ll_intent, "action_sequence"):
            horizon = len(ll_intent.action_sequence)
        else:
            horizon = ll_intent.horizon if ll_intent.horizon else 5

        max_horizon = max(max_horizon, horizon)

    return float(max_horizon)


def compute_ll_covered_cells(ll_intents: Dict[int, Any]) -> set:
    """
    Compute set of cells that will be covered by LL intents.

    Args:
        ll_intents: Dict mapping agent_id -> LLIntent

    Returns:
        Set of (i, j) cell coordinates that will be covered
    """
    if not ll_intents:
        return set()

    # Collect all cells from all agents' footprints
    all_cells = set()
    for agent_id, ll_intent in ll_intents.items():
        if ll_intent is None or not hasattr(ll_intent, "footprint_sequence"):
            continue

        for footprint in ll_intent.footprint_sequence:
            if footprint is None or len(footprint) != 4:
                continue
            imin, imax, jmin, jmax = footprint
            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    all_cells.add((i, j))

    return all_cells


def compute_ll_end_positions(
    ll_intents: Dict[int, Any],
) -> Dict[int, Tuple[float, float]]:
    """
    Compute agent end positions after LL intent execution.

    Args:
        ll_intents: Dict mapping agent_id -> LLIntent

    Returns:
        Dict mapping agent_id -> (x, y) end position
    """
    end_positions = {}

    for agent_id, ll_intent in ll_intents.items():
        if ll_intent is None or not hasattr(ll_intent, "state_sequence"):
            continue

        # Get last position from state sequence
        if ll_intent.state_sequence and len(ll_intent.state_sequence) > 0:
            last_state = ll_intent.state_sequence[-1]
            if len(last_state) >= 2:
                end_positions[agent_id] = (last_state[0], last_state[1])

    return end_positions


def compute_ll_overlap_penalty(ll_intents: Dict[int, Any]) -> float:
    """
    Compute penalty for overlapping LL intent footprints.

    Overlap means multiple agents plan to cover the same cells,
    which wastes effort and increases mission time.

    Args:
        ll_intents: Dict mapping agent_id -> LLIntent

    Returns:
        Overlap penalty (in arbitrary units, higher = worse)
    """
    if not ll_intents or len(ll_intents) < 2:
        return 0.0

    # Build cell coverage map: cell -> set of agents
    cell_coverage: Dict[Tuple[int, int], set] = {}

    for agent_id, ll_intent in ll_intents.items():
        if ll_intent is None or not hasattr(ll_intent, "footprint_sequence"):
            continue

        for footprint in ll_intent.footprint_sequence:
            if footprint is None or len(footprint) != 4:
                continue
            imin, imax, jmin, jmax = footprint
            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    if (i, j) not in cell_coverage:
                        cell_coverage[(i, j)] = set()
                    cell_coverage[(i, j)].add(agent_id)

    # Count overlapping cells (covered by >1 agent)
    overlap_count = sum(1 for agents in cell_coverage.values() if len(agents) > 1)

    # Penalty scales with amount of overlap
    # Each overlapping cell adds to mission time
    return float(overlap_count) / max(1, len(cell_coverage))


def evaluate_region_coverage(
    hl_intents: Dict[int, Any],
    belief: np.ndarray,
    grid_info: Any,
) -> float:
    """
    Evaluate how well agent HL intents cover high-uncertainty regions.

    Args:
        hl_intents: Dict of agent_id -> HLIntent
        belief: Current belief map
        grid_info: Grid metadata

    Returns:
        Coverage score (higher = better coverage of uncertain regions)
    """
    # TODO: Implement region coverage evaluation
    # For now, return 0 (no bonus for good region allocation)
    return 0.0


def compute_coordination_penalty(
    ll_intents: Dict[int, Any],
    hl_intents: Dict[int, Any],
) -> float:
    """
    Compute penalty for overlapping agent plans.

    Args:
        ll_intents: Dict of agent_id -> LLIntent
        hl_intents: Dict of agent_id -> HLIntent

    Returns:
        Penalty value (higher = more overlap/conflict)
    """
    # TODO: Implement overlap/conflict detection
    # For now, return 0 (no coordination penalty)
    return 0.0


# =============================================================================
# Utility functions for g2 computation
# =============================================================================


def get_region_uncertainty(
    belief: np.ndarray,
    region_bounds: Tuple[int, int, int, int],
) -> float:
    """
    Get total uncertainty in a specific region.

    Args:
        belief: Belief map
        region_bounds: (imin, imax, jmin, jmax) region boundaries

    Returns:
        Uncertainty in the region
    """
    imin, imax, jmin, jmax = region_bounds
    region_belief = belief[imin:imax, jmin:jmax, :]
    return compute_uncertain_area(region_belief)


def compute_region_overlap(
    region1: Tuple[int, int, int, int],
    region2: Tuple[int, int, int, int],
) -> float:
    """
    Compute overlap between two regions.

    Args:
        region1: (imin, imax, jmin, jmax)
        region2: (imin, imax, jmin, jmax)

    Returns:
        Overlap area (in cells)
    """
    i1_min, i1_max, j1_min, j1_max = region1
    i2_min, i2_max, j2_min, j2_max = region2

    # Compute intersection
    i_overlap = max(0, min(i1_max, i2_max) - max(i1_min, i2_min))
    j_overlap = max(0, min(j1_max, j2_max) - max(j1_min, j2_min))

    return float(i_overlap * j_overlap)

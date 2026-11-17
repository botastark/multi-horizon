"""
Hierarchical Decentralized MCTS Planner with Shared Beliefs and Intents

Implements the Dec-MCTS framework from "Multi-Horizon Multi-Agent Planning Using
Decentralised Monte Carlo Tree Search" with:

1. Core Objects:
   - Belief: Shared occupancy/uncertainty maps updated via LBP fusion
   - Intent: Current best solution (state sequence) from each planner level

2. Two-Level Planning:
   - LLP (Low-Level Planner): Short-horizon, detailed motion planning, IG-based reward
     * Option A: Random rollout sampling (default, use_mcts_llp=False)
     * Option B: MCTS tree search (use_mcts_llp=True)
   - HLP (High-Level Planner): Long-horizon, region/cluster allocation (MCTS-based)

3. Intent Sharing:
   - LL-intent: Detailed short-horizon motion plan (primitive actions + footprints)
   - HL-intent: Long-horizon region sequence the agent intends to cover

4. Reward Decomposition:
   - g = g1(LL intents) + g2(all intents)
   - g1: Short-horizon information gain of the LL plan (bits)
   - g2: Remaining coverage effort in agent-steps until every cell has been
     observed at least once (see g2_evaluator.py). Lower g2 is better, so the
     LLP adds  g2_weight * (steps saved by the trajectory)  and the HLP
     maximises the steps saved by committing to a region sequence.

5. Timing:
   - Everything is measured in simulation steps, never wall-clock seconds:
     intent age/staleness (D-UCT discount), HLP replan interval.
   - Intents are exchanged only between agents within communication range and
     only in news-sharing modes (IG_BS/IG_BM/IGd_BS/IGd_BM); see IntentBus.

Baseline Comparison:
   1. Greedy IG: Single-step lookahead, no planning
   2. Dec-MCTS: Single-level MCTS, no hierarchical structure
   3. MH Dec-MCTS (use_mcts_llp=False): HLP uses MCTS, LLP uses random rollouts
   4. MH Dec-MCTS Both (use_mcts_llp=True): Both HLP and LLP use MCTS tree search
"""

import time
import threading
import queue
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import math
import numpy as np
from collections import defaultdict
import copy
import dataclasses
from helper import cH as _cH, get_sensor_params as _get_sensor_params_fn
from g2_evaluator import (
    CoverageStepModel,
    auto_g2_weight,
    camera_step_geometry,
    uncovered_from_belief,
)
from d_uct import DUCTStats, d_uct_score

logger = logging.getLogger(__name__)


# =============================================================================
# Detailed Logging for Hierarchical Dec-MCTS
# =============================================================================


def log_planning_decision(
    agent_id: int,
    step: int,
    llp_action_scores: Dict[str, float],
    hlp_region_scores: Dict[int, float],
    selected_action: str,
    target_region: Optional[int],
    intents_received: Dict[str, Any],
):
    """
    Log detailed planning decision with LLP scores and HLP scores.

    Args:
        agent_id: Agent ID
        step: Current step number
        llp_action_scores: LLP action scores (IG-only, g1)
        hlp_region_scores: HLP region scores
        selected_action: The action selected
        target_region: HLP target region
        intents_received: Summary of teammate intents received
    """
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"[Agent {agent_id}] PLANNING DECISION")
    logger.info(f"{'='*60}")

    # LLP Action Scores
    logger.info("")
    logger.info("LLP ACTION SCORES (Information Gain):")
    sorted_actions = sorted(llp_action_scores.items(), key=lambda x: x[1], reverse=True)
    for action, score in sorted_actions:
        marker = " <--" if action == selected_action else ""
        logger.info(f"  {action:8s}: {score:10.2f}{marker}")

    # HLP Region Scores
    logger.info("")
    logger.info("HLP REGION SCORES:")
    sorted_regions = sorted(
        hlp_region_scores.items(), key=lambda x: x[1], reverse=True
    )[:5]
    for region_id, score in sorted_regions:
        marker = " <-- TARGET" if region_id == target_region else ""
        logger.info(f"  Region {region_id:2d}: {score:.4f}{marker}")

    logger.info("")
    logger.info(f"SELECTED ACTION: {selected_action}")

    # Intents received
    if intents_received:
        logger.info("")
        logger.info("INTENTS FROM TEAMMATES:")
        for key, value in intents_received.items():
            logger.info(f"  {key}: {value}")

    logger.info(f"{'='*60}")


def log_intent_sharing(
    agent_id: int,
    ll_intent_summary: Dict[str, Any],
    hl_intent_summary: Dict[str, Any],
):
    """
    Log intent sharing details.

    Args:
        agent_id: Agent ID
        ll_intent_summary: Summary of LL intent being broadcast
        hl_intent_summary: Summary of HL intent being broadcast
    """
    logger.info("")
    logger.info(f"[Agent {agent_id}] BROADCASTING INTENTS:")
    logger.info(
        f"  LL Intent: actions={ll_intent_summary.get('actions', [])[:3]}..., "
        f"total_ig={ll_intent_summary.get('total_ig', 0):.2f}"
    )
    logger.info(
        f"  HL Intent: target_region={hl_intent_summary.get('target_region')}, "
        f"region_sequence={hl_intent_summary.get('region_sequence', [])}"
    )


# =============================================================================
# Intent Data Structures
# =============================================================================


@dataclass
class LLIntent:
    """
    Low-Level Intent: Short-horizon motion plan.

    Contains the agent's planned sequence of primitive actions over the
    short horizon, including predicted footprints and IG at each step.
    """

    agent_id: int
    # Sequence of planned actions (e.g., ['front', 'front', 'right', 'down'])
    action_sequence: List[str] = field(default_factory=list)
    # Predicted state sequence [(x, y, altitude), ...] in WORLD metres;
    # element 0 is the state the plan starts from.
    state_sequence: List[Tuple[float, float, float]] = field(default_factory=list)
    # Grid footprint observed after each action: [(imin, imax, jmin, jmax), ...]
    footprint_sequence: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Expected IG at each step
    ig_sequence: List[float] = field(default_factory=list)
    # Total expected IG for this plan
    total_expected_ig: float = 0.0
    # Simulation step at which the plan was made (its first action is executed
    # during this step). All ages are measured in steps.
    step: int = 0
    # Planning horizon (number of steps)
    horizon: int = 5
    # Value of this plan (g1 + g2_weight * steps saved)
    value: float = 0.0
    # Number of MCTS iterations behind this intent
    iterations: int = 0

    def age(self, current_step: int) -> int:
        """Steps elapsed since this intent was made."""
        return max(0, int(current_step) - int(self.step))

    def is_stale(self, current_step: int, max_age_steps: float) -> bool:
        """True when the intent is older than max_age_steps."""
        return self.age(current_step) > max_age_steps

    def staleness_discount(
        self, current_step: int, decay_factor: float = 0.9, threshold_steps: float = 2.0
    ) -> float:
        """
        D-UCT weight for an intent: decay_factor ** (age / threshold_steps).

        1.0 for an intent made this step, smaller for older intents.
        """
        return decay_factor ** (self.age(current_step) / max(threshold_steps, 1e-9))

    def get_covered_cells(self) -> Set[Tuple[int, int]]:
        """Get set of (row, col) cells this plan will cover."""
        cells = set()
        for fp in self.footprint_sequence:
            imin, imax, jmin, jmax = fp
            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    cells.add((i, j))
        return cells


@dataclass
class HLIntent:
    """
    High-Level Intent: Long-horizon region/cluster plan.

    Contains the agent's planned sequence of regions to cover,
    including estimated completion times and priority scores.
    """

    agent_id: int
    # Sequence of region IDs the agent plans to cover, in order
    region_sequence: List[int] = field(default_factory=list)
    # Cumulative steps (from now) until each region is REACHED
    eta_sequence: List[float] = field(default_factory=list)
    # Cumulative steps (from now) until each region is fully OBSERVED
    completion_sequence: List[float] = field(default_factory=list)
    # Steps spent on each region (reach + scan)
    score_sequence: List[float] = field(default_factory=list)
    # Current target region
    current_target_region: Optional[int] = None
    # Center of current target region, grid (row, col)
    target_center: Optional[Tuple[float, float]] = None
    # Simulation step at which the agent last asserted this plan
    step: int = 0
    # Planning horizon (number of regions)
    horizon: int = 3
    # Agent-steps saved by this plan vs. claiming nothing (higher is better)
    value: float = 0.0
    # Number of MCTS iterations
    iterations: int = 0

    def age(self, current_step: int) -> int:
        return max(0, int(current_step) - int(self.step))

    def is_stale(self, current_step: int, max_age_steps: float) -> bool:
        return self.age(current_step) > max_age_steps

    def staleness_discount(
        self, current_step: int, decay_factor: float = 0.9, threshold_steps: float = 2.0
    ) -> float:
        return decay_factor ** (self.age(current_step) / max(threshold_steps, 1e-9))

    def get_target_regions(self) -> Set[int]:
        """Get set of regions this agent intends to cover."""
        return set(self.region_sequence)


@dataclass
class BeliefUpdate:
    """
    Belief update message for sharing local observations.
    """

    agent_id: int
    # Updated cell probabilities: {(row, col): probability}
    cell_updates: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # Observed region mask (compressed)
    observed_mask_bounds: Optional[Tuple[int, int, int, int]] = None
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    # Step number
    step: int = 0


# =============================================================================
# Intent Bus: Asynchronous Communication Channel
# =============================================================================


class IntentBus:
    """
    Communication bus for sharing intents between agents.

    Delivery model (synchronous simulation):
    - When an agent broadcasts, the intent is delivered to every other agent
      that is within communication range of the sender at that moment
      (`neighbor_fn(sender_id)`), and stored in that receiver's inbox.
    - A receiver keeps the last intent it got from each teammate. If the
      teammate later moves out of range, the old intent stays in the inbox and
      simply ages; planners down-weight / drop it by its age in steps.
    - With `enabled=False` (IG / IGd modes: no information sharing) nothing is
      ever delivered.
    """

    def __init__(
        self,
        num_agents: int,
        max_history: int = 10,
        neighbor_fn=None,
        enabled: bool = True,
    ):
        """
        Initialize intent bus.

        Args:
            num_agents: Number of agents in the system
            max_history: Number of historical intents to keep per agent
            neighbor_fn: callable(agent_id) -> list of agent ids currently in
                communication range. None = everyone is always in range.
            enabled: False disables all delivery (no intent sharing).
        """
        self.num_agents = num_agents
        self.max_history = max_history
        self.neighbor_fn = neighbor_fn
        self.enabled = enabled
        # Deferred delivery (simulated asynchrony): broadcasts are queued and
        # delivered together by flush() at the end of a planning slice.
        self.deferred = False
        self._pending: List[Tuple[str, Any]] = []

        # Latest intent broadcast by each agent (sender side; for logging/viz)
        self._ll_intents: Dict[int, LLIntent] = {}
        self._hl_intents: Dict[int, HLIntent] = {}

        # Receiver inboxes: receiver_id -> {sender_id: last intent received}
        self._ll_inbox: Dict[int, Dict[int, LLIntent]] = defaultdict(dict)
        self._hl_inbox: Dict[int, Dict[int, HLIntent]] = defaultdict(dict)

        # Intent history for temporal reasoning
        self._ll_history: Dict[int, List[LLIntent]] = defaultdict(list)
        self._hl_history: Dict[int, List[HLIntent]] = defaultdict(list)

        # Belief updates queue
        self._belief_updates: queue.Queue = queue.Queue(maxsize=100)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "ll_broadcasts": 0,
            "hl_broadcasts": 0,
            "belief_broadcasts": 0,
        }

    def _receivers(self, sender_id: int) -> List[int]:
        """Agents that receive a broadcast from sender_id right now."""
        if not self.enabled:
            return []
        if self.neighbor_fn is None:
            return [a for a in range(self.num_agents) if a != sender_id]
        return [a for a in self.neighbor_fn(sender_id) if a != sender_id]

    def flush(self) -> None:
        """Deliver all queued broadcasts (deferred mode)."""
        with self._lock:
            pending, self._pending = self._pending, []
            deferred, self.deferred = self.deferred, False
        try:
            for kind, intent in pending:
                if kind == "ll":
                    self.broadcast_ll_intent(intent)
                else:
                    self.broadcast_hl_intent(intent)
        finally:
            self.deferred = deferred

    def broadcast_ll_intent(self, intent: LLIntent) -> None:
        """Broadcast a low-level intent to agents currently in range."""
        if self.deferred:
            with self._lock:
                self._pending.append(("ll", intent))
            return
        with self._lock:
            self._ll_intents[intent.agent_id] = intent
            self._ll_history[intent.agent_id].append(intent)
            if len(self._ll_history[intent.agent_id]) > self.max_history:
                self._ll_history[intent.agent_id].pop(0)
            for receiver in self._receivers(intent.agent_id):
                self._ll_inbox[receiver][intent.agent_id] = intent
            self._stats["ll_broadcasts"] += 1

    def broadcast_hl_intent(self, intent: HLIntent) -> None:
        """Broadcast a high-level intent to agents currently in range."""
        if self.deferred:
            with self._lock:
                self._pending.append(("hl", intent))
            return
        with self._lock:
            self._hl_intents[intent.agent_id] = intent
            self._hl_history[intent.agent_id].append(intent)
            if len(self._hl_history[intent.agent_id]) > self.max_history:
                self._hl_history[intent.agent_id].pop(0)
            for receiver in self._receivers(intent.agent_id):
                self._hl_inbox[receiver][intent.agent_id] = intent
            self._stats["hl_broadcasts"] += 1

    def broadcast_belief_update(self, update: BeliefUpdate) -> None:
        """Broadcast belief update (non-blocking, drops if full)."""
        try:
            self._belief_updates.put_nowait(update)
            self._stats["belief_broadcasts"] += 1
        except queue.Full:
            pass  # Drop old updates if queue is full

    def get_all_ll_intents(
        self, exclude_agent: Optional[int] = None
    ) -> Dict[int, LLIntent]:
        """Get latest LL intents from all agents (optionally excluding one)."""
        with self._lock:
            if exclude_agent is not None:
                return {k: v for k, v in self._ll_intents.items() if k != exclude_agent}
            return dict(self._ll_intents)

    def get_all_hl_intents(
        self, exclude_agent: Optional[int] = None
    ) -> Dict[int, HLIntent]:
        """Get latest HL intents from all agents (optionally excluding one)."""
        with self._lock:
            if exclude_agent is not None:
                return {k: v for k, v in self._hl_intents.items() if k != exclude_agent}
            return dict(self._hl_intents)

    def get_teammate_ll_intents(self, agent_id: int) -> Dict[int, LLIntent]:
        """LL intents this agent has received (last one per teammate, may be old)."""
        with self._lock:
            return dict(self._ll_inbox[agent_id])

    def get_teammate_hl_intents(self, agent_id: int) -> Dict[int, HLIntent]:
        """HL intents this agent has received (last one per teammate, may be old)."""
        with self._lock:
            return dict(self._hl_inbox[agent_id])

    def get_pending_belief_updates(self) -> List[BeliefUpdate]:
        """Get all pending belief updates (non-blocking)."""
        updates = []
        while True:
            try:
                updates.append(self._belief_updates.get_nowait())
            except queue.Empty:
                break
        return updates

    def get_statistics(self) -> Dict[str, int]:
        """Get bus statistics."""
        with self._lock:
            return dict(self._stats)


# =============================================================================
# Low-Level Planner (LLP) with Dec-MCTS
# =============================================================================


class LowLevelPlanner:
    """
    Low-Level Planner: short-horizon motion planning for one agent.

    Reward of a candidate action sequence (see _simulate_trajectory):

        r = g1 + g2_weight * (g2_baseline - g2_after)

    g1  discounted sensor-aware information gain of the footprints it visits,
        with cells that teammates' LL intents will observe down-weighted
        (_compute_teammate_coverage_mask)
    g2  remaining agent-steps until every cell is observed at least once,
        given this trajectory, the agent's own HL region sequence and the
        teammates' LL/HL intents (g2_evaluator.CoverageStepModel).
        g2_baseline is g2 if the agent stayed where it is, so the second term is
        "steps saved" by the trajectory: moving toward the next HL region and
        closing small uncovered spots both save steps.
    """

    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        conf_dict: Optional[Dict] = None,
        horizon: int = 5,
        num_iterations: int = 100,
        ucb_c: float = 1.41,
        discount: float = 0.95,
        intent_discount: float = 0.8,  # How fast teammate-plan suppression fades with plan depth
        use_mcts_llp: bool = False,  # Use MCTS for LLP instead of random rollouts
        use_g2: bool = False,  # Add the g2 steps-saved term to the reward
        g2_weight: Optional[float] = None,  # bits per step saved; None = auto
        d_uct_decay: float = 0.9,
        d_uct_threshold_steps: float = 2.0,
        d_uct_gamma: float = 0.99,
        region_tile_size: Tuple[int, int] = (100, 100),
        step_cells: Optional[float] = None,
        fp_half: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize LLP.

        Args:
            agent_id: This agent's ID
            camera: UAV camera model
            grid_info: Grid information
            conf_dict: Sensor model dict {altitude: (s0, s1)}; if None uses analytic formula
            horizon: Planning horizon (steps)
            num_iterations: Rollouts (or MCTS iterations) per planning cycle
            ucb_c: UCB exploration constant (MCTS LLP only)
            discount: Reward discount factor per step of the plan
            intent_discount: Weight of a teammate's k-th remaining planned
                footprint is intent_discount**k (k=0 is its next footprint)
            use_mcts_llp: If True, use MCTS tree search for LLP; if False, use random rollouts
            use_g2: If True, add g2_weight * (steps saved) to the reward
            g2_weight: Information (bits) one saved step is worth. None = the
                IG of one fresh step at minimum altitude (see _auto_g2_weight)
            d_uct_decay, d_uct_threshold_steps: teammate intents are weighted by
                d_uct_decay ** (age_steps / d_uct_threshold_steps) and ignored when
                older than 2 * d_uct_threshold_steps
            region_tile_size: HLP region tile size (cells) used by g2
            step_cells: lateral move length in grid cells
            fp_half: half-width (cells) of the footprint at minimum altitude
        """
        self.agent_id = agent_id
        self.camera = camera
        self.grid_info = grid_info
        self.conf_dict = conf_dict
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.ucb_c = ucb_c
        self.discount = discount
        self.intent_discount = intent_discount
        self.use_mcts_llp = use_mcts_llp
        self.use_g2 = use_g2
        self.d_uct_decay = d_uct_decay
        self.d_uct_threshold_steps = d_uct_threshold_steps
        self.d_uct_gamma = d_uct_gamma  # tree discount (paper Eq. 4-5)
        self.max_intent_age_steps = 2.0 * d_uct_threshold_steps
        self.region_tile_size = tuple(region_tile_size)

        # Anytime search state of the current round (one simulation step)
        self._round_state: Optional[Tuple[float, float, float]] = None
        self._tree: Dict[Tuple, Dict[str, DUCTStats]] = {}
        self._tree_t = 0
        self._best_actions: Optional[List[str]] = None
        self._coverage_discount: Optional[np.ndarray] = None
        self._mcts_action_values: Dict[str, float] = {}

        geom_step, geom_half = camera_step_geometry(camera)
        self.step_cells = float(step_cells if step_cells is not None else geom_step)
        self.fp_half = float(fp_half if fp_half is not None else geom_half)
        self.g2_weight = (
            float(g2_weight) if g2_weight is not None else self._auto_g2_weight()
        )

        # Current belief map and knowledge of which cells were observed
        self.belief: Optional[np.ndarray] = None
        self._uncovered: Optional[np.ndarray] = None
        self.current_step: int = 0

        # Teammate intents
        self._teammate_ll_intents: Dict[int, LLIntent] = {}
        self._teammate_hl_intents: Dict[int, HLIntent] = {}

        # HLP guidance (from own agent's HLP)
        self._hl_guidance: Optional[HLIntent] = None

        # Current intent
        self.current_intent: Optional[LLIntent] = None

        # Statistics
        self._stats = {
            "plans_generated": 0,
            "total_iterations": 0,
            "intent_updates_received": 0,
        }

        # Actions
        self.actions = ["front", "back", "left", "right", "up", "down", "hover"]

        # RNG for reproducible randomization
        self._rng = np.random.default_rng(seed)

        # IG cache: key -> ig_array; invalidated via version counter (avoids tobytes() hashing)
        self._ig_cache: Dict[Any, np.ndarray] = {}
        self._belief_version: int = 0
        self._ig_cache_belief_version: int = -1

        # Cache static camera properties — avoids repeated attribute lookups in hot loops
        self._grid_length: float = camera.grid.length
        self._grid_center = camera.grid.center
        self._tan_half_fov: float = math.tan(np.deg2rad(camera.fov) / 2)
        self._cam_x_range = camera.x_range
        self._cam_y_range = camera.y_range
        self._cam_h_step: float = camera.h_step
        self._cam_xy_step: float = camera.xy_step
        self._cam_h_range = camera.h_range

        # Admissible-actions cache (grid never changes, so keyed on rounded pos+alt)
        self._admissible_cache: Dict[Tuple, List[str]] = {}

        # Per-plan g2 context (built in plan(), see _prepare_g2_context)
        self._g2_ctx: Optional[Dict[str, Any]] = None
        self.last_g2_steps: Optional[float] = None

    def _auto_g2_weight(self) -> float:
        """IG of one fresh move at minimum altitude (see g2_evaluator.auto_g2_weight)."""
        return auto_g2_weight(self.camera, self.conf_dict)

    def update_belief(self, belief: np.ndarray) -> None:
        """Update local belief map."""
        self.belief = np.asarray(belief)
        self._belief_version += 1  # Invalidate IG cache cheaply

    def update_coverage(self, uncovered: np.ndarray) -> None:
        """Cells (H, W bool) not yet known to be observed at least once."""
        self._uncovered = np.asarray(uncovered, dtype=bool)

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def update_teammate_intents(
        self,
        ll_intents: Dict[int, LLIntent],
        hl_intents: Dict[int, HLIntent],
    ) -> None:
        """Store the teammate intents received so far (may include old ones)."""
        self._teammate_ll_intents = ll_intents
        self._teammate_hl_intents = hl_intents
        self._stats["intent_updates_received"] += 1

    def update_hl_guidance(self, hl_intent: HLIntent) -> None:
        """Update guidance from own HLP."""
        self._hl_guidance = hl_intent

    def _get_sensor_params(self, altitude: float) -> Tuple[float, float]:
        """Return (s0, s1) sensor noise for altitude. See helper.get_sensor_params."""
        return _get_sensor_params_fn(altitude, self.conf_dict)

    def _fresh_teammate_ll_intents(self) -> Dict[int, LLIntent]:
        return {
            tid: intent
            for tid, intent in self._teammate_ll_intents.items()
            if not intent.is_stale(self.current_step, self.max_intent_age_steps)
        }

    def _fresh_teammate_hl_intents(self) -> Dict[int, HLIntent]:
        return {
            tid: intent
            for tid, intent in self._teammate_hl_intents.items()
            if not intent.is_stale(self.current_step, self.max_intent_age_steps)
        }

    def _compute_teammate_coverage_mask(self) -> np.ndarray:
        """
        Per-cell IG multiplier from teammates' LL intents.

        A teammate intent made `age` steps ago has already executed its first
        `age` footprints (those observations are already in the belief), so the
        k-th REMAINING footprint (k = 0 is the teammate's next one) gets weight

            w_k = intent_discount**k * staleness,  staleness = decay**(age/threshold)

        and cells inside it have their IG multiplied by (1 - w_k). The
        teammate's next footprint from a fresh intent is therefore fully
        suppressed; footprints further into its plan (less certain) less so.
        """
        H, W = self.grid_info.shape
        coverage_discount: Optional[np.ndarray] = None

        for ll_intent in self._fresh_teammate_ll_intents().values():
            age = ll_intent.age(self.current_step)
            staleness_factor = ll_intent.staleness_discount(
                self.current_step, self.d_uct_decay, self.d_uct_threshold_steps
            )
            for idx, fp in enumerate(ll_intent.footprint_sequence):
                k = idx - age
                if k < 0:
                    continue  # already executed
                weight = (self.intent_discount**k) * staleness_factor
                if weight <= 0.0:
                    continue
                imin, imax, jmin, jmax = fp
                imin = max(0, min(imin, H))
                imax = max(0, min(imax, H))
                jmin = max(0, min(jmin, W))
                jmax = max(0, min(jmax, W))
                if imax <= imin or jmax <= jmin:
                    continue
                if coverage_discount is None:
                    coverage_discount = np.ones((H, W), dtype=np.float32)
                coverage_discount[imin:imax, jmin:jmax] *= 1.0 - min(1.0, weight)

        if coverage_discount is None:
            return np.ones((H, W), dtype=np.float32)

        return coverage_discount

    def _compute_ig(
        self,
        position: Tuple[float, float],
        altitude: float,
        coverage_discount: np.ndarray,
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Compute Information Gain for a position, considering teammate intents.

        Returns (ig_value, (imin, imax, jmin, jmax)) so callers can reuse the
        footprint bounds without a second computation.  Returns (0.0, None) on
        any failure path.

        Uses cached camera properties to avoid repeated attribute lookups.
        """
        if self.belief is None:
            return 0.0, None

        # Cache key for position/altitude
        cache_key = (round(position[0], 1), round(position[1], 1), round(altitude, 1))

        # Compute footprint using cached camera properties
        try:
            x_dist = round(altitude * self._tan_half_fov / self._grid_length) * self._grid_length
            y_dist = x_dist  # square FOV

            x_min = max(position[0] - x_dist, self._cam_x_range[0])
            x_max = min(position[0] + x_dist, self._cam_x_range[1])
            y_min = max(position[1] - y_dist, self._cam_y_range[0])
            y_max = min(position[1] + y_dist, self._cam_y_range[1])

            if x_max - x_min <= 0 or y_max - y_min <= 0:
                return 0.0, None

            # Convert to grid indices
            i_max, j_min = self.camera.convert_xy_ij(x_min, y_min, self._grid_center)
            i_min, j_max = self.camera.convert_xy_ij(x_max, y_max, self._grid_center)
            imin, imax, jmin, jmax = i_min, i_max, j_min, j_max
        except Exception:
            return 0.0, None

        H, W = self.belief.shape[:2]
        imin = max(0, min(imin, H))
        imax = max(0, min(imax, H))
        jmin = max(0, min(jmin, W))
        jmax = max(0, min(jmax, W))

        if imax <= imin or jmax <= jmin:
            return 0.0, None

        footprint = (imin, imax, jmin, jmax)

        # Check cache — invalidate only when belief version changed (O(1), no tobytes())
        if self._belief_version != self._ig_cache_belief_version:
            self._ig_cache = {}
            self._ig_cache_belief_version = self._belief_version

        # Return cached IG array if available (cache key includes altitude for sensor-aware IG)
        if cache_key in self._ig_cache:
            ig_array = self._ig_cache[cache_key]
        else:
            # Extract belief probabilities (occupied channel)
            if self.belief.ndim == 3:
                belief_slice = self.belief[imin:imax, jmin:jmax, 1]
            else:
                belief_slice = self.belief[imin:imax, jmin:jmax]

            # Sensor-aware IG = H(prior) - E[H(posterior | observation)]
            # Uses sensor noise sigma(altitude): lower altitude → smaller sigma → more IG per cell
            s0, s1 = self._get_sensor_params(altitude)

            # Compute per-cell entropy of prior
            eps = 1e-10
            p = np.clip(belief_slice, eps, 1 - eps)
            h_prior = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

            # Compute conditional entropy using imported _cH (same as planner.py / greedy_ig)
            h_cond = _cH(belief_slice, s0, s1)

            # IG per cell; clamp small negatives from floating-point error to 0
            ig_array = np.maximum(h_prior - h_cond, 0.0)

            # Cache the IG array so per-cell discounting works
            self._ig_cache[cache_key] = ig_array

        # Apply teammate coverage discount per-cell (spatially targeted suppression)
        discount_slice = coverage_discount[imin:imax, jmin:jmax]
        discounted_ig = ig_array * discount_slice

        return float(np.sum(discounted_ig)), footprint

    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self.camera.convert_xy_ij(x, y, self._grid_center)
        return float(i), float(j)

    def _prepare_g2_context(self, current_state: Tuple[float, float, float]) -> None:
        """
        Build the per-plan g2 inputs that do not depend on the candidate
        trajectory: coverage model after the teammates' fresh LL plans, the
        teammates' (start, HL sequence) and the baseline g2 if this agent does
        not move.
        """
        uncovered = (
            self._uncovered
            if self._uncovered is not None
            else uncovered_from_belief(self.belief)
        )
        base = CoverageStepModel(
            uncovered, self.region_tile_size, self.step_cells, self.fp_half
        )
        fresh_ll = self._fresh_teammate_ll_intents()
        fresh_hl = self._fresh_teammate_hl_intents()
        teammate_fps = [
            fp for intent in fresh_ll.values() for fp in intent.footprint_sequence
        ]
        model = base.with_footprints(teammate_fps)

        teammates: Dict[int, Tuple[Optional[Tuple[float, float]], List[int]]] = {}
        for tid in set(fresh_ll) | set(fresh_hl):
            start = None
            ll = fresh_ll.get(tid)
            if ll is not None and ll.state_sequence:
                x, y, _ = ll.state_sequence[-1]
                start = self._world_to_grid(x, y)
            hl = fresh_hl.get(tid)
            seq = list(hl.region_sequence) if hl is not None else []
            teammates[tid] = (start, seq)

        own_seq = (
            list(self._hl_guidance.region_sequence) if self._hl_guidance else []
        )
        agents = dict(teammates)
        agents[self.agent_id] = (
            self._world_to_grid(current_state[0], current_state[1]),
            own_seq,
        )
        self._g2_ctx = {
            "model": model,
            "teammates": teammates,
            "own_seq": own_seq,
            "baseline": model.team_steps(agents),
        }

    def _g2_steps(
        self,
        state_sequence: List[Tuple[float, float, float]],
        footprint_sequence: List[Tuple[int, int, int, int]],
    ) -> float:
        """g2 (agent-steps still needed) after executing this trajectory."""
        ctx = self._g2_ctx
        model = ctx["model"].with_footprints(footprint_sequence)
        end = state_sequence[-1]
        agents = dict(ctx["teammates"])
        agents[self.agent_id] = (self._world_to_grid(end[0], end[1]), ctx["own_seq"])
        return model.team_steps(agents)

    def _evaluate_single_action(
        self,
        current_state: Tuple[float, float, float],
        action: str,
        coverage_discount: np.ndarray,
    ) -> float:
        """
        Evaluate a single action for logging purposes.

        Returns:
            ig_score (IG-only, no alignment bonus)
        """
        # Get current state
        current_pos = current_state[:2]
        current_alt = current_state[2]

        # Temporarily set camera state
        org = self.camera.get_x()
        original_pos, original_alt = org.position, org.altitude

        self.camera.set_position(current_pos)
        self.camera.set_altitude(current_alt)

        # Get next state
        future_state = self.camera.x_future(action)
        if future_state is None:
            self.camera.set_position(original_pos)
            self.camera.set_altitude(original_alt)
            return 0.0

        next_pos = future_state[0]
        next_alt = future_state[1]

        # Compute IG (footprint not needed here)
        ig, _ = self._compute_ig(next_pos, next_alt, coverage_discount)

        # Restore camera state
        self.camera.set_position(original_pos)
        self.camera.set_altitude(original_alt)

        return ig

    def _simulate_trajectory(
        self,
        start_state: Tuple[float, float, float],
        actions: List[str],
        coverage_discount: np.ndarray,
    ) -> Tuple[
        float,
        List[Tuple[float, float, float]],
        List[Tuple[int, int, int, int]],
        List[float],
    ]:
        """
        Simulate a trajectory and compute its reward.

            r = g1 + g2_weight * (g2_baseline - g2_after)      (second term only if use_g2)

        - g1: sum over steps of discount**k * IG(footprint_k), IG evaluated on
          the current belief (the belief is not updated along the rollout, so a
          footprint visited twice is counted twice) and down-weighted where
          teammates' LL plans will look.
        - g2_after: agent-steps still needed until every cell is observed at
          least once, if this trajectory is executed (g2_evaluator). g2 is a
          cost, so the reward uses the steps SAVED relative to not moving.

        Returns:
            (total_reward, state_sequence, footprint_sequence, ig_sequence)
        """
        # Compute g1: immediate IG-based reward
        g1_reward = 0.0
        state_sequence = [start_state]
        footprint_sequence = []
        ig_sequence = []

        current_pos = start_state[:2]
        current_alt = start_state[2]

        # NO camera state mutations - compute state transitions directly
        for step_idx, action in enumerate(actions):
            # Get next state via direct computation (avoid camera.x_future mutations)
            future_state = self._apply_action(current_pos, current_alt, action)
            if future_state is None:
                break

            next_pos, next_alt = future_state

            # Compute IG and footprint in one call — footprint reused directly,
            # no second computation needed
            ig, fp = self._compute_ig(next_pos, next_alt, coverage_discount)
            step_reward = (self.discount**step_idx) * ig
            g1_reward += step_reward

            # Record trajectory
            state_sequence.append((next_pos[0], next_pos[1], next_alt))
            ig_sequence.append(ig)
            footprint_sequence.append(fp if fp is not None else (0, 0, 0, 0))

            # Update position for next step (no camera mutation)
            current_pos = next_pos
            current_alt = next_alt

        # g2: steps saved toward full coverage (cost, so baseline - after)
        g2_bonus = 0.0
        if self.use_g2 and self._g2_ctx is not None:
            steps_after = self._g2_steps(state_sequence, footprint_sequence)
            g2_bonus = self.g2_weight * (self._g2_ctx["baseline"] - steps_after)

        total_reward = g1_reward + g2_bonus

        return total_reward, state_sequence, footprint_sequence, ig_sequence

    def _apply_action(
        self, pos: Tuple[float, float], alt: float, action: str
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Apply action to state without camera mutations (uses cached camera properties)."""
        if action == "up" and alt + self._cam_h_step <= self._cam_h_range[1]:
            return pos, alt + self._cam_h_step
        elif action == "down" and alt - self._cam_h_step >= self._cam_h_range[0]:
            return pos, alt - self._cam_h_step
        elif action == "front" and pos[1] + self._cam_xy_step <= self._cam_y_range[1]:
            return (pos[0], pos[1] + self._cam_xy_step), alt
        elif action == "back" and pos[1] - self._cam_xy_step >= self._cam_y_range[0]:
            return (pos[0], pos[1] - self._cam_xy_step), alt
        elif action == "right" and pos[0] + self._cam_xy_step <= self._cam_x_range[1]:
            return (pos[0] + self._cam_xy_step, pos[1]), alt
        elif action == "left" and pos[0] - self._cam_xy_step >= self._cam_x_range[0]:
            return (pos[0] - self._cam_xy_step, pos[1]), alt
        elif action == "hover":
            return pos, alt
        else:
            return None

    def _admissible_actions(self, pos: Tuple[float, float], alt: float) -> List[str]:
        """Return admissible actions for (pos, alt) with a position-keyed cache.

        Grid boundaries never change, so results are valid for the planner lifetime.
        Avoids 7× _apply_action calls on every repeated (pos, alt) during rollouts.
        """
        key = (round(pos[0], 1), round(pos[1], 1), round(alt, 1))
        cached = self._admissible_cache.get(key)
        if cached is not None:
            return cached
        result = [
            a for a in self.actions if self._apply_action(pos, alt, a) is not None
        ] or ["hover"]
        self._admissible_cache[key] = result
        return result

    # ------------------------------------------------------- anytime search
    def begin_round(self, current_state: Tuple[float, float, float]) -> None:
        """
        Start a planning round (one simulation step) from current_state.

        The MCTS tree is restarted (receding horizon: the first action of the
        previous plan has been executed). Within the round, run_iterations()
        can be called several times while intents change in between; the tree
        then uses D-UCT discounting for the drifting reward.
        """
        self._round_state = (
            float(current_state[0]),
            float(current_state[1]),
            float(current_state[2]),
        )
        self._tree = {}
        self._tree_t = 0
        self._best_actions = None
        self._mcts_action_values = {action: float("-inf") for action in self.actions}

    def _refresh_context(self) -> None:
        """Re-read teammate intents / HL guidance into the reward (start of each slice)."""
        self._coverage_discount = self._compute_teammate_coverage_mask()
        self._g2_ctx = None
        if self.use_g2:
            self._prepare_g2_context(self._round_state)

    def _evaluate(self, actions: List[str]):
        reward, states, footprints, igs = self._simulate_trajectory(
            self._round_state, actions, self._coverage_discount
        )
        return reward, list(actions), states, footprints, igs

    def _random_actions(self) -> List[str]:
        pos = (self._round_state[0], self._round_state[1])
        alt = self._round_state[2]
        actions = []
        for _ in range(self.horizon):
            a = self._rng.choice(self._admissible_actions(pos, alt))
            actions.append(a)
            nxt = self._apply_action(pos, alt, a)
            if nxt is not None:
                pos, alt = nxt
        return actions

    @staticmethod
    def _state_key(pos, alt) -> Tuple[float, float, float]:
        return (round(pos[0], 2), round(pos[1], 2), round(alt, 2))

    def _mcts_select(self) -> Tuple[List[str], List[Tuple[Tuple, str]]]:
        """
        One D-UCT descent: tree policy until an untried action is expanded,
        then a random rollout to the horizon.

        Returns:
            (actions, tree edges [(state_key, action), ...] to backpropagate)
        """
        self._tree_t += 1
        t, gamma = self._tree_t, self.d_uct_gamma
        pos = (self._round_state[0], self._round_state[1])
        alt = self._round_state[2]
        actions: List[str] = []
        edges: List[Tuple[Tuple, str]] = []
        in_tree = True
        for _ in range(self.horizon):
            admissible = self._admissible_actions(pos, alt)
            if in_tree:
                key = self._state_key(pos, alt)
                node = self._tree.setdefault(key, {})
                untried = [a for a in admissible if a not in node]
                if untried:
                    a = untried[int(self._rng.integers(len(untried)))]
                    node[a] = DUCTStats()
                    in_tree = False  # expanded: rest is rollout
                else:
                    parent = sum(st.count_at(t, gamma) for st in node.values())
                    a = max(
                        admissible,
                        key=lambda act: d_uct_score(node[act], parent, t, gamma, self.ucb_c),
                    )
                edges.append((key, a))
            else:
                a = admissible[int(self._rng.integers(len(admissible)))]
            actions.append(a)
            nxt = self._apply_action(pos, alt, a)
            if nxt is not None:
                pos, alt = nxt
        return actions, edges

    def run_iterations(self, n: int) -> LLIntent:
        """
        Spend n iterations on the current round and return the current best
        plan (anytime). The incumbent plan is re-scored first, because teammate
        intents / HL guidance may have changed since it was found.
        """
        if self.belief is None or self._round_state is None:
            self.current_intent = LLIntent(agent_id=self.agent_id, step=self.current_step)
            return self.current_intent

        self._refresh_context()
        best = self._evaluate(self._best_actions) if self._best_actions else None

        for _ in range(int(n)):
            if self.use_mcts_llp:
                actions, edges = self._mcts_select()
                cand = self._evaluate(actions)
                for key, a in edges:
                    self._tree[key][a].add(self._tree_t, self.d_uct_gamma, cand[0])
            else:
                cand = self._evaluate(self._random_actions())
            first = cand[1][0]
            if cand[0] > self._mcts_action_values.get(first, float("-inf")):
                self._mcts_action_values[first] = cand[0]
            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:  # n == 0 and no incumbent yet
            best = self._evaluate(self._random_actions())

        best_reward, best_actions, best_states, best_footprints, best_igs = best
        self._best_actions = best_actions
        self._stats["total_iterations"] += int(n)

        intent = LLIntent(
            agent_id=self.agent_id,
            action_sequence=best_actions,
            state_sequence=best_states,
            footprint_sequence=best_footprints,
            ig_sequence=best_igs,
            total_expected_ig=sum(best_igs) if best_igs else 0.0,
            step=self.current_step,
            horizon=self.horizon,
            value=best_reward,
            iterations=self._tree_t if self.use_mcts_llp else int(n),
        )
        self.last_g2_steps = None
        if self._g2_ctx is not None:
            self.last_g2_steps = (
                self._g2_steps(best_states, best_footprints)
                if best_states
                else self._g2_ctx["baseline"]
            )
        self.current_intent = intent
        return intent

    def plan(self, current_state: Tuple[float, float, float]) -> LLIntent:
        """Synchronous planning: one round with the full iteration budget."""
        if self.belief is None:
            return LLIntent(agent_id=self.agent_id, step=self.current_step)
        self.begin_round(current_state)
        intent = self.run_iterations(self.num_iterations)
        self._stats["plans_generated"] += 1
        return intent

    def get_best_action(self) -> str:
        """Get the first action from current intent."""
        if self.current_intent and self.current_intent.action_sequence:
            return self.current_intent.action_sequence[0]
        return "hover"

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return dict(self._stats)


# =============================================================================
# High-Level Planner (HLP) with Dec-MCTS
# =============================================================================


class HighLevelPlanner:
    """
    High-Level Planner: which regions this agent covers next, and in what order.

    Goal: the whole field observed at least once, without leaving small
    uncovered spots behind that someone has to come back for.

    Objective (g2 from g2_evaluator, in agent-steps, lower = better):

        value(seq) = g2(my HL sequence = [])  -  g2(my HL sequence = seq)

    i.e. the agent-steps the team saves when this agent commits to `seq`.
    Uncovered regions that nobody claims are still paid for (sweep effort +
    spanning-tree travel from the agents), so skipping a nearby leftover spot
    costs the detour needed to come back for it later.

    Search: MCTS over ORDERED region sequences of length `horizon`
      - actions at a node: the `candidate_regions` uncovered regions with the
        fewest reach steps from the agent's position after the sequence so far
      - rollout: complete the sequence greedily with the nearest uncovered region
      - reward: value(full sequence) as above
    Replanning: every `replan_interval` steps, or immediately when the current
    target region has been fully observed (or there is no plan yet).
    """

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        grid_shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (100, 100),
        horizon: int = 3,
        num_iterations: int = 50,
        ucb_c: float = 1.0,
        replan_interval: int = 1,
        candidate_regions: int = 6,
        step_cells: float = 25.0,
        fp_half: float = 25.0,
        d_uct_threshold_steps: float = 2.0,
        d_uct_gamma: float = 0.99,
        world_to_grid=None,
        seed: Optional[int] = None,
        grid_info=None,
    ):
        """
        Initialize HLP.

        Args:
            agent_id: This agent's ID
            num_agents: Total number of agents
            grid_shape: (H, W) shape of the grid
            tile_size: Size of region tiles (cells)
            horizon: Number of regions to plan ahead
            num_iterations: MCTS iterations per replan
            ucb_c: UCB exploration constant (values are in steps)
            replan_interval: Replan at least every this many simulation steps
            candidate_regions: Branching factor: nearest uncovered regions considered per node
            step_cells: Lateral move length in cells
            fp_half: Half-width (cells) of the footprint at minimum altitude
            d_uct_threshold_steps: teammate intents older than 2x this are ignored
            world_to_grid: callable(x, y) -> (i, j) for teammates' LL end states
            grid_info: Grid metadata (kept for visualisation helpers)
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.grid_shape = grid_shape
        self.grid_info = grid_info
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        self.horizon = int(horizon)
        self.num_iterations = int(num_iterations)
        self.ucb_c = ucb_c
        self.replan_interval = max(1, int(round(replan_interval)))
        self.candidate_regions = max(1, int(candidate_regions))
        self.step_cells = float(step_cells)
        self.fp_half = float(fp_half)
        self.max_intent_age_steps = 2.0 * d_uct_threshold_steps
        self.d_uct_gamma = d_uct_gamma  # tree discount (paper Eq. 4-5)
        self._world_to_grid = world_to_grid

        # Anytime search state of the current HLP cycle (persists across steps
        # until the next replan; rewards drift, hence D-UCT)
        self._tree: Optional[Dict[Tuple[int, ...], Dict[str, Any]]] = None
        self._tree_t = 0
        self._cycle_position: Optional[Tuple[float, float]] = None
        self._best_sequence: Optional[List[int]] = None

        # RNG
        self._rng = np.random.default_rng(seed)

        # Partition grid into regions (ids match g2_evaluator's tile layout)
        self.regions = self._partition_grid()
        self.num_regions = len(self.regions)

        # Current belief / coverage knowledge
        self.belief: Optional[np.ndarray] = None
        self._uncovered: Optional[np.ndarray] = None
        self._base_model: Optional[CoverageStepModel] = None
        self.current_step: int = 0

        # Teammate intents
        self._teammate_ll_intents: Dict[int, LLIntent] = {}
        self._teammate_hl_intents: Dict[int, HLIntent] = {}

        # This agent's own latest LL intent (bottom-up information, as in the
        # paper's Alg. 2: the HL plan starts where the LL plan ends)
        self._own_ll_intent: Optional[LLIntent] = None
        self.last_plan_start: Optional[Tuple[float, float]] = None

        # Current intent
        self.current_intent: Optional[HLIntent] = None
        self._last_replan_step: Optional[int] = None
        self.last_plan_replanned: bool = False

        # Fraction of each region observed at least once (for logging / viewer)
        self._region_coverage: Dict[int, float] = {
            i: 0.0 for i in range(self.num_regions)
        }

        # Statistics
        self._stats = {
            "plans_generated": 0,
            "total_iterations": 0,
            "intent_updates_received": 0,
        }

    def _partition_grid(self) -> Dict[int, Dict[str, Any]]:
        """Partition grid into rectangular regions (row-major tile order)."""
        H, W = self.grid_shape
        tile_h, tile_w = self.tile_size

        regions = {}
        region_id = 0

        for i in range(0, H, tile_h):
            for j in range(0, W, tile_w):
                i_end = min(i + tile_h, H)
                j_end = min(j + tile_w, W)

                regions[region_id] = {
                    # Format: ((row_min, row_max), (col_min, col_max)) for viewer compatibility
                    "bounds": ((i, i_end), (j, j_end)),
                    "center": ((i + i_end) / 2, (j + j_end) / 2),
                    "area": (i_end - i) * (j_end - j),
                }
                region_id += 1

        return regions

    # ------------------------------------------------------------- inputs
    def update_belief(self, belief: np.ndarray) -> None:
        """Store the belief (used only if no explicit coverage mask is given)."""
        self.belief = np.asarray(belief)
        if self._uncovered is None:
            self._set_uncovered(uncovered_from_belief(self.belief))

    def update_coverage(self, uncovered: np.ndarray) -> None:
        """Cells (H, W bool) not yet known to be observed at least once."""
        self._set_uncovered(np.asarray(uncovered, dtype=bool))

    def _set_uncovered(self, uncovered: np.ndarray) -> None:
        self._uncovered = uncovered
        self._base_model = CoverageStepModel(
            uncovered, self.tile_size, self.step_cells, self.fp_half
        )
        for rid, region in self.regions.items():
            self._region_coverage[rid] = 1.0 - (
                self._base_model.counts[rid] / max(1, region["area"])
            )

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def update_teammate_intents(
        self,
        ll_intents: Dict[int, LLIntent],
        hl_intents: Dict[int, HLIntent],
    ) -> None:
        """Update stored teammate intents."""
        self._teammate_ll_intents = ll_intents
        self._teammate_hl_intents = hl_intents
        self._stats["intent_updates_received"] += 1

    def update_own_ll_intent(self, intent: Optional[LLIntent]) -> None:
        """Latest LL intent of this agent's own LLP (bottom-up information)."""
        self._own_ll_intent = intent

    # --------------------------------------------------------- team context
    def _team_context(self):
        """
        Coverage model after all fresh LL plans (teammates' AND this agent's
        own), plus the teammates' (start, HL sequence) pairs and this agent's
        HL start position (end of its own LL plan, if any). Intents older than
        max_intent_age_steps are ignored.
        """
        fresh_ll = {
            tid: it
            for tid, it in self._teammate_ll_intents.items()
            if not it.is_stale(self.current_step, self.max_intent_age_steps)
        }
        fresh_hl = {
            tid: it
            for tid, it in self._teammate_hl_intents.items()
            if not it.is_stale(self.current_step, self.max_intent_age_steps)
        }
        own = self._own_ll_intent
        if own is not None and own.is_stale(self.current_step, self.max_intent_age_steps):
            own = None
        ll_plans = list(fresh_ll.values()) + ([own] if own is not None else [])
        model = self._base_model.with_footprints(
            fp for it in ll_plans for fp in it.footprint_sequence
        )
        own_start = None
        if own is not None and own.state_sequence and self._world_to_grid:
            x, y, _ = own.state_sequence[-1]
            own_start = self._world_to_grid(x, y)
        teammates = {}
        for tid in set(fresh_ll) | set(fresh_hl):
            start = None
            ll = fresh_ll.get(tid)
            if ll is not None and ll.state_sequence and self._world_to_grid:
                x, y, _ = ll.state_sequence[-1]
                start = self._world_to_grid(x, y)
            hl = fresh_hl.get(tid)
            teammates[tid] = (start, list(hl.region_sequence) if hl else [])
        return model, teammates, own_start

    def _should_replan(self, model: CoverageStepModel) -> bool:
        if self.current_intent is None or not self.current_intent.region_sequence:
            return True
        target = self.current_intent.current_target_region
        if target is None or model.counts[target] == 0:
            return True  # target finished (or covered by a teammate's plan)
        if self._last_replan_step is None:
            return True
        return self.current_step - self._last_replan_step >= self.replan_interval

    # ---------------------------------------------------------------- plan
    def _begin_cycle(self, current_position: Tuple[float, float]) -> None:
        """Start a new search cycle (fresh tree) from the agent's position (grid)."""
        self._last_replan_step = self.current_step
        self.last_plan_replanned = True
        self._cycle_position = (float(current_position[0]), float(current_position[1]))
        self._tree = {}
        self._tree_t = 0
        self._best_sequence = None
        self._stats["plans_generated"] += 1

    def begin_step(self, current_position: Tuple[float, float]) -> bool:
        """
        Asynchronous mode, once per simulation step: start a new cycle if a
        replan is due (interval elapsed, target finished, or no plan yet);
        otherwise the current tree keeps growing. Returns True on a new cycle.
        """
        self.last_plan_replanned = False
        if self._base_model is None:
            return False
        model, _, _ = self._team_context()
        if self._tree is not None and not self._should_replan(model):
            self._cycle_position = (float(current_position[0]), float(current_position[1]))
            return False
        self._begin_cycle(current_position)
        return True

    def run_iterations(self, n: int) -> HLIntent:
        """
        Spend n MCTS iterations on the current cycle with the CURRENT context
        (coverage, own LL plan, teammate intents) and return the best region
        sequence found so far (anytime). The incumbent and the greedy sequence
        are re-scored first because the context may have changed.
        """
        if self._base_model is None or self._tree is None:
            self.current_intent = HLIntent(agent_id=self.agent_id, step=self.current_step)
            return self.current_intent

        model, teammates, own_start = self._team_context()
        # Paper (Alg. 2): the HL plan continues from the END of this agent's own
        # LL plan, whose cells count as covered; without one, from the agent.
        start = own_start if own_start is not None else self._cycle_position
        start = (float(start[0]), float(start[1]))
        self.last_plan_start = start
        evaluator = _SequenceEvaluator(model, teammates, start)

        best_sequence, best_value = [], float("-inf")
        if model.open_ids.size > 0:
            for seq in (self._best_sequence, self._greedy_complete(model, [], start)):
                if seq:
                    value = evaluator.value(seq)
                    if value > best_value:
                        best_sequence, best_value = list(seq), value
            for _ in range(int(n)):
                seq, value = self._mcts_iteration(model, start, evaluator)
                if value > best_value:
                    best_sequence, best_value = seq, value
        self._best_sequence = best_sequence or None
        self._stats["total_iterations"] += int(n)
        self.current_intent = self._make_intent(model, start, best_sequence, best_value)
        return self.current_intent

    def plan(self, current_position: Tuple[float, float]) -> HLIntent:
        """
        Synchronous API: keep the current plan, or (when a replan is due) run a
        fresh cycle with the full iteration budget.

        Args:
            current_position: agent position in grid coordinates (row, col)
        """
        self.last_plan_replanned = False
        if self._base_model is None:
            return HLIntent(agent_id=self.agent_id, step=self.current_step)
        model, _, _ = self._team_context()
        if not self._should_replan(model) and self.current_intent is not None:
            return self.current_intent
        self._begin_cycle(current_position)
        return self.run_iterations(self.num_iterations)

    def _make_intent(
        self,
        model: CoverageStepModel,
        start: Tuple[float, float],
        sequence: List[int],
        value: float,
    ) -> HLIntent:
        if not sequence:
            return HLIntent(
                agent_id=self.agent_id,
                step=self.current_step,
                horizon=self.horizon,
                iterations=self._tree_t,
            )
        _, etas, finished, _ = model.sequence_cost(start, sequence)
        per_region = [
            f - (finished[k - 1] if k > 0 else 0.0) for k, f in enumerate(finished)
        ]
        target = sequence[0]
        return HLIntent(
            agent_id=self.agent_id,
            region_sequence=list(sequence),
            eta_sequence=etas,
            completion_sequence=finished,
            score_sequence=per_region,
            current_target_region=target,
            target_center=self.regions[target]["center"],
            step=self.current_step,
            horizon=self.horizon,
            value=value,
            iterations=self._tree_t,
        )

    def _candidates(
        self,
        model: CoverageStepModel,
        pos: Tuple[float, float],
        used: Tuple[int, ...],
    ) -> List[int]:
        """Nearest uncovered regions (by reach steps, then most uncovered cells)."""
        open_ids = model.open_ids
        if used:
            open_ids = open_ids[~np.isin(open_ids, used)]
        if open_ids.size == 0:
            return []
        reach = model.reach_steps_all(pos)[open_ids]
        order = np.lexsort((-model.counts[open_ids], reach))
        return [int(r) for r in open_ids[order[: self.candidate_regions]]]

    def _greedy_complete(
        self, model: CoverageStepModel, sequence: List[int], start: Tuple[float, float]
    ) -> List[int]:
        """Rollout policy: append the nearest uncovered region until horizon."""
        seq = list(sequence)
        pos = model.exit_position(seq[-1]) if seq else start
        while len(seq) < self.horizon:
            cands = self._candidates(model, pos, tuple(seq))
            if not cands:
                break
            seq.append(cands[0])
            pos = model.exit_position(cands[0])
        return seq

    def _mcts_iteration(
        self,
        model: CoverageStepModel,
        start: Tuple[float, float],
        evaluator: "_SequenceEvaluator",
    ) -> Tuple[List[int], float]:
        """
        One MCTS iteration over ORDERED region sequences (tree key = tuple of
        regions so far): D-UCT selection, expansion of one untried candidate,
        greedy nearest-region rollout to the horizon, D-UCT backpropagation.
        Node candidate lists are fixed when a node is created.
        """
        tree = self._tree
        self._tree_t += 1
        t, gamma = self._tree_t, self.d_uct_gamma

        def node_for(key: Tuple[int, ...]) -> Dict[str, Any]:
            node = tree.get(key)
            if node is None:
                pos = model.exit_position(key[-1]) if key else start
                node = {
                    "stats": DUCTStats(),
                    "candidates": self._candidates(model, pos, key),
                    "children": set(),
                }
                tree[key] = node
            return node

        key: Tuple[int, ...] = ()
        path = [key]
        node = node_for(key)
        while len(key) < self.horizon and node["candidates"]:
            unexplored = [c for c in node["candidates"] if c not in node["children"]]
            if unexplored:
                action = unexplored[int(self._rng.integers(len(unexplored)))]
                node["children"].add(action)
                key = key + (action,)
                path.append(key)
                node_for(key)
                break
            parent_count = node["stats"].count_at(t, gamma)
            action = max(
                node["candidates"],
                key=lambda c: d_uct_score(
                    tree[key + (c,)]["stats"], parent_count, t, gamma, self.ucb_c
                ),
            )
            key = key + (action,)
            path.append(key)
            node = tree[key]

        sequence = self._greedy_complete(model, list(key), start)
        value = evaluator.value(sequence)
        for k in path:
            tree[k]["stats"].add(t, gamma, value)
        return sequence, value

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return dict(self._stats)


class _SequenceEvaluator:
    """
    value(seq) = g2(own sequence empty) - g2(own sequence = seq), in agent-steps.

    Teammates' claimed work is fixed during one HLP replan, so it is computed
    once here.
    """

    def __init__(self, model: CoverageStepModel, teammates, start):
        self.model = model
        self.start = start
        self.teammate_cost = 0.0
        self.teammate_claimed = np.zeros(model.num_regions, dtype=bool)
        self.teammate_ends: List[Tuple[float, float]] = []
        for t_start, t_seq in teammates.values():
            cost, _, _, end = model.sequence_cost(t_start, t_seq)
            self.teammate_cost += cost
            for r in t_seq:
                if 0 <= r < model.num_regions:
                    self.teammate_claimed[r] = True
            if end is not None:
                self.teammate_ends.append(end)
        self.null_steps = self.steps([])

    def steps(self, sequence: List[int]) -> float:
        own_cost, _, _, end = self.model.sequence_cost(self.start, sequence)
        claimed = self.teammate_claimed.copy()
        if sequence:
            claimed[np.asarray(sequence, dtype=np.int64)] = True
        ends = self.teammate_ends + ([end] if end is not None else [])
        return self.teammate_cost + own_cost + self.model.leftover_steps(claimed, ends)

    def value(self, sequence: List[int]) -> float:
        return self.null_steps - self.steps(sequence)


# =============================================================================
# Hierarchical Dec-MCTS Planner (Combined)
# =============================================================================


class HierarchicalDecMCTSPlanner:
    """
    Per-agent MH-Dec-MCTS planner combining an HLP and an LLP.

    Each simulation step (plan()):
      1. read teammate intents this agent has received (range-limited IntentBus)
      2. HLP: keep or replan the region sequence (every hlp_replan_interval
         steps, or when the current target region is fully observed)
      3. LLP: choose an action sequence maximising g1 + g2_weight * steps saved
      4. broadcast own LL and HL intents (stamped with the current step)
    """

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        camera,
        grid_info,
        intent_bus: IntentBus,
        llp_horizon: int = 5,
        llp_iterations: int = 100,
        llp_ucb_c: float = 1.41,
        llp_discount_factor: float = 0.95,
        hlp_horizon: int = 3,
        hlp_iterations: int = 50,
        hlp_ucb_c: float = 1.0,
        tile_size: Tuple[int, int] = (100, 100),
        hlp_replan_interval: int = 1,
        hlp_candidate_regions: int = 6,
        use_mcts_llp: bool = False,
        use_g2: bool = False,
        g2_weight: Optional[float] = None,
        d_uct_decay: float = 0.9,
        d_uct_threshold_steps: float = 2.0,
        d_uct_gamma: float = 0.99,
        conf_dict: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize hierarchical planner.

        Args:
            agent_id: This agent's ID
            num_agents: Total number of agents
            camera: UAV camera model
            grid_info: Grid information
            intent_bus: Shared intent bus for communication
            llp_horizon: LLP planning horizon (steps)
            llp_iterations: LLP rollouts / MCTS iterations
            llp_ucb_c: LLP UCB exploration constant (MCTS LLP only)
            llp_discount_factor: LLP reward discount factor
            hlp_horizon: HLP planning horizon (regions)
            hlp_iterations: HLP MCTS iterations
            hlp_ucb_c: HLP UCB exploration constant
            tile_size: Region tile size for HLP (cells)
            hlp_replan_interval: Replan the HLP at least every N simulation steps
            hlp_candidate_regions: HLP branching factor (nearest uncovered regions)
            use_mcts_llp: If True, use MCTS tree search for LLP; if False, use random rollouts
            use_g2: If True, the LLP reward includes g2 (steps saved). Without it the
                HLP still plans and broadcasts, but does not influence this agent's actions.
            g2_weight: bits per saved step in the LLP reward (None = auto)
            d_uct_decay, d_uct_threshold_steps: step-based staleness of teammate intents
            d_uct_gamma: D-UCT tree discount per iteration (paper Eq. 4-5)
            conf_dict: Sensor model dict {altitude: (s0, s1)} for accurate IG computation
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.camera = camera
        self.grid_info = grid_info
        self.intent_bus = intent_bus

        step_cells, fp_half = camera_step_geometry(camera)

        # Create LLP
        self.llp = LowLevelPlanner(
            agent_id=agent_id,
            camera=camera,
            grid_info=grid_info,
            conf_dict=conf_dict,
            horizon=llp_horizon,
            num_iterations=llp_iterations,
            ucb_c=llp_ucb_c,
            discount=llp_discount_factor,
            use_mcts_llp=use_mcts_llp,
            use_g2=use_g2,
            g2_weight=g2_weight,
            d_uct_decay=d_uct_decay,
            d_uct_threshold_steps=d_uct_threshold_steps,
            d_uct_gamma=d_uct_gamma,
            region_tile_size=tile_size,
            step_cells=step_cells,
            fp_half=fp_half,
            seed=seed,
        )

        # Create HLP
        self.hlp = HighLevelPlanner(
            agent_id=agent_id,
            num_agents=num_agents,
            grid_shape=grid_info.shape,
            tile_size=tile_size,
            horizon=hlp_horizon,
            num_iterations=hlp_iterations,
            ucb_c=hlp_ucb_c,
            replan_interval=hlp_replan_interval,
            candidate_regions=hlp_candidate_regions,
            step_cells=step_cells,
            fp_half=fp_half,
            d_uct_threshold_steps=d_uct_threshold_steps,
            d_uct_gamma=d_uct_gamma,
            world_to_grid=self._world_to_grid,
            seed=seed if seed is None else seed + 1000,  # Offset for HLP
            grid_info=grid_info,
        )

        # Current state
        self._current_position: Optional[Tuple[float, float]] = None
        self._current_altitude: float = 0.0
        self._current_step: int = 0
        self._step_given: bool = False

        # Statistics
        self._stats = {
            "planning_cycles": 0,
            "ll_intents_broadcast": 0,
            "hl_intents_broadcast": 0,
        }

    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self.camera.convert_xy_ij(x, y, self.camera.grid.center)
        return float(i), float(j)

    @property
    def current_region_metadata(self) -> Optional[Dict]:
        """Get current HLP region metadata for visualization."""
        return self.hlp.regions

    @property
    def current_selected_region(self) -> Optional[int]:
        """Get currently selected target region."""
        if self.hlp.current_intent is not None:
            return self.hlp.current_intent.current_target_region
        return None

    @property
    def current_region_scores(self) -> Optional[Dict[int, float]]:
        """Get current region scores."""
        return dict(self.hlp._region_coverage)

    def update_state(
        self,
        position: Tuple[float, float],
        altitude: float,
        belief: np.ndarray,
        coverage_mask: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Update planner state.

        Args:
            position: Current (x, y) position (world metres)
            altitude: Current altitude
            belief: Current belief map
            coverage_mask: (H, W) bool, True for cells this agent knows were
                observed at least once (own observations + teammates' news).
                If None, cells whose belief left 0.5 are treated as observed.
            step: Current simulation step (used for intent ages and HLP
                replanning). If None, an internal counter of plan() calls is used.
        """
        self._current_position = position
        self._current_altitude = altitude
        if step is not None:
            self._current_step = int(step)
            self._step_given = True

        self.llp.update_belief(belief)
        self.hlp.update_belief(belief)
        if coverage_mask is not None:
            uncovered = ~np.asarray(coverage_mask, dtype=bool)
            self.llp.update_coverage(uncovered)
            self.hlp.update_coverage(uncovered)
        else:
            uncovered = uncovered_from_belief(np.asarray(belief))
            self.llp.update_coverage(uncovered)
            self.hlp.update_coverage(uncovered)

    def receive_intents(self) -> None:
        """
        Receive and process intents from teammates.

        This updates both LLP and HLP with teammate intent information.
        """
        # Get teammate intents
        ll_intents = self.intent_bus.get_teammate_ll_intents(self.agent_id)
        hl_intents = self.intent_bus.get_teammate_hl_intents(self.agent_id)

        # Update planners
        self.llp.update_teammate_intents(ll_intents, hl_intents)
        self.hlp.update_teammate_intents(ll_intents, hl_intents)

    # ------------------------------------------------------------ step API
    def _sync_step_counters(self) -> int:
        if not self._step_given:
            # Standalone use without a simulator step: count planning cycles.
            self._current_step = self._stats["planning_cycles"]
        step = self._current_step
        self.llp.set_step(step)
        self.hlp.set_step(step)
        return step

    def _guidance_for_llp(self, hl_intent: HLIntent) -> HLIntent:
        """HL intent for the LLP (region ids drive g2; centre in world coords for logs)."""
        hl_for_llp = copy.copy(hl_intent)
        if hl_intent.target_center is not None:
            gi, gj = hl_intent.target_center
            hl_for_llp.target_center = self.camera.ij_to_xy(gi, gj)
        return hl_for_llp

    def _publish(self, ll_intent: LLIntent, hl_intent: HLIntent, step: int) -> None:
        # The HL plan is re-asserted each time, so a copy stamped with the
        # current step is sent; stored objects are never mutated because
        # receivers keep references to them.
        self.intent_bus.broadcast_ll_intent(ll_intent)
        self.intent_bus.broadcast_hl_intent(dataclasses.replace(hl_intent, step=step))
        self._stats["ll_intents_broadcast"] += 1
        self._stats["hl_intents_broadcast"] += 1

    def begin_step(self) -> None:
        """
        Asynchronous mode: start this simulation step (call after update_state).

        The HLP starts a new cycle if a replan is due, otherwise its tree keeps
        growing; the LLP starts a fresh round from the current state.
        """
        step = self._sync_step_counters()
        self.receive_intents()
        self.hlp.update_own_ll_intent(self.llp.current_intent)
        grid_pos = self._world_to_grid(self._current_position[0], self._current_position[1])
        self.hlp.begin_step(grid_pos)
        self._step_replanned = self.hlp.last_plan_replanned
        self.llp.begin_round(
            (self._current_position[0], self._current_position[1], self._current_altitude)
        )
        self._slice_timing = {"hlp_ms": 0.0, "llp_ms": 0.0, "start": None, "end": None}
        self._step_index = step

    def run_slice(self, llp_iterations: int, hlp_iterations: int) -> None:
        """
        Asynchronous mode: one slice of concurrent planning.

        Both levels read the intents delivered so far (teammates' and each
        other's), search for their share of this step's budget, and publish
        their current best plans. With a deferred IntentBus, teammates receive
        them only at the slice boundary.
        """
        import time

        t0 = time.perf_counter()
        if self._slice_timing["start"] is None:
            self._slice_timing["start"] = t0
        self.receive_intents()
        # vertical exchange inside the robot: HLP <- own LL plan, LLP <- own HL plan
        self.hlp.update_own_ll_intent(self.llp.current_intent)
        hl_intent = self.hlp.run_iterations(hlp_iterations)
        t1 = time.perf_counter()
        self.llp.update_hl_guidance(self._guidance_for_llp(hl_intent))
        ll_intent = self.llp.run_iterations(llp_iterations)
        t2 = time.perf_counter()
        self._publish(ll_intent, hl_intent, self._step_index)
        self._slice_timing["hlp_ms"] += (t1 - t0) * 1000.0
        self._slice_timing["llp_ms"] += (t2 - t1) * 1000.0
        self._slice_timing["end"] = t2

    def finish_step(self) -> Tuple[str, Dict[str, Any]]:
        """Asynchronous mode: the action to execute and the step's metrics."""
        timing = self._slice_timing
        start = timing["start"] or 0.0
        hlp_ms, llp_ms = timing["hlp_ms"], timing["llp_ms"]
        metrics = self._compile_metrics(
            self.llp.current_intent or LLIntent(agent_id=self.agent_id),
            self.hlp.current_intent or HLIntent(agent_id=self.agent_id),
            hlp_ms,
            llp_ms,
            start * 1000.0,
            self._step_replanned,
        )
        self.llp._stats["plans_generated"] += 1
        self._stats["planning_cycles"] += 1
        return self.llp.get_best_action(), metrics

    def plan(self) -> Tuple[str, Dict[str, Any]]:
        """
        Synchronous planning (one call per step; used by tests and when
        decentralized.async.enabled is false):

        1. Receive teammate intents (only those delivered to this agent)
        2. HLP: keep the plan or replan (it starts from the end of the agent's
           own latest LL plan)
        3. Hand the HL sequence to the LLP (used by its g2 term)
        4. LLP: plan the next actions with the full budget
        5. Broadcast LL and HL intents stamped with the current step

        Returns:
            (best_action, metrics_dict)
        """
        import time

        if self._current_position is None:
            return "hover", {}

        step = self._sync_step_counters()
        self.receive_intents()

        grid_pos = self._world_to_grid(
            self._current_position[0], self._current_position[1]
        )
        hlp_start_time = time.perf_counter()
        # Bottom-up: the HLP sees this agent's latest LL plan (made last step;
        # its first action has already been executed).
        self.hlp.update_own_ll_intent(self.llp.current_intent)
        hl_intent = self.hlp.plan(grid_pos)
        hlp_end_time = time.perf_counter()

        self.llp.update_hl_guidance(self._guidance_for_llp(hl_intent))
        current_state = (
            self._current_position[0],
            self._current_position[1],
            self._current_altitude,
        )
        llp_start_time = time.perf_counter()
        ll_intent = self.llp.plan(current_state)
        llp_end_time = time.perf_counter()

        self._publish(ll_intent, hl_intent, step)
        metrics = self._compile_metrics(
            ll_intent,
            hl_intent,
            (hlp_end_time - hlp_start_time) * 1000.0,
            (llp_end_time - llp_start_time) * 1000.0,
            hlp_start_time * 1000.0,
            self.hlp.last_plan_replanned,
        )
        self._stats["planning_cycles"] += 1
        return self.llp.get_best_action(), metrics

    def _compile_metrics(
        self,
        ll_intent: LLIntent,
        hl_intent: HLIntent,
        hlp_time_ms: float,
        llp_time_ms: float,
        start_ms: float,
        hlp_replanned: bool,
    ) -> Dict[str, Any]:
        best_action = self.llp.get_best_action()
        ll_intents = self.intent_bus.get_teammate_ll_intents(self.agent_id)
        hl_intents = self.intent_bus.get_teammate_hl_intents(self.agent_id)
        intents_received = {
            "ll_intents_from": list(ll_intents.keys()),
            "hl_intents_from": list(hl_intents.keys()),
        }
        for tid, hl_int in hl_intents.items():
            intents_received[f"agent_{tid}_target"] = hl_int.current_target_region
        hlp_end_ms = start_ms + hlp_time_ms
        return {
            "ll_intent": ll_intent,
            "hl_intent": hl_intent,
            "action": best_action,
            "ll_value": ll_intent.value,
            "hl_value": hl_intent.value,
            "target_region": hl_intent.current_target_region,
            "expected_ig": ll_intent.total_expected_ig,
            # Timing breakdown (compute time; in async mode HLP/LLP slices are
            # interleaved, so start/end are start + accumulated compute time)
            "hlp_time_ms": hlp_time_ms,
            "llp_time_ms": llp_time_ms,
            "hlp_start_ms": start_ms,
            "hlp_end_ms": hlp_end_ms,
            "llp_start_ms": hlp_end_ms,
            "llp_end_ms": hlp_end_ms + llp_time_ms,
            "hlp_replanned": hlp_replanned,
            "g2_steps": self.llp.last_g2_steps,
            "mcts_action_values": getattr(self.llp, "_mcts_action_values", {}),
            "hlp_region_scores": dict(self.hlp._region_coverage),
            "intents_received": intents_received,
        }

    def select_action(
        self,
        belief: np.ndarray,
        visited_positions: List[Any],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Interface compatible with existing planner API.

        Args:
            belief: Current belief map
            visited_positions: List of visited positions

        Returns:
            (action, {action: ig_score for all actions})
        """
        # Update belief
        if len(visited_positions) > 0:
            last_pos = visited_positions[-1]
            self._current_position = last_pos.position
            self._current_altitude = last_pos.altitude

        self.update_state(
            self._current_position or (0, 0),
            self._current_altitude,
            belief,
        )

        # Run planning
        action, metrics = self.plan()

        # Build action scores (for compatibility)
        action_scores = {}
        ll_intent = metrics.get("ll_intent")
        if ll_intent:
            for i, act in enumerate(ll_intent.action_sequence):
                if i < len(ll_intent.ig_sequence):
                    action_scores[act] = ll_intent.ig_sequence[i]

        # Ensure all actions have a score
        for act in self.llp.actions:
            if act not in action_scores:
                action_scores[act] = 0.0

        # Add timing breakdown as special keys (prefixed with _timing_)
        action_scores["_timing_hlp_ms"] = metrics.get("hlp_time_ms", 0.0)
        action_scores["_timing_llp_ms"] = metrics.get("llp_time_ms", 0.0)
        action_scores["_timing_hlp_start_ms"] = metrics.get("hlp_start_ms")
        action_scores["_timing_hlp_end_ms"] = metrics.get("hlp_end_ms")
        action_scores["_timing_llp_start_ms"] = metrics.get("llp_start_ms")
        action_scores["_timing_llp_end_ms"] = metrics.get("llp_end_ms")
        action_scores["_timing_hlp_replanned"] = (
            1.0 if metrics.get("hlp_replanned", False) else 0.0
        )

        return action, action_scores

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "hierarchical": dict(self._stats),
            "llp": self.llp.get_statistics(),
            "hlp": self.hlp.get_statistics(),
            "intent_bus": self.intent_bus.get_statistics(),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_hierarchical_planner(
    agent_id: int,
    num_agents: int,
    camera,
    grid_info,
    intent_bus: Optional[IntentBus] = None,
    config: Optional[Dict[str, Any]] = None,
    conf_dict: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> HierarchicalDecMCTSPlanner:
    """
    Factory function to create a hierarchical Dec-MCTS planner.

    Args:
        agent_id: Agent ID
        num_agents: Total number of agents
        camera: UAV camera
        grid_info: Grid information
        intent_bus: Shared intent bus (created if None)
        config: Configuration dict

    Returns:
        HierarchicalDecMCTSPlanner instance
    """
    config = config or {}

    # Create intent bus if not provided
    if intent_bus is None:
        intent_bus = IntentBus(num_agents=num_agents)

    # Extract config
    if "g2_mode" in config:
        logger.warning(
            "hierarchical_dec_mcts.g2_mode is no longer used: g2 is the step-based "
            "coverage estimate in g2_evaluator.py"
        )

    return HierarchicalDecMCTSPlanner(
        agent_id=agent_id,
        num_agents=num_agents,
        camera=camera,
        grid_info=grid_info,
        intent_bus=intent_bus,
        llp_horizon=config.get("llp_horizon", 5),
        llp_iterations=config.get("llp_iterations", 100),
        llp_ucb_c=config.get("llp_ucb_c", 1.41),
        llp_discount_factor=config.get("llp_discount_factor", 0.95),
        hlp_horizon=config.get("hlp_horizon", 3),
        hlp_iterations=config.get("hlp_iterations", 50),
        hlp_ucb_c=config.get("hlp_ucb_c", 1.0),
        tile_size=tuple(config.get("tile_size", [100, 100])),
        hlp_replan_interval=config.get("hlp_replan_interval", 1),
        hlp_candidate_regions=config.get("hlp_candidate_regions", 6),
        use_mcts_llp=config.get("use_mcts_llp", False),
        use_g2=config.get("use_g2", False),
        g2_weight=config.get("g2_weight"),
        d_uct_decay=config.get("d_uct_decay", 0.9),
        d_uct_threshold_steps=config.get("d_uct_threshold_steps", 2.0),
        d_uct_gamma=config.get("d_uct_gamma", 0.99),
        conf_dict=conf_dict,
        seed=seed,
    )

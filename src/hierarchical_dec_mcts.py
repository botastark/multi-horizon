"""
Hierarchical Decentralized MCTS Planner with Shared Beliefs and Intents

Implements the Dec-MCTS framework from "Multi-Horizon Multi-Agent Planning Using
Decentralised Monte Carlo Tree Search" with:

1. Core Objects:
   - Belief: Shared occupancy/uncertainty maps updated via LBP fusion
   - Intent: Current best solution (state sequence) from each planner level

2. Two-Level Planning:
   - LLP (Low-Level Planner): Short-horizon, detailed motion planning, IG-based reward
   - HLP (High-Level Planner): Long-horizon, region/cluster allocation

3. Intent Sharing:
   - LL-intent: Detailed short-horizon motion plan (primitive actions + footprints)
   - HL-intent: Long-horizon region sequence the agent intends to cover

4. Reward Decomposition:
   - g = g1(LL intents) + g2(all intents)
   - g1: Immediate task quality (IG, coverage)
   - g2: Long-horizon mission estimate

5. Asynchronous Operation:
   - Both planners run independently with continuous intent exchange
   - D-UCT style discounting for handling asynchronous drift
"""

import time
import threading
import queue
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from collections import defaultdict
import copy

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
    # Predicted state sequence [(row, col, altitude), ...]
    state_sequence: List[Tuple[float, float, float]] = field(default_factory=list)
    # Predicted footprint indices at each step [[(imin, imax, jmin, jmax)], ...]
    footprint_sequence: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Expected IG at each step
    ig_sequence: List[float] = field(default_factory=list)
    # Total expected IG for this plan
    total_expected_ig: float = 0.0
    # Timestamp when intent was generated
    timestamp: float = field(default_factory=time.time)
    # Planning horizon (number of steps)
    horizon: int = 5
    # Confidence/value of this plan (from MCTS)
    value: float = 0.0
    # Number of MCTS iterations behind this intent
    iterations: int = 0

    def is_stale(self, max_age: float = 2.0) -> bool:
        """Check if intent is too old to be useful."""
        return time.time() - self.timestamp > max_age

    def staleness_discount(
        self, decay_factor: float = 0.9, threshold_sec: float = 2.0
    ) -> float:
        """
        Compute D-UCT discount factor based on intent age.

        Returns:
        - 1.0 = fresh intent (full influence)
        - <1.0 = stale intent (reduced influence)
        """
        age = time.time() - self.timestamp
        staleness = max(0, age / threshold_sec)
        return decay_factor**staleness

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
    # Sequence of region IDs the agent plans to visit
    region_sequence: List[int] = field(default_factory=list)
    # Estimated time to reach each region
    eta_sequence: List[float] = field(default_factory=list)
    # Estimated completion time for each region
    completion_sequence: List[float] = field(default_factory=list)
    # Priority/score for each region in the plan
    score_sequence: List[float] = field(default_factory=list)
    # Current target region
    current_target_region: Optional[int] = None
    # Center position of current target
    target_center: Optional[Tuple[float, float]] = None
    # Timestamp when intent was generated
    timestamp: float = field(default_factory=time.time)
    # Planning horizon (number of regions)
    horizon: int = 3
    # Value of this plan (from MCTS)
    value: float = 0.0
    # Number of MCTS iterations
    iterations: int = 0

    def is_stale(self, max_age: float = 5.0) -> bool:
        """Check if intent is too old (HLP has longer validity)."""
        return time.time() - self.timestamp > max_age

    def staleness_discount(
        self, decay_factor: float = 0.95, threshold_sec: float = 5.0
    ) -> float:
        """
        Compute D-UCT discount factor based on intent age.

        HLP intents use slower decay since regions change less frequently.
        """
        age = time.time() - self.timestamp
        staleness = max(0, age / threshold_sec)
        return decay_factor**staleness

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
    Thread-safe communication bus for sharing intents between agents.

    Supports:
    - Broadcasting LL and HL intents
    - Receiving latest intents from all teammates
    - Optional belief update sharing
    """

    def __init__(self, num_agents: int, max_history: int = 10):
        """
        Initialize intent bus.

        Args:
            num_agents: Number of agents in the system
            max_history: Number of historical intents to keep per agent
        """
        self.num_agents = num_agents
        self.max_history = max_history

        # Latest intents from each agent
        self._ll_intents: Dict[int, LLIntent] = {}
        self._hl_intents: Dict[int, HLIntent] = {}

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

    def broadcast_ll_intent(self, intent: LLIntent) -> None:
        """Broadcast a low-level intent to all agents."""
        with self._lock:
            self._ll_intents[intent.agent_id] = intent
            self._ll_history[intent.agent_id].append(intent)
            if len(self._ll_history[intent.agent_id]) > self.max_history:
                self._ll_history[intent.agent_id].pop(0)
            self._stats["ll_broadcasts"] += 1

    def broadcast_hl_intent(self, intent: HLIntent) -> None:
        """Broadcast a high-level intent to all agents."""
        with self._lock:
            self._hl_intents[intent.agent_id] = intent
            self._hl_history[intent.agent_id].append(intent)
            if len(self._hl_history[intent.agent_id]) > self.max_history:
                self._hl_history[intent.agent_id].pop(0)
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
        """Get LL intents from all teammates (excluding self)."""
        return self.get_all_ll_intents(exclude_agent=agent_id)

    def get_teammate_hl_intents(self, agent_id: int) -> Dict[int, HLIntent]:
        """Get HL intents from all teammates (excluding self)."""
        return self.get_all_hl_intents(exclude_agent=agent_id)

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
    Low-Level Planner using Dec-MCTS for short-horizon motion planning.

    Features:
    - IG-based reward (g1) with teammate intent consideration
    - Short horizon (5-10 steps)
    - Fast planning cycle
    - Collision avoidance via intent prediction
    """

    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        horizon: int = 5,
        num_iterations: int = 100,
        ucb_c: float = 1.41,
        discount: float = 0.95,
        intent_discount: float = 0.8,  # Discount for teammate intent influence
    ):
        """
        Initialize LLP.

        Args:
            agent_id: This agent's ID
            camera: UAV camera model
            grid_info: Grid information
            horizon: Planning horizon (steps)
            num_iterations: MCTS iterations per planning cycle
            ucb_c: UCB exploration constant
            discount: Reward discount factor
            intent_discount: Discount for teammate intent cells
        """
        self.agent_id = agent_id
        self.camera = camera
        self.grid_info = grid_info
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.ucb_c = ucb_c
        self.discount = discount
        self.intent_discount = intent_discount

        # Current belief map
        self.belief: Optional[np.ndarray] = None

        # Teammate intents
        self._teammate_ll_intents: Dict[int, LLIntent] = {}
        self._teammate_hl_intents: Dict[int, HLIntent] = {}

        # HLP guidance (from own agent's HLP)
        self._hl_guidance: Optional[HLIntent] = None

        # MCTS tree (persistent across cycles for D-UCT)
        self._tree: Dict[str, Any] = {}
        self._tree_valid = False

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

    def update_belief(self, belief: np.ndarray) -> None:
        """Update local belief map."""
        self.belief = belief.copy()

    def update_teammate_intents(
        self,
        ll_intents: Dict[int, LLIntent],
        hl_intents: Dict[int, HLIntent],
    ) -> None:
        """
        Update stored teammate intents.

        This causes the reward landscape to shift (D-UCT style drift).
        """
        self._teammate_ll_intents = ll_intents
        self._teammate_hl_intents = hl_intents
        self._stats["intent_updates_received"] += 1

        # Intents changed → partial tree invalidation
        # In full D-UCT, we'd discount old statistics; here we mark for re-evaluation
        self._tree_valid = False

    def update_hl_guidance(self, hl_intent: HLIntent) -> None:
        """Update guidance from own HLP."""
        self._hl_guidance = hl_intent

    def _compute_teammate_coverage_mask(self) -> np.ndarray:
        """
        Compute mask of cells teammates plan to cover.

        Returns a discount map where cells covered by teammates have reduced IG.
        """
        H, W = self.grid_info.shape
        coverage_discount = np.ones((H, W), dtype=np.float32)

        for teammate_id, ll_intent in self._teammate_ll_intents.items():
            if ll_intent.is_stale():
                continue

            # D-UCT: Apply staleness discount for asynchronous drift
            staleness_factor = ll_intent.staleness_discount()

            # Apply decreasing discount for future steps
            for step_idx, fp in enumerate(ll_intent.footprint_sequence):
                step_discount = self.intent_discount**step_idx * staleness_factor
                imin, imax, jmin, jmax = fp
                # Clamp to valid bounds
                imin = max(0, min(imin, H))
                imax = max(0, min(imax, H))
                jmin = max(0, min(jmin, W))
                jmax = max(0, min(jmax, W))

                # Reduce IG in cells teammate will cover
                coverage_discount[imin:imax, jmin:jmax] *= step_discount

        return coverage_discount

    def _compute_ig(
        self,
        position: Tuple[float, float],
        altitude: float,
        coverage_discount: np.ndarray,
    ) -> float:
        """
        Compute Information Gain for a position, considering teammate intents.

        IG is reduced for cells that teammates plan to observe.
        """
        if self.belief is None:
            return 0.0

        # Get camera footprint
        try:
            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=position,
                altitude=altitude,
                index_form=True,
            )
        except Exception:
            return 0.0

        H, W = self.belief.shape[:2]
        imin = max(0, min(imin, H))
        imax = max(0, min(imax, H))
        jmin = max(0, min(jmin, W))
        jmax = max(0, min(jmax, W))

        if imax <= imin or jmax <= jmin:
            return 0.0

        # Extract belief probabilities (occupied channel)
        if self.belief.ndim == 3:
            belief_slice = self.belief[imin:imax, jmin:jmax, 1]
        else:
            belief_slice = self.belief[imin:imax, jmin:jmax]

        # Compute entropy (uncertainty)
        eps = 1e-10
        p = np.clip(belief_slice, eps, 1 - eps)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        # Apply teammate coverage discount
        discount_slice = coverage_discount[imin:imax, jmin:jmax]
        discounted_entropy = entropy * discount_slice

        return float(np.sum(discounted_entropy))

    def _compute_g2_for_trajectory(
        self,
        state_sequence: List[Tuple[float, float, float]],
        footprint_sequence: List[Tuple[int, int, int, int]],
        ig_sequence: List[float],
    ) -> float:
        """
        Compute g2 (mission time estimate) for a candidate trajectory.

        Uses the centralized g2() function from g2_evaluator with:
        - This agent's trajectory as LL intent
        - Own HLP intent
        - Teammate LL and HL intents

        Args:
            state_sequence: Predicted states from trajectory
            footprint_sequence: Predicted footprints from trajectory
            ig_sequence: Predicted IG values from trajectory

        Returns:
            g2_value: Mission time estimate (lower is better)
        """
        from g2_evaluator import g2

        # Build temporary LL intent for this trajectory
        temp_ll_intent = LLIntent(
            agent_id=self.agent_id,
            action_sequence=[],  # Not needed for g2
            state_sequence=state_sequence,
            footprint_sequence=footprint_sequence,
            ig_sequence=ig_sequence,
            total_expected_ig=sum(ig_sequence) if ig_sequence else 0.0,
        )

        # Combine own trajectory with teammate LL intents
        all_ll_intents = dict(self._teammate_ll_intents)
        all_ll_intents[self.agent_id] = temp_ll_intent

        # Combine own HL guidance with teammate HL intents
        all_hl_intents = dict(self._teammate_hl_intents)
        if self._hl_guidance is not None:
            all_hl_intents[self.agent_id] = self._hl_guidance

        # Build env_state for g2
        env_state = {
            "belief": self.belief,
            "grid_info": self.grid_info,
        }

        # Compute g2
        g2_value = g2(all_ll_intents, all_hl_intents, env_state, agent_id=self.agent_id)

        return g2_value

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

        # Compute IG
        ig = self._compute_ig(next_pos, next_alt, coverage_discount)

        # Restore camera state
        self.camera.set_position(original_pos)
        self.camera.set_altitude(original_alt)

        return ig

    def _compute_alignment_bonus(
        self,
        position: Tuple[float, float],
        prev_position: Tuple[float, float],
    ) -> float:
        """
        Compute alignment bonus with HLP guidance.

        Returns positive bonus if moving toward HLP target, 0 otherwise (soft guidance).
        """
        if self._hl_guidance is None or self._hl_guidance.target_center is None:
            return 0.0

        target = self._hl_guidance.target_center

        # Distance before and after move
        dist_before = np.sqrt(
            (prev_position[0] - target[0]) ** 2 + (prev_position[1] - target[1]) ** 2
        )
        dist_after = np.sqrt(
            (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
        )

        # Soft guidance: bonus for moving closer, no penalty for moving away
        improvement = dist_before - dist_after
        if improvement > 0:
            # Normalize by max possible improvement
            max_dist = np.sqrt(
                self.grid_info.shape[0] ** 2 + self.grid_info.shape[1] ** 2
            )
            return 0.3 * improvement / max_dist

        return 0.0

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
        Simulate a trajectory and compute total reward (g1 + g2).

        Paper-correct reward decomposition:
            r_LLP = g1(LL prefix) + g2(LL prefix, HL intents, teammate intents)

        Where:
        - g1: Immediate task quality (sum of discounted IG)
        - g2: Mission completion time estimate (from g2_evaluator)

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

        # Temporarily set camera state
        org = self.camera.get_x()
        original_pos, original_alt = org.position, org.altitude

        self.camera.set_position(current_pos)
        self.camera.set_altitude(current_alt)

        for step_idx, action in enumerate(actions):
            # Get next state
            future_state = self.camera.x_future(action)
            if future_state is None:
                break

            # x_future returns (position, altitude) where position is (x, y)
            next_pos = future_state[0]  # This is (x, y) tuple
            next_alt = future_state[1]  # This is altitude

            # Compute IG for this step (g1)
            ig = self._compute_ig(next_pos, next_alt, coverage_discount)
            step_reward = (self.discount**step_idx) * ig
            g1_reward += step_reward

            # Record trajectory
            state_sequence.append((next_pos[0], next_pos[1], next_alt))
            ig_sequence.append(ig)

            # Get footprint
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=next_pos, altitude=next_alt, index_form=True
                )
                footprint_sequence.append((imin, imax, jmin, jmax))
            except Exception:
                footprint_sequence.append((0, 0, 0, 0))

            # Update camera for next step
            self.camera.set_position(next_pos)
            self.camera.set_altitude(next_alt)
            current_pos = next_pos
            current_alt = next_alt

        # Restore camera state
        self.camera.set_position(original_pos)
        self.camera.set_altitude(original_alt)

        # Compute g2: mission completion time estimate
        # Use this trajectory as the LL intent for g2 evaluation
        g2_value = self._compute_g2_for_trajectory(
            state_sequence, footprint_sequence, ig_sequence
        )

        # Total LLP reward = g1 + g2
        total_reward = g1_reward + g2_value

        return total_reward, state_sequence, footprint_sequence, ig_sequence

    def plan(self, current_state: Tuple[float, float, float]) -> LLIntent:
        """
        Run MCTS planning and generate LL intent.

        Args:
            current_state: (x, y, altitude) current position

        Returns:
            LLIntent with planned trajectory
        """
        if self.belief is None:
            # Return empty intent
            return LLIntent(agent_id=self.agent_id)

        # Compute teammate coverage discount
        coverage_discount = self._compute_teammate_coverage_mask()

        # Track per-action scores for logging
        self._action_scores = {}  # IG scores (single-step)
        self._mcts_action_values = {
            action: float("-inf") for action in self.actions
        }  # Best rollout value per first action

        # Simple MCTS: evaluate random rollouts and pick best
        best_reward = float("-inf")
        best_actions = []
        best_states = []
        best_footprints = []
        best_igs = []

        # First, compute single-step scores for each action (for reference)
        for action in self.actions:
            ig = self._evaluate_single_action(current_state, action, coverage_discount)
            self._action_scores[action] = ig

        for _ in range(self.num_iterations):
            # Random action sequence
            actions = [np.random.choice(self.actions) for _ in range(self.horizon)]

            # Simulate and evaluate
            reward, states, footprints, igs = self._simulate_trajectory(
                current_state, actions, coverage_discount
            )

            # Track best rollout value for each first action (this is what MCTS actually uses)
            first_action = actions[0]
            if reward > self._mcts_action_values[first_action]:
                self._mcts_action_values[first_action] = reward

            if reward > best_reward:
                best_reward = reward
                best_actions = actions
                best_states = states
                best_footprints = footprints
                best_igs = igs

        # Create intent
        intent = LLIntent(
            agent_id=self.agent_id,
            action_sequence=best_actions,
            state_sequence=best_states,
            footprint_sequence=best_footprints,
            ig_sequence=best_igs,
            total_expected_ig=sum(best_igs) if best_igs else 0.0,
            horizon=self.horizon,
            value=best_reward,
            iterations=self.num_iterations,
        )

        self.current_intent = intent
        self._stats["plans_generated"] += 1
        self._stats["total_iterations"] += self.num_iterations

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
    High-Level Planner using Dec-MCTS for long-horizon region allocation.

    Features:
    - Region-based planning with g2 reward
    - Long horizon (3-5 regions)
    - Slow planning cycle with tree persistence
    - Considers all agents' intents for task allocation
    """

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        grid_shape: Tuple[int, int],
        tile_size: Tuple[int, int] = (100, 100),
        horizon: int = 3,
        num_iterations: int = 50,
        replan_interval: float = 1.0,
    ):
        """
        Initialize HLP.

        Args:
            agent_id: This agent's ID
            num_agents: Total number of agents
            grid_shape: (H, W) shape of the grid
            tile_size: Size of region tiles
            horizon: Number of regions to plan ahead
            num_iterations: MCTS iterations per cycle
            replan_interval: Minimum time between replans
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.grid_shape = grid_shape
        self.tile_size = tile_size
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.replan_interval = replan_interval

        # Partition grid into regions
        self.regions = self._partition_grid()
        self.num_regions = len(self.regions)

        # Current belief
        self.belief: Optional[np.ndarray] = None

        # Teammate intents
        self._teammate_ll_intents: Dict[int, LLIntent] = {}
        self._teammate_hl_intents: Dict[int, HLIntent] = {}

        # Current intent
        self.current_intent: Optional[HLIntent] = None

        # Last replan time
        self._last_replan_time: float = 0.0

        # Region coverage estimates
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
        """Partition grid into rectangular regions."""
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

    def update_belief(self, belief: np.ndarray) -> None:
        """Update local belief and recompute region coverage."""
        self.belief = belief.copy()
        self._update_region_coverage()

    def _update_region_coverage(self) -> None:
        """Recompute coverage estimate for each region."""
        if self.belief is None:
            return

        # Extract probability map
        if self.belief.ndim == 3:
            prob_map = self.belief[:, :, 1]
        else:
            prob_map = self.belief

        # Compute entropy for coverage estimate
        eps = 1e-10
        p = np.clip(prob_map, eps, 1 - eps)
        entropy_map = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        for region_id, region in self.regions.items():
            (imin, imax), (jmin, jmax) = region["bounds"]
            region_entropy = entropy_map[imin:imax, jmin:jmax]

            # Coverage = 1 - normalized entropy
            max_entropy = 1.0  # Binary entropy max
            avg_entropy = np.mean(region_entropy)
            self._region_coverage[region_id] = 1.0 - (avg_entropy / max_entropy)

    def update_teammate_intents(
        self,
        ll_intents: Dict[int, LLIntent],
        hl_intents: Dict[int, HLIntent],
    ) -> None:
        """Update stored teammate intents."""
        self._teammate_ll_intents = ll_intents
        self._teammate_hl_intents = hl_intents
        self._stats["intent_updates_received"] += 1

    def _get_teammate_target_regions(self) -> Dict[int, Tuple[Set[int], float]]:
        """Get regions each teammate is targeting with D-UCT staleness discount."""
        targets = {}
        for teammate_id, hl_intent in self._teammate_hl_intents.items():
            if not hl_intent.is_stale():
                staleness_discount = hl_intent.staleness_discount()
                targets[teammate_id] = (
                    hl_intent.get_target_regions(),
                    staleness_discount,
                )
            else:
                targets[teammate_id] = (set(), 0.0)
        return targets

    def _estimate_region_completion_time(
        self,
        region_id: int,
        agent_position: Tuple[float, float],
    ) -> float:
        """Estimate time to complete a region."""
        region = self.regions[region_id]
        center = region["center"]

        # Travel time (simplified)
        distance = np.sqrt(
            (agent_position[0] - center[0]) ** 2 + (agent_position[1] - center[1]) ** 2
        )
        travel_time = distance / 10.0  # Assume speed of 10 units/step

        # Coverage time based on remaining entropy
        remaining_coverage = 1.0 - self._region_coverage[region_id]
        coverage_time = remaining_coverage * region["area"] / 100.0

        return travel_time + coverage_time

    def _compute_region_score(
        self,
        region_id: int,
        agent_position: Tuple[float, float],
        teammate_targets: Dict[int, Tuple[Set[int], float]],
    ) -> float:
        """
        Compute marginal g2 score for a region.

        Paper-correct HLP reward (Eq. 7):
            r_HLP = g2(with my HL intent) - g2(with null HL intent)

        Where null HL intent = no regions allocated to this agent.

        Args:
            region_id: Region to evaluate
            agent_position: Current agent position
            teammate_targets: Dict of teammate HL intents

        Returns:
            Marginal contribution score (positive = helpful, negative = harmful)
        """
        from g2_evaluator import g2

        # Build HL intent WITH this region
        hl_with = HLIntent(
            agent_id=self.agent_id,
            region_sequence=[region_id],
            current_target_region=region_id,
            target_center=self.regions[region_id]["center"],
        )

        # Build HL intent WITHOUT any regions (null intent)
        hl_null = HLIntent(
            agent_id=self.agent_id,
            region_sequence=[],
            current_target_region=None,
            target_center=None,
        )

        # Prepare env_state for g2
        env_state = {
            "belief": self.belief,
            "grid_info": None,  # g2 doesn't need grid_info for minimal implementation
        }

        # Get teammate HL intents
        all_hl_with = dict(self._teammate_hl_intents)
        all_hl_with[self.agent_id] = hl_with

        all_hl_null = dict(self._teammate_hl_intents)
        all_hl_null[self.agent_id] = hl_null

        # Get LL intents (same for both scenarios)
        all_ll = dict(self._teammate_ll_intents)

        # Compute g2 with and without this region
        g2_with = g2(all_ll, all_hl_with, env_state, agent_id=self.agent_id)
        g2_null = g2(all_ll, all_hl_null, env_state, agent_id=self.agent_id)

        # Marginal contribution: g2_null - g2_with
        # (Lower g2 is better, so positive marginal score means this region helps)
        marginal_score = g2_null - g2_with

        return marginal_score

    def _should_replan(self) -> bool:
        """Check if we should replan."""
        if self.current_intent is None:
            return True

        if time.time() - self._last_replan_time < self.replan_interval:
            return False

        # Check if current target is mostly covered
        if self.current_intent.current_target_region is not None:
            coverage = self._region_coverage.get(
                self.current_intent.current_target_region, 0.0
            )
            if coverage > 0.9:
                return True

        return True

    def plan(self, current_position: Tuple[float, float]) -> HLIntent:
        """
        Run HLP planning with MCTS and generate HL intent.

        Uses MCTS tree search over region sequences:
        - State: visited regions so far
        - Action: next region to visit
        - Reward: marginal g2 contribution
        - Rollout: random region selection

        Args:
            current_position: (row, col) current position

        Returns:
            HLIntent with planned region sequence
        """
        if not self._should_replan() and self.current_intent is not None:
            return self.current_intent

        self._last_replan_time = time.time()

        # Get teammate targets
        teammate_targets = self._get_teammate_target_regions()

        # Run MCTS over region sequences
        best_sequence = self._run_mcts_region_search(current_position, teammate_targets)

        if not best_sequence:
            # Fallback: no valid regions found
            intent = HLIntent(
                agent_id=self.agent_id,
                region_sequence=[],
                eta_sequence=[],
                completion_sequence=[],
                score_sequence=[],
                current_target_region=None,
                target_center=None,
                horizon=self.horizon,
                value=0.0,
                iterations=self.num_iterations,
            )
            self.current_intent = intent
            self._stats["plans_generated"] += 1
            return intent

        # Build intent from best sequence
        region_sequence = best_sequence
        eta_sequence = []
        completion_sequence = []
        score_sequence = []

        cumulative_time = 0.0
        pos = current_position

        for region_id in region_sequence:
            # Compute marginal score for this region
            score = self._compute_region_score(region_id, pos, teammate_targets)

            # Estimate completion time
            eta = self._estimate_region_completion_time(region_id, pos)
            cumulative_time += eta

            eta_sequence.append(cumulative_time)
            completion_sequence.append(cumulative_time + eta)
            score_sequence.append(score)

            # Update position for next region
            pos = self.regions[region_id]["center"]

        # Create intent
        current_target = region_sequence[0] if region_sequence else None
        target_center = None
        if current_target is not None:
            target_center = self.regions[current_target]["center"]

        intent = HLIntent(
            agent_id=self.agent_id,
            region_sequence=region_sequence,
            eta_sequence=eta_sequence,
            completion_sequence=completion_sequence,
            score_sequence=score_sequence,
            current_target_region=current_target,
            target_center=target_center,
            horizon=self.horizon,
            value=sum(score_sequence),
            iterations=self.num_iterations,
        )

        self.current_intent = intent
        self._stats["plans_generated"] += 1
        self._stats["total_iterations"] += self.num_iterations

        return intent

    def _run_mcts_region_search(
        self,
        start_position: Tuple[float, float],
        teammate_targets: Dict[int, Tuple[Set[int], float]],
    ) -> List[int]:
        """
        Run MCTS tree search over region sequences.

        State: frozenset of visited regions
        Action: next region to visit
        Reward: cumulative marginal g2

        Args:
            start_position: Starting position
            teammate_targets: Teammate region targets

        Returns:
            Best region sequence found
        """
        # MCTS tree: {state_key: {'visits': int, 'value': float, 'children': dict}}
        tree = {}
        root_state = frozenset()  # No regions visited yet
        tree[root_state] = {"visits": 0, "value": 0.0, "children": {}}

        best_sequence = []
        best_value = float("-inf")

        for iteration in range(self.num_iterations):
            # Selection + Expansion + Simulation
            sequence, value = self._mcts_iteration(
                tree, root_state, start_position, teammate_targets
            )

            # Track best sequence found
            if value > best_value:
                best_value = value
                best_sequence = sequence

        return best_sequence

    def _mcts_iteration(
        self,
        tree: Dict,
        root_state: frozenset,
        start_position: Tuple[float, float],
        teammate_targets: Dict[int, Tuple[Set[int], float]],
    ) -> Tuple[List[int], float]:
        """
        Single MCTS iteration: selection, expansion, simulation, backpropagation.

        Returns:
            (sequence, value) tuple
        """
        # Selection phase: traverse tree using UCB
        state = root_state
        position = start_position
        path = []  # List of (state, action) tuples
        sequence = []  # List of region IDs

        while len(sequence) < self.horizon:
            if state not in tree:
                tree[state] = {"visits": 0, "value": 0.0, "children": {}}

            node = tree[state]
            node["visits"] += 1

            # Get available actions (unvisited regions)
            available_regions = self._get_available_regions(state)

            if not available_regions:
                break

            # Check if this is a leaf node (unexpanded)
            unexplored = [r for r in available_regions if r not in node["children"]]

            if unexplored:
                # Expansion: pick random unexplored action
                action = random.choice(unexplored)
                node["children"][action] = {"visits": 0, "value": 0.0}

                # Simulate from this new node
                sequence.append(action)
                path.append((state, action))

                # Rollout to get value estimate
                value = self._rollout_region_sequence(
                    sequence, position, start_position, teammate_targets
                )

                # Backpropagate
                self._backpropagate(tree, path, value)

                return sequence, value
            else:
                # Selection: use UCB to pick best child
                action = self._select_best_region_ucb(node, available_regions)
                sequence.append(action)
                path.append((state, action))

                # Transition to next state
                state = frozenset(sequence)
                position = self.regions[action]["center"]

        # Terminal node reached (horizon limit or no more regions)
        value = self._evaluate_region_sequence(
            sequence, start_position, teammate_targets
        )
        self._backpropagate(tree, path, value)

        return sequence, value

    def _get_available_regions(self, visited: frozenset) -> List[int]:
        """Get regions not yet visited in current path."""
        return [r for r in self.regions.keys() if r not in visited]

    def _select_best_region_ucb(self, node: Dict, available_regions: List[int]) -> int:
        """Select region using UCB formula."""
        parent_visits = node["visits"]
        ucb_c = 1.0  # Exploration constant for HLP

        best_score = float("-inf")
        best_region = available_regions[0]

        for region_id in available_regions:
            if region_id not in node["children"]:
                continue

            child = node["children"][region_id]
            child_visits = child["visits"]
            child_value = child["value"]

            if child_visits == 0:
                ucb_score = float("inf")
            else:
                exploitation = child_value / child_visits
                exploration = ucb_c * np.sqrt(np.log(parent_visits) / child_visits)
                ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_region = region_id

        return best_region

    def _rollout_region_sequence(
        self,
        current_sequence: List[int],
        current_position: Tuple[float, float],
        start_position: Tuple[float, float],
        teammate_targets: Dict[int, Tuple[Set[int], float]],
    ) -> float:
        """
        Rollout policy: randomly complete region sequence to horizon.

        Args:
            current_sequence: Regions selected so far
            current_position: Current position after current_sequence
            start_position: Initial starting position
            teammate_targets: Teammate region targets

        Returns:
            Total value estimate
        """
        sequence = list(current_sequence)
        position = current_position
        visited = set(sequence)

        # Random completion to horizon
        while len(sequence) < self.horizon:
            available = [r for r in self.regions.keys() if r not in visited]
            if not available:
                break

            # Pick random region (simple rollout policy)
            region_id = random.choice(available)
            sequence.append(region_id)
            visited.add(region_id)
            position = self.regions[region_id]["center"]

        # Evaluate complete sequence
        return self._evaluate_region_sequence(
            sequence, start_position, teammate_targets
        )

    def _evaluate_region_sequence(
        self,
        sequence: List[int],
        start_position: Tuple[float, float],
        teammate_targets: Dict[int, Tuple[Set[int], float]],
    ) -> float:
        """
        Evaluate cumulative marginal g2 for a region sequence.

        Args:
            sequence: Region sequence to evaluate
            start_position: Starting position
            teammate_targets: Teammate region targets

        Returns:
            Total marginal g2 value (higher is better)
        """
        if not sequence:
            return 0.0

        from g2_evaluator import g2

        # Build HL intent with full sequence
        hl_with_sequence = HLIntent(
            agent_id=self.agent_id,
            region_sequence=sequence,
            current_target_region=sequence[0],
            target_center=self.regions[sequence[0]]["center"],
        )

        # Build null intent (no regions)
        hl_null = HLIntent(
            agent_id=self.agent_id,
            region_sequence=[],
            current_target_region=None,
            target_center=None,
        )

        # Prepare env_state
        env_state = {
            "belief": self.belief,
            "grid_info": None,
        }

        # Get teammate HL intents
        all_hl_with = dict(self._teammate_hl_intents)
        all_hl_with[self.agent_id] = hl_with_sequence

        all_hl_null = dict(self._teammate_hl_intents)
        all_hl_null[self.agent_id] = hl_null

        # Get LL intents
        all_ll = dict(self._teammate_ll_intents)

        # Compute marginal g2
        g2_with = g2(all_ll, all_hl_with, env_state, agent_id=self.agent_id)
        g2_null = g2(all_ll, all_hl_null, env_state, agent_id=self.agent_id)

        # Marginal value (positive = helpful)
        marginal_value = g2_null - g2_with

        return marginal_value

    def _backpropagate(
        self, tree: Dict, path: List[Tuple[frozenset, int]], value: float
    ) -> None:
        """
        Backpropagate value up the tree.

        Args:
            tree: MCTS tree
            path: List of (state, action) tuples traversed
            value: Value to backpropagate
        """
        for state, action in path:
            if state not in tree:
                continue
            node = tree[state]
            if action in node["children"]:
                child = node["children"][action]
                child["visits"] += 1
                child["value"] += value

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return dict(self._stats)


# =============================================================================
# Hierarchical Dec-MCTS Planner (Combined)
# =============================================================================


class HierarchicalDecMCTSPlanner:
    """
    Main hierarchical planner combining LLP and HLP with Dec-MCTS.

    Orchestrates:
    - Two-level planning (LLP + HLP)
    - Asynchronous intent sharing via IntentBus
    - Belief updates
    - Reward decomposition (g = g1 + g2)
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
        hlp_horizon: int = 3,
        hlp_iterations: int = 50,
        tile_size: Tuple[int, int] = (100, 100),
        hlp_replan_interval: float = 1.0,
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
            llp_iterations: LLP MCTS iterations
            hlp_horizon: HLP planning horizon (regions)
            hlp_iterations: HLP MCTS iterations
            tile_size: Region tile size for HLP
            hlp_replan_interval: Minimum time between HLP replans
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.camera = camera
        self.grid_info = grid_info
        self.intent_bus = intent_bus

        # Create LLP
        self.llp = LowLevelPlanner(
            agent_id=agent_id,
            camera=camera,
            grid_info=grid_info,
            horizon=llp_horizon,
            num_iterations=llp_iterations,
        )

        # Create HLP
        self.hlp = HighLevelPlanner(
            agent_id=agent_id,
            num_agents=num_agents,
            grid_shape=grid_info.shape,
            tile_size=tile_size,
            horizon=hlp_horizon,
            num_iterations=hlp_iterations,
            replan_interval=hlp_replan_interval,
        )

        # Current state
        self._current_position: Optional[Tuple[float, float]] = None
        self._current_altitude: float = 0.0

        # Statistics
        self._stats = {
            "planning_cycles": 0,
            "ll_intents_broadcast": 0,
            "hl_intents_broadcast": 0,
        }

        # Region metadata for visualization (exposed as properties)
        self._current_region_metadata: Optional[Dict] = None
        self._current_selected_region: Optional[int] = None
        self._current_region_scores: Optional[Dict[int, float]] = None

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
    ) -> None:
        """
        Update planner state with current position and belief.

        Args:
            position: Current (x, y) position
            altitude: Current altitude
            belief: Current belief map
        """
        self._current_position = position
        self._current_altitude = altitude

        # Update both planners with belief
        self.llp.update_belief(belief)
        self.hlp.update_belief(belief)

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

    def plan(self) -> Tuple[str, Dict[str, Any]]:
        """
        Run one planning cycle (both LLP and HLP).

        Following the Dec-MCTS pattern:
        1. Receive teammate intents
        2. Run HLP (slow cycle, may reuse cached plan)
        3. Update LLP with HLP guidance
        4. Run LLP (fast cycle)
        5. Broadcast intents

        Returns:
            (best_action, metrics_dict)
        """
        if self._current_position is None:
            return "hover", {}

        # Step 1: Receive teammate intents
        self.receive_intents()

        # Step 2: Run HLP
        # Convert position to grid coordinates for HLP
        grid_pos = self.camera.convert_xy_ij(
            self._current_position[0],
            self._current_position[1],
            self.camera.grid.center,
        )
        hl_intent = self.hlp.plan((grid_pos[0], grid_pos[1]))

        # Step 3: Update LLP with HLP guidance
        # Convert target_center from grid coords to world coords for LLP
        if hl_intent.target_center is not None:
            # Create a copy with world coordinates
            hl_intent_for_llp = copy.copy(hl_intent)
            grid_center = hl_intent.target_center
            # Convert (row, col) grid indices to (x, y) world coordinates
            world_x, world_y = self.camera.ij_to_xy(grid_center[0], grid_center[1])
            hl_intent_for_llp.target_center = (world_x, world_y)
            self.llp.update_hl_guidance(hl_intent_for_llp)
        else:
            self.llp.update_hl_guidance(hl_intent)

        # Step 4: Run LLP
        current_state = (
            self._current_position[0],
            self._current_position[1],
            self._current_altitude,
        )
        ll_intent = self.llp.plan(current_state)

        # Step 5: Broadcast intents
        self.intent_bus.broadcast_ll_intent(ll_intent)
        self.intent_bus.broadcast_hl_intent(hl_intent)
        self._stats["ll_intents_broadcast"] += 1
        self._stats["hl_intents_broadcast"] += 1

        # Get best action
        best_action = self.llp.get_best_action()

        # Collect detailed scoring info for logging
        llp_action_scores = getattr(self.llp, "_action_scores", {})
        mcts_action_values = getattr(self.llp, "_mcts_action_values", {})
        hlp_region_scores = dict(self.hlp._region_coverage)

        # Get teammate intent summary
        ll_intents = self.intent_bus.get_teammate_ll_intents(self.agent_id)
        hl_intents = self.intent_bus.get_teammate_hl_intents(self.agent_id)
        intents_received = {
            "ll_intents_from": list(ll_intents.keys()),
            "hl_intents_from": list(hl_intents.keys()),
        }
        for tid, hl_int in hl_intents.items():
            intents_received[f"agent_{tid}_target"] = hl_int.current_target_region

        # Compile metrics
        metrics = {
            "ll_intent": ll_intent,
            "hl_intent": hl_intent,
            "action": best_action,
            "ll_value": ll_intent.value,
            "hl_value": hl_intent.value,
            "target_region": hl_intent.current_target_region,
            "expected_ig": ll_intent.total_expected_ig,
            # Detailed scoring for logging
            "llp_action_scores": llp_action_scores,  # Single-step IG (g1 only)
            "mcts_action_values": mcts_action_values,  # MCTS rollout values (actual decision basis)
            "hlp_region_scores": hlp_region_scores,
            "intents_received": intents_received,
        }

        self._stats["planning_cycles"] += 1

        return best_action, metrics

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
    llp_horizon = config.get("llp_horizon", 5)
    llp_iterations = config.get("llp_iterations", 100)
    hlp_horizon = config.get("hlp_horizon", 3)
    hlp_iterations = config.get("hlp_iterations", 50)
    tile_size = tuple(config.get("tile_size", [100, 100]))
    hlp_replan_interval = config.get("hlp_replan_interval", 1.0)

    return HierarchicalDecMCTSPlanner(
        agent_id=agent_id,
        num_agents=num_agents,
        camera=camera,
        grid_info=grid_info,
        intent_bus=intent_bus,
        llp_horizon=llp_horizon,
        llp_iterations=llp_iterations,
        hlp_horizon=hlp_horizon,
        hlp_iterations=hlp_iterations,
        tile_size=tile_size,
        hlp_replan_interval=hlp_replan_interval,
    )

"""
Greedy Information Gain (IG) Planner for Multi-UAV Active Sensing

This module implements the greedy one-step lookahead IG planner as described in:
"Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

The greedy IG approach serves as a baseline/benchmark:
1. Single-step lookahead: Only considers immediate information gain
2. No trajectory planning: Myopic decision making
3. Pure belief-based coordination: No overlap penalties
4. Fast: No MCTS tree search overhead

Paper's Approach to Multi-Agent Coordination:
- Each agent maintains its own local belief map
- Agents share positions (to determine neighbors) and news beliefs
- Coordination emerges from Bayesian fusion of news beliefs
- NO penalty terms - pure IG computed on FUSED belief
- Areas observed by teammates have lower uncertainty -> lower IG -> natural dispersion

Key characteristics:
- Computes IG for each primitive action using FUSED belief
- Selects action with maximum IG (no penalties)
- Simple and fast, but doesn't consider future rewards

Reference: Equation (1) in the paper for IG definition
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import time
import logging
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def H(p: np.ndarray) -> np.ndarray:
    """
    Compute binary entropy H(p) for occupancy probabilities.

    Args:
        p: Probability array (values in [0, 1])

    Returns:
        Entropy values (same shape as p)
    """
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def cH(p: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """
    Compute conditional entropy given sensor model parameters.

    Using the sensor model:
    - P(z=1|m=1) = 1 - s1 (true positive rate)
    - P(z=1|m=0) = s0 (false positive rate)

    Args:
        p: Prior probability P(m=1)
        s0: False positive rate
        s1: False negative rate

    Returns:
        Conditional entropy H(m|z)
    """
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)

    # P(z=1) = p*(1-s1) + (1-p)*s0
    pz1 = p * (1 - s1) + (1 - p) * s0
    pz0 = 1 - pz1

    # Posterior P(m=1|z=1)
    pm1_z1 = np.where(pz1 > eps, p * (1 - s1) / pz1, 0.5)
    # Posterior P(m=1|z=0)
    pm1_z0 = np.where(pz0 > eps, p * s1 / pz0, 0.5)

    # Conditional entropy H(m|z) = P(z=1)*H(m|z=1) + P(z=0)*H(m|z=0)
    H_m_z1 = H(pm1_z1)
    H_m_z0 = H(pm1_z0)

    return pz1 * H_m_z1 + pz0 * H_m_z0


def compute_footprint_iou(
    footprint1: Tuple[int, int, int, int],
    footprint2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) of two footprints.

    Args:
        footprint1: (imin, imax, jmin, jmax) for agent 1
        footprint2: (imin, imax, jmin, jmax) for agent 2

    Returns:
        IoU value in [0, 1]
    """
    imin1, imax1, jmin1, jmax1 = footprint1
    imin2, imax2, jmin2, jmax2 = footprint2

    # Compute intersection
    inter_imin = max(imin1, imin2)
    inter_imax = min(imax1, imax2)
    inter_jmin = max(jmin1, jmin2)
    inter_jmax = min(jmax1, jmax2)

    # Check if there's any intersection
    if inter_imax <= inter_imin or inter_jmax <= inter_jmin:
        return 0.0

    # Compute areas
    intersection_area = (inter_imax - inter_imin) * (inter_jmax - inter_jmin)
    area1 = (imax1 - imin1) * (jmax1 - jmin1)
    area2 = (imax2 - imin2) * (jmax2 - jmin2)
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area <= 0:
        return 0.0

    iou = intersection_area / union_area
    return float(np.clip(iou, 0.0, 1.0))


# =============================================================================
# Data Structures for Action Return
# =============================================================================


@dataclass
class GreedyIGDecision:
    """
    Decision output from greedy IG planner.

    Paper's approach:
    - IG: Pure belief-based coordination via news sharing
    - IGd: Null policy assumption (teammates stay at current positions)
           Footprint overlap discount based on CURRENT teammate positions

    NO intent-based future position prediction!
    """

    agent_id: int
    action: str = "hover"
    position: Tuple[float, float] = (0.0, 0.0)
    altitude: float = 0.0
    expected_ig: float = 0.0
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Greedy IG Planner
# =============================================================================


class GreedyIGPlanner:
    """
    Greedy one-step lookahead Information Gain planner.

    This is the simplest IG-based planner:
    1. For each action, compute expected IG from resulting position
    2. Select action with maximum IG
    3. No penalty terms - pure belief-based IG

    Multi-agent coordination (paper's approach):
    - Planner receives FUSED belief (after news sharing with neighbors)
    - IG computed on fused belief naturally accounts for teammates' observations
    - Areas teammates observed have lower entropy -> lower IG -> dispersion
    - No explicit overlap penalties or intent-based coordination
    """

    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        conf_dict: Optional[Dict] = None,
        intent_discount: float = 0.0,
        overlap_penalty_weight: float = 0.0,
        enable_discounting: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize greedy IG planner.

        Args:
            agent_id: This agent's ID
            camera: UAV camera model
            grid_info: Grid information
            conf_dict: Sensor model parameters by altitude
            intent_discount: DEPRECATED - kept for compatibility, not used
            overlap_penalty_weight: DEPRECATED - kept for compatibility, should be 0.0
            enable_discounting: If True, use IGd (footprint-based IoU discounting)
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.camera = camera
        self.grid_info = grid_info
        self.conf_dict = conf_dict
        # Paper approach: no penalties, pure belief-based IG
        # These are kept for backward compatibility but should be 0.0
        self.intent_discount = intent_discount
        self.overlap_penalty_weight = overlap_penalty_weight
        # IGd option: footprint-based discounting
        self.enable_discounting = enable_discounting

        # Random number generator for reproducible tie-breaking
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Available actions
        self.actions = ["front", "back", "left", "right", "up", "down", "hover"]

        # Current belief
        self.belief: Optional[np.ndarray] = None

        # Teammate current states (for IGd null policy assumption)
        # Maps teammate_id -> (position, altitude, footprint)
        self._teammate_states: Dict[
            int, Tuple[Tuple[float, float], float, Tuple[int, int, int, int]]
        ] = {}

        # Current decision (for returning)
        self.current_decision: Optional[GreedyIGDecision] = None

        # Statistics
        self._stats = {
            "plans_generated": 0,
            "total_ig": 0.0,
            "total_igd": 0.0,
            "teammate_updates_received": 0,
        }

        # Per-action scores for logging
        self._action_scores: Dict[str, float] = {}
        self._raw_ig_scores: Dict[str, float] = {}
        self._overlap_penalties: Dict[str, float] = {}
        self._discount_factors: Dict[str, float] = {}

    def update_belief(self, belief: np.ndarray) -> None:
        """Update local belief map."""
        self.belief = belief.copy()

    def update_teammate_states(
        self, teammate_states: Dict[int, Tuple[Tuple[float, float], float]]
    ) -> None:
        """
        Update teammate current states for IGd null policy assumption.

        IGd assumes teammates remain at current positions (null policy).
        This is used ONLY for footprint-based overlap discounting in IGd mode.

        Args:
            teammate_states: Dict mapping teammate_id -> (position, altitude)
        """
        # Store teammate states with their current footprints
        for tid, (pos, alt) in teammate_states.items():
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=pos,
                    altitude=alt,
                    index_form=True,
                )
                footprint = (imin, imax, jmin, jmax)
            except Exception:
                footprint = (0, 0, 0, 0)

            self._teammate_states[tid] = (pos, alt, footprint)

        self._stats["teammate_updates_received"] += 1

    def _get_sensor_params(self, altitude: float) -> Tuple[float, float]:
        """Get sensor model parameters for given altitude."""
        if self.conf_dict is not None:
            return self.conf_dict[np.round(altitude, decimals=2)]
        else:
            # Default sensor model
            a = 1
            b = 0.015
            sigma = a * (1 - np.exp(-b * altitude))
            return sigma, sigma

    def _compute_ig(
        self,
        position: Tuple[float, float],
        altitude: float,
    ) -> float:
        """
        Compute Information Gain for observing from a position.

        IG = H(prior) - E[H(posterior|observation)]

        Args:
            position: (x, y) position
            altitude: observation altitude

        Returns:
            Total IG for all cells in footprint
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

        H_grid, W_grid = self.belief.shape[:2]
        imin = max(0, min(imin, H_grid))
        imax = max(0, min(imax, H_grid))
        jmin = max(0, min(jmin, W_grid))
        jmax = max(0, min(jmax, W_grid))

        if imax <= imin or jmax <= jmin:
            return 0.0

        # Extract belief in footprint
        if self.belief.ndim == 3:
            prior = self.belief[imin:imax, jmin:jmax, 1]
        else:
            prior = self.belief[imin:imax, jmin:jmax]

        # Get sensor parameters
        s0, s1 = self._get_sensor_params(altitude)

        # Compute IG = H(prior) - E[H(posterior)]
        prior_entropy = H(prior)
        conditional_entropy = cH(prior, s0, s1)
        ig = prior_entropy - conditional_entropy

        return float(np.sum(ig))

    def _compute_discount_factor(
        self,
        my_footprint: Tuple[int, int, int, int],
    ) -> float:
        """
        Compute discount factor based on footprint overlap with neighbors.

        IGd null policy assumption: teammates remain at current positions.

        For each neighbor j:
            α_ij = 1 - IoU(fp(my_next_pos), fp(teammate_current_pos))

        Total discount: Π_{j ∈ neighbors} α_ij

        Args:
            my_footprint: (imin, imax, jmin, jmax) proposed footprint

        Returns:
            Discount factor in [0, 1]
        """
        if not self.enable_discounting or not self._teammate_states:
            return 1.0

        discount = 1.0
        for teammate_id, (
            pos,
            alt,
            teammate_footprint,
        ) in self._teammate_states.items():
            # Compute IoU with teammate's CURRENT footprint (null policy)
            iou = compute_footprint_iou(my_footprint, teammate_footprint)

            # α_ij = 1 - IoU
            alpha_ij = 1.0 - iou

            # Multiply discount factors
            discount *= alpha_ij

        return discount

    def plan(
        self,
        current_position: Tuple[float, float],
        current_altitude: float,
    ) -> GreedyIGDecision:
        """
        Run greedy IG planning using FUSED belief.

        Paper's approach:
        1. IG mode: Pure belief-based coordination via news fusion
           - Compute IG from fused belief for each action
           - Select action with highest IG
           - NO penalties, NO future prediction

        2. IGd mode: Null policy assumption (teammates stay at current positions)
           - Compute IG for each action
           - Discount by footprint IoU with teammates' CURRENT footprints
           - Select action with highest discounted IG

        Args:
            current_position: Current (x, y) position
            current_altitude: Current altitude

        Returns:
            GreedyIGDecision with selected action
        """
        import time

        start_time = time.perf_counter()  # Use high-resolution timer

        if self.belief is None:
            return GreedyIGDecision(agent_id=self.agent_id)

        # Clear logging dicts
        self._action_scores = {}
        self._raw_ig_scores = {}
        self._overlap_penalties = {}
        self._discount_factors = {}

        best_actions = []  # Track all actions with best score for random tie-breaking
        best_score = float("-inf")
        best_ig = 0.0

        # Set camera state for x_future computation
        self.camera.set_position(current_position)
        self.camera.set_altitude(current_altitude)

        for action in self.actions:
            # Get future state
            future_state = self.camera.x_future(action)
            if future_state is None:
                self._action_scores[action] = float("-inf")
                self._raw_ig_scores[action] = 0.0
                self._overlap_penalties[action] = 0.0
                continue

            next_pos, next_alt = future_state

            # Compute IG using FUSED belief (paper's approach)
            ig = self._compute_ig(next_pos, next_alt)
            self._raw_ig_scores[action] = ig

            # Get footprint
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=next_pos,
                    altitude=next_alt,
                    index_form=True,
                )
                footprint = (imin, imax, jmin, jmax)
            except Exception:
                footprint = (0, 0, 0, 0)

            # Compute discount factor (IGd null policy approach)
            # Uses CURRENT teammate positions, not future predictions
            discount = self._compute_discount_factor(footprint)
            self._discount_factors[action] = discount

            # Paper approach: NO overlap penalty
            # Coordination emerges from fused belief, not penalties
            self._overlap_penalties[action] = 0.0

            # Score = IG * discount (if discounting enabled, otherwise discount=1.0)
            score = ig * discount
            self._action_scores[action] = score

            if score > best_score:
                best_score = score
                best_actions = [action]  # New best, reset list
                best_ig = ig
            elif abs(score - best_score) < 1e-10:  # Tie (floating point comparison)
                best_actions.append(action)

        # Randomly break ties if multiple best actions exist
        # Uses self.rng for reproducible random selection
        best_action = self.rng.choice(best_actions) if best_actions else "hover"

        # Stop timer immediately after action selection is complete
        end_time = time.perf_counter()
        planning_time_ms = (end_time - start_time) * 1000.0

        # Save timestamps for external logging
        self._timing_start_ms = start_time * 1000.0
        self._timing_end_ms = end_time * 1000.0

        # Create decision output
        self.current_decision = GreedyIGDecision(
            agent_id=self.agent_id,
            action=best_action,
            position=current_position,
            altitude=current_altitude,
            expected_ig=best_ig,
        )

        self._stats["plans_generated"] += 1
        self._stats["total_ig"] += best_ig
        self._stats["total_igd"] += best_score
        self._stats["last_planning_time_ms"] = planning_time_ms

        return self.current_decision

    def get_best_action(self) -> str:
        """Get the selected action."""
        if self.current_decision:
            return self.current_decision.action
        return "hover"

    def get_action_scores(self) -> Dict[str, float]:
        """Get all action scores (for logging/analysis)."""
        return dict(self._action_scores)

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return dict(self._stats)


# =============================================================================
# Logging Functions
# =============================================================================


def log_greedy_ig_decision(
    agent_id: int,
    step: int,
    raw_ig_scores: Dict[str, float],
    overlap_penalties: Dict[str, float],
    final_scores: Dict[str, float],
    selected_action: str,
    teammate_info: Optional[Dict[str, Any]] = None,
    discount_factors: Optional[Dict[str, float]] = None,
):
    """
    Log greedy IG planning decision.

    Args:
        agent_id: Agent ID
        step: Current step
        raw_ig_scores: Raw IG per action
        overlap_penalties: Overlap penalty per action (should be 0 for paper's approach)
        final_scores: Final scores (IG or IG * discount for IGd)
        selected_action: Selected action
        teammate_info: Optional summary of teammate states (for IGd mode)
        discount_factors: Optional discount factors per action (for IGd)
    """
    logger.info("")
    logger.info(f"{'='*60}")
    mode = (
        "IGd"
        if discount_factors and any(d < 1.0 for d in discount_factors.values())
        else "IG"
    )
    logger.info(f"[Agent {agent_id}] GREEDY {mode} DECISION (Step {step})")
    logger.info(f"{'='*60}")

    logger.info("")
    logger.info("RAW INFORMATION GAIN:")
    sorted_ig = sorted(raw_ig_scores.items(), key=lambda x: x[1], reverse=True)
    for action, ig in sorted_ig:
        logger.info(f"  {action:8s}: {ig:10.2f}")

    if discount_factors and any(d < 1.0 for d in discount_factors.values()):
        logger.info("")
        logger.info(
            "DISCOUNT FACTORS (footprint overlap with neighbor current positions):"
        )
        for action, discount in sorted(discount_factors.items(), key=lambda x: x[1]):
            logger.info(f"  {action:8s}: {discount:10.4f}")

    if any(p > 0 for p in overlap_penalties.values()):
        logger.info("")
        logger.info("OVERLAP PENALTIES (not used in paper's approach):")
        for action, penalty in sorted(
            overlap_penalties.items(), key=lambda x: x[1], reverse=True
        ):
            if penalty > 0:
                logger.info(f"  {action:8s}: -{penalty:10.2f}")

    logger.info("")
    score_label = (
        "DISCOUNTED IG (IG × discount, null policy)"
        if discount_factors and any(d < 1.0 for d in discount_factors.values())
        else "FINAL SCORES (Pure IG, fused belief)"
    )
    logger.info(f"{score_label}:")
    sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    for action, score in sorted_final:
        marker = " <-- SELECTED" if action == selected_action else ""
        if score > float("-inf"):
            logger.info(f"  {action:8s}: {score:10.2f}{marker}")
        else:
            logger.info(f"  {action:8s}:        N/A{marker}")

    if teammate_info:
        logger.info("")
        logger.info("TEAMMATE STATES (null policy assumption):")
        for key, value in teammate_info.items():
            logger.info(f"  {key}: {value}")

    logger.info(f"{'='*60}")


# =============================================================================
# Factory Function
# =============================================================================


def create_greedy_ig_planner(
    agent_id: int,
    camera,
    grid_info,
    conf_dict: Optional[Dict] = None,
    config: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> GreedyIGPlanner:
    """
    Factory function to create a greedy IG planner.

    Args:
        agent_id: Agent ID
        camera: UAV camera
        grid_info: Grid information
        conf_dict: Sensor model parameters
        config: Configuration dict
        seed: Random seed for reproducibility

    Returns:
        GreedyIGPlanner instance
    """
    config = config or {}

    return GreedyIGPlanner(
        agent_id=agent_id,
        camera=camera,
        grid_info=grid_info,
        conf_dict=conf_dict,
        intent_discount=config.get("intent_discount", 0.5),
        overlap_penalty_weight=config.get("overlap_penalty_weight", 0.3),
        enable_discounting=config.get("enable_discounting", False),
        seed=seed,
    )

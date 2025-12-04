"""
Greedy Information Gain (IG) Planner for Multi-UAV Active Sensing

This module implements the greedy one-step lookahead IG planner as described in:
"Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

The greedy IG approach serves as a baseline/benchmark:
1. Single-step lookahead: Only considers immediate information gain
2. No trajectory planning: Myopic decision making
3. Multi-agent support: Position sharing for overlap avoidance
4. Async compatible: Works with threaded execution

Key characteristics:
- Computes IG for each primitive action
- Selects action with maximum IG (with optional teammate overlap penalty)
- Simple and fast, but doesn't consider future rewards

Use as benchmark to compare against:
- Multi-horizon planning (MH-MCTS)
- Hierarchical Dec-MCTS
- Other planning strategies

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


# =============================================================================
# Intent Data Structures (compatible with async runner)
# =============================================================================

@dataclass
class GreedyIGIntent:
    """
    Intent for greedy IG planner.
    
    Simpler than LLIntent since greedy only plans one step,
    but includes future footprint for coordination.
    """
    agent_id: int
    action: str = "hover"
    position: Tuple[float, float] = (0.0, 0.0)
    altitude: float = 0.0
    # Future position after action
    next_position: Tuple[float, float] = (0.0, 0.0)
    next_altitude: float = 0.0
    # Footprint of next position
    footprint: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (imin, imax, jmin, jmax)
    # Expected IG for this action
    expected_ig: float = 0.0
    # Timestamp for staleness tracking
    timestamp: float = field(default_factory=time.time)
    
    def is_stale(self, max_age: float = 2.0) -> bool:
        """Check if intent is stale."""
        return time.time() - self.timestamp > max_age
    
    def staleness_discount(
        self, decay_factor: float = 0.9, threshold_sec: float = 2.0
    ) -> float:
        """
        Compute D-UCT discount factor based on intent age.
        
        Returns:
        - 1.0 = fresh intent
        - <1.0 = stale intent (reduced influence)
        """
        age = time.time() - self.timestamp
        staleness = max(0, age / threshold_sec)
        return decay_factor ** staleness


# =============================================================================
# Greedy IG Planner
# =============================================================================

class GreedyIGPlanner:
    """
    Greedy one-step lookahead Information Gain planner.
    
    This is the simplest IG-based planner:
    1. For each action, compute expected IG from resulting position
    2. Select action with maximum IG
    3. Optionally apply teammate overlap penalty
    
    Multi-agent support:
    - Receives teammate intents (positions/footprints)
    - Reduces IG for cells teammates will observe
    - Broadcasts own intent after planning
    
    Async support:
    - Maintains timestamp on intents
    - D-UCT staleness discount for teammate intents
    """
    
    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        conf_dict: Optional[Dict] = None,
        intent_discount: float = 0.5,
        overlap_penalty_weight: float = 0.3,
    ):
        """
        Initialize greedy IG planner.
        
        Args:
            agent_id: This agent's ID
            camera: UAV camera model
            grid_info: Grid information
            conf_dict: Sensor model parameters by altitude
            intent_discount: Discount for teammate intent cells
            overlap_penalty_weight: Weight for overlap penalty
        """
        self.agent_id = agent_id
        self.camera = camera
        self.grid_info = grid_info
        self.conf_dict = conf_dict
        self.intent_discount = intent_discount
        self.overlap_penalty_weight = overlap_penalty_weight
        
        # Available actions
        self.actions = ["front", "back", "left", "right", "up", "down", "hover"]
        
        # Current belief
        self.belief: Optional[np.ndarray] = None
        
        # Teammate intents
        self._teammate_intents: Dict[int, GreedyIGIntent] = {}
        
        # Current intent (for broadcasting)
        self.current_intent: Optional[GreedyIGIntent] = None
        
        # Statistics
        self._stats = {
            "plans_generated": 0,
            "total_ig": 0.0,
            "intent_updates_received": 0,
        }
        
        # Per-action scores for logging
        self._action_scores: Dict[str, float] = {}
        self._raw_ig_scores: Dict[str, float] = {}
        self._overlap_penalties: Dict[str, float] = {}
    
    def update_belief(self, belief: np.ndarray) -> None:
        """Update local belief map."""
        self.belief = belief.copy()
    
    def update_teammate_intents(
        self, intents: Dict[int, GreedyIGIntent]
    ) -> None:
        """
        Update stored teammate intents.
        
        Args:
            intents: Dict mapping teammate_id -> GreedyIGIntent
        """
        self._teammate_intents = intents
        self._stats["intent_updates_received"] += 1
    
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
    
    def _compute_teammate_overlap(
        self,
        footprint: Tuple[int, int, int, int],
    ) -> float:
        """
        Compute overlap penalty with teammate footprints.
        
        Args:
            footprint: (imin, imax, jmin, jmax) proposed footprint
            
        Returns:
            Overlap penalty (higher = more overlap with teammates)
        """
        if not self._teammate_intents:
            return 0.0
        
        imin, imax, jmin, jmax = footprint
        H_grid, W_grid = self.belief.shape[:2] if self.belief is not None else (1, 1)
        
        # Create mask for proposed footprint
        my_cells = set()
        for i in range(max(0, imin), min(imax, H_grid)):
            for j in range(max(0, jmin), min(jmax, W_grid)):
                my_cells.add((i, j))
        
        if not my_cells:
            return 0.0
        
        # Count overlapping cells with teammates
        total_overlap = 0.0
        for teammate_id, intent in self._teammate_intents.items():
            if intent.is_stale():
                continue
            
            # D-UCT staleness discount
            staleness_discount = intent.staleness_discount()
            
            t_imin, t_imax, t_jmin, t_jmax = intent.footprint
            teammate_cells = set()
            for i in range(max(0, t_imin), min(t_imax, H_grid)):
                for j in range(max(0, t_jmin), min(t_jmax, W_grid)):
                    teammate_cells.add((i, j))
            
            # Count overlap
            overlap = len(my_cells & teammate_cells)
            total_overlap += overlap * staleness_discount
        
        # Normalize by footprint size
        return total_overlap / len(my_cells) if my_cells else 0.0
    
    def plan(
        self,
        current_position: Tuple[float, float],
        current_altitude: float,
    ) -> GreedyIGIntent:
        """
        Run greedy IG planning.
        
        For each action:
        1. Compute resulting position
        2. Compute IG from that position
        3. Apply teammate overlap penalty
        4. Select action with highest adjusted IG
        
        Args:
            current_position: Current (x, y) position
            current_altitude: Current altitude
            
        Returns:
            GreedyIGIntent with selected action
        """
        if self.belief is None:
            return GreedyIGIntent(agent_id=self.agent_id)
        
        # Clear logging dicts
        self._action_scores = {}
        self._raw_ig_scores = {}
        self._overlap_penalties = {}
        
        best_action = "hover"
        best_score = float("-inf")
        best_next_pos = current_position
        best_next_alt = current_altitude
        best_footprint = (0, 0, 0, 0)
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
            
            # Compute raw IG
            ig = self._compute_ig(next_pos, next_alt)
            self._raw_ig_scores[action] = ig
            
            # Get footprint for overlap calculation
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=next_pos,
                    altitude=next_alt,
                    index_form=True,
                )
                footprint = (imin, imax, jmin, jmax)
            except Exception:
                footprint = (0, 0, 0, 0)
            
            # Compute overlap penalty
            overlap = self._compute_teammate_overlap(footprint)
            overlap_penalty = overlap * self.overlap_penalty_weight * ig
            self._overlap_penalties[action] = overlap_penalty
            
            # Adjusted score
            score = ig - overlap_penalty
            self._action_scores[action] = score
            
            if score > best_score:
                best_score = score
                best_action = action
                best_next_pos = next_pos
                best_next_alt = next_alt
                best_footprint = footprint
                best_ig = ig
        
        # Create intent
        self.current_intent = GreedyIGIntent(
            agent_id=self.agent_id,
            action=best_action,
            position=current_position,
            altitude=current_altitude,
            next_position=best_next_pos,
            next_altitude=best_next_alt,
            footprint=best_footprint,
            expected_ig=best_ig,
        )
        
        self._stats["plans_generated"] += 1
        self._stats["total_ig"] += best_ig
        
        return self.current_intent
    
    def get_best_action(self) -> str:
        """Get the selected action."""
        if self.current_intent:
            return self.current_intent.action
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
    intents_received: Dict[str, Any],
):
    """
    Log greedy IG planning decision.
    
    Args:
        agent_id: Agent ID
        step: Current step
        raw_ig_scores: Raw IG per action
        overlap_penalties: Overlap penalty per action
        final_scores: Final scores (IG - penalty)
        selected_action: Selected action
        intents_received: Summary of teammate intents
    """
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"[Agent {agent_id}] GREEDY IG DECISION (Step {step})")
    logger.info(f"{'='*60}")
    
    logger.info("")
    logger.info("RAW INFORMATION GAIN:")
    sorted_ig = sorted(raw_ig_scores.items(), key=lambda x: x[1], reverse=True)
    for action, ig in sorted_ig:
        logger.info(f"  {action:8s}: {ig:10.2f}")
    
    if any(p > 0 for p in overlap_penalties.values()):
        logger.info("")
        logger.info("OVERLAP PENALTIES (teammate avoidance):")
        for action, penalty in sorted(
            overlap_penalties.items(), key=lambda x: x[1], reverse=True
        ):
            if penalty > 0:
                logger.info(f"  {action:8s}: -{penalty:10.2f}")
    
    logger.info("")
    logger.info("FINAL SCORES (IG - Overlap Penalty):")
    sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    for action, score in sorted_final:
        marker = " <-- SELECTED" if action == selected_action else ""
        if score > float("-inf"):
            logger.info(f"  {action:8s}: {score:10.2f}{marker}")
        else:
            logger.info(f"  {action:8s}:        N/A{marker}")
    
    if intents_received:
        logger.info("")
        logger.info("TEAMMATE INTENTS:")
        for key, value in intents_received.items():
            logger.info(f"  {key}: {value}")
    
    logger.info(f"{'='*60}")


# =============================================================================
# Multi-Agent Greedy IG Coordinator
# =============================================================================

class GreedyIGCoordinator:
    """
    Coordinator for multi-agent greedy IG planning.
    
    Manages intent sharing between agents running greedy IG planners.
    Compatible with both synchronous and asynchronous execution.
    """
    
    def __init__(self, num_agents: int):
        """
        Initialize coordinator.
        
        Args:
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        self._intents: Dict[int, GreedyIGIntent] = {}
        self._lock = None  # Set if async
        
        self._stats = {
            "intents_shared": 0,
            "intents_retrieved": 0,
        }
    
    def enable_async(self):
        """Enable thread-safe operations for async execution."""
        import threading
        self._lock = threading.Lock()
    
    def share_intent(self, intent: GreedyIGIntent) -> None:
        """
        Share an agent's intent.
        
        Args:
            intent: Agent's current intent
        """
        if self._lock:
            with self._lock:
                self._intents[intent.agent_id] = intent
        else:
            self._intents[intent.agent_id] = intent
        self._stats["intents_shared"] += 1
    
    def get_teammate_intents(self, agent_id: int) -> Dict[int, GreedyIGIntent]:
        """
        Get intents from all teammates (excluding self).
        
        Args:
            agent_id: Requesting agent's ID
            
        Returns:
            Dict mapping teammate_id -> GreedyIGIntent
        """
        self._stats["intents_retrieved"] += 1
        
        if self._lock:
            with self._lock:
                return {
                    tid: intent
                    for tid, intent in self._intents.items()
                    if tid != agent_id
                }
        else:
            return {
                tid: intent
                for tid, intent in self._intents.items()
                if tid != agent_id
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return dict(self._stats)


# =============================================================================
# Factory Function
# =============================================================================

def create_greedy_ig_planner(
    agent_id: int,
    camera,
    grid_info,
    conf_dict: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> GreedyIGPlanner:
    """
    Factory function to create a greedy IG planner.
    
    Args:
        agent_id: Agent ID
        camera: UAV camera
        grid_info: Grid information
        conf_dict: Sensor model parameters
        config: Configuration dict
        
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
    )

"""
Decentralized Monte Carlo Tree Search (Dec-MCTS) Planner for Multi-UAV Active Sensing

This module implements single-level Dec-MCTS planning as described in:
"Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

The Dec-MCTS approach:
1. MCTS-based trajectory planning with rollouts
2. Multi-agent support with intent sharing
3. D-UCT staleness discounting for async operation
4. IG-based reward function for information gathering

Key characteristics:
- Multi-step lookahead via MCTS tree search
- Decentralized coordination via intent sharing
- Penalizes overlap with teammate planned trajectories
- Single-level planning (no hierarchical LLP/HLP split)

Use as comparison against:
- Greedy IG (single-step, no planning)
- Multi-Horizon Dec-MCTS (hierarchical planning)

Reference: Section 3 of the paper for Dec-MCTS framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import time
import logging
import copy
import threading

from helper import uav_position, H, expected_posterior

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def copy_state(state: Dict) -> Dict:
    """Deep copy state dict with efficient numpy copy."""
    new_state = {
        "uav_pos": copy.deepcopy(state["uav_pos"]),
        "belief": state["belief"].copy(),
    }
    if "remaining_steps" in state:
        new_state["remaining_steps"] = int(state["remaining_steps"])
    if "covered_mask" in state:
        new_state["covered_mask"] = state["covered_mask"].copy()
    if "teammate_mask" in state:
        new_state["teammate_mask"] = state["teammate_mask"].copy()
    return new_state


# =============================================================================
# Intent Data Structures
# =============================================================================


@dataclass
class DecMCTSIntent:
    """
    Intent for Dec-MCTS planner.

    Contains planned trajectory from MCTS for coordination.
    """

    agent_id: int
    action_sequence: List[str] = field(default_factory=list)
    position_sequence: List[Tuple[float, float]] = field(default_factory=list)
    altitude_sequence: List[float] = field(default_factory=list)
    footprint_sequence: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )  # [(imin, imax, jmin, jmax), ...]
    ig_sequence: List[float] = field(default_factory=list)
    total_expected_ig: float = 0.0
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
        return decay_factor**staleness

    def get_future_footprints(
        self, max_steps: int = None
    ) -> List[Tuple[int, int, int, int]]:
        """Get future footprints up to max_steps."""
        if max_steps is None:
            return self.footprint_sequence
        return self.footprint_sequence[:max_steps]


# =============================================================================
# Dec-MCTS Node
# =============================================================================


class DecMCTSNode:
    """
    MCTS tree node for Dec-MCTS planning.

    Extended from MCTSNode to support:
    - Teammate intent-aware rewards
    - D-UCT discounted overlap penalties
    - IG-based rollout rewards
    """

    def __init__(
        self,
        state: Dict,
        camera,
        parent=None,
        action: str = None,
        conf_dict: Dict = None,
        config: Dict = None,
    ):
        self.state = copy_state(state)
        self.parent = parent
        self.action_from_parent = action
        self.children: Dict[str, "DecMCTSNode"] = {}
        self.visit_count = 0
        self.value = 0.0
        self.camera = camera
        self.conf_dict = conf_dict
        self.config = config or {}

        # Available actions at this state
        self.untried_actions = sorted(camera.permitted_actions(self.state["uav_pos"]))

        # Thread safety
        self.lock = threading.Lock()
        self._rng = np.random.default_rng()

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if node is terminal."""
        if "remaining_steps" not in self.state:
            return False
        return self.state.get("remaining_steps", 1) <= 0

    def best_child(self, c_param: float = 1.4) -> "DecMCTSNode":
        """Select best child using UCB1."""
        best_node, best_score = None, -float("inf")

        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploitation = child.value / child.visit_count
                exploration = c_param * np.sqrt(
                    2.0 * np.log(max(1, self.visit_count)) / child.visit_count
                )
                ucb = exploitation + exploration
            if ucb > best_score:
                best_score, best_node = ucb, child

        return best_node

    def expand(self) -> "DecMCTSNode":
        """Expand node by adding child for untried action."""
        action = self.untried_actions.pop()
        new_state = self.apply_action(self.state, action, copy_belief=True)
        child = DecMCTSNode(
            new_state,
            camera=self.camera,
            parent=self,
            action=action,
            conf_dict=self.conf_dict,
            config=self.config,
        )
        self.children[action] = child
        return child

    def apply_action(self, state: Dict, action: str, copy_belief: bool = True) -> Dict:
        """Apply action to state and return new state."""
        if copy_belief:
            next_state = copy_state(state)
        else:
            next_state = state

        next_state["uav_pos"] = uav_position(
            self.camera.x_future(action, x=state["uav_pos"])
        )

        if "remaining_steps" in next_state:
            next_state["remaining_steps"] = max(0, next_state["remaining_steps"] - 1)

        return next_state

    def _get_sensor_params(self, altitude: float) -> Tuple[float, float]:
        """Get sensor model parameters."""
        if self.conf_dict is not None and self.conf_dict != {}:
            return self.conf_dict[np.round(altitude, decimals=2)]
        else:
            a = 1
            b = 0.015
            sigma = a * (1 - np.exp(-b * altitude))
            return sigma, sigma

    def compute_ig_reward(
        self,
        state: Dict,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> float:
        """
        Compute IG reward for observing footprint.

        IG = H(prior) - E[H(posterior)]
        """
        prior = state["belief"][imin:imax, jmin:jmax, 1]
        s0, s1 = self._get_sensor_params(state["uav_pos"].altitude)

        # Expected posterior calculation
        Pz0, Pz1, p10, p11 = expected_posterior(prior, s0, s1)

        # Compute IG
        curr_entropy = H(prior)
        expected_entropy = Pz0 * H(p10) + Pz1 * H(p11)
        ig = np.sum(curr_entropy - expected_entropy)

        return float(ig)

    def compute_overlap_penalty(
        self,
        state: Dict,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> float:
        """
        Compute penalty for overlapping with teammate planned footprints.

        Uses teammate_mask if available (set by planner from intents).
        """
        if "teammate_mask" not in state:
            return 0.0

        mask = state["teammate_mask"]
        overlap = np.sum(mask[imin:imax, jmin:jmax])
        footprint_size = (imax - imin) * (jmax - jmin)

        if footprint_size == 0:
            return 0.0

        # Normalize penalty
        overlap_ratio = overlap / footprint_size
        penalty_weight = self.config.get("overlap_penalty_weight", 0.3)

        return penalty_weight * overlap_ratio

    def belief_update(
        self,
        belief: np.ndarray,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
        Pz0: np.ndarray,
        Pz1: np.ndarray,
        p10: np.ndarray,
        p11: np.ndarray,
    ) -> np.ndarray:
        """Update belief using expected posterior."""
        expected_post = Pz1 * p11 + Pz0 * p10
        belief[imin:imax, jmin:jmax, 1] = expected_post
        belief[imin:imax, jmin:jmax, 0] = 1 - expected_post
        return belief

    def rollout(
        self,
        rng,
        max_depth: int = 10,
        discount_factor: float = 1.0,
    ) -> Tuple[float, List[Dict]]:
        """
        Simulate random actions and return cumulative discounted IG reward.

        Returns:
            Tuple of (total_reward, trajectory_info)
        """
        state = copy_state(self.state)
        total_reward = 0.0
        discount = 1.0
        trajectory = []

        for t in range(max_depth):
            # Get permitted actions
            if state.get("remaining_steps", 1) <= 0:
                break

            actions = sorted(self.camera.permitted_actions(state["uav_pos"]))
            if not actions:
                break

            # Random action selection
            action = actions[rng.integers(len(actions))]

            # Apply action
            state = self.apply_action(state, action, copy_belief=False)

            # Get footprint
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=state["uav_pos"].position,
                    altitude=state["uav_pos"].altitude,
                    index_form=True,
                )
            except Exception:
                continue

            # Compute IG reward
            prior = state["belief"][imin:imax, jmin:jmax, 1]
            s0, s1 = self._get_sensor_params(state["uav_pos"].altitude)
            Pz0, Pz1, p10, p11 = expected_posterior(prior, s0, s1)

            ig_reward = self.compute_ig_reward(state, imin, imax, jmin, jmax)
            overlap_penalty = self.compute_overlap_penalty(
                state, imin, imax, jmin, jmax
            )

            reward = ig_reward - overlap_penalty
            total_reward += discount * reward
            discount *= discount_factor

            # Update belief
            state["belief"] = self.belief_update(
                state["belief"], imin, imax, jmin, jmax, Pz0, Pz1, p10, p11
            )

            trajectory.append(
                {
                    "action": action,
                    "position": state["uav_pos"].position,
                    "altitude": state["uav_pos"].altitude,
                    "ig": ig_reward,
                }
            )

        return total_reward, trajectory

    def backpropagate(self, reward: float) -> None:
        """Backpropagate reward up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

    @staticmethod
    def apply_virtual_loss(path: List["DecMCTSNode"], vloss: float = 1.0) -> None:
        """Apply virtual loss for parallel MCTS."""
        for n in path:
            n.visit_count += 1
            n.value -= vloss

    @staticmethod
    def backprop_with_reward(
        path: List["DecMCTSNode"],
        reward: float,
        vloss: float = 1.0,
    ) -> None:
        """Backpropagate with virtual loss correction."""
        for n in path:
            n.value += vloss + reward


# =============================================================================
# Dec-MCTS Planner
# =============================================================================


class DecMCTSPlanner:
    """
    Decentralized MCTS Planner for multi-agent active sensing.

    Features:
    - Single-level MCTS planning (no hierarchical split)
    - Intent sharing for decentralized coordination
    - D-UCT discounting for async operation
    - IG-based rollout rewards

    Configuration:
    - horizon: MCTS planning depth
    - iterations: Number of MCTS iterations
    - ucb_c: UCB1 exploration constant
    - discount_factor: Gamma for future rewards
    - overlap_penalty_weight: Penalty for teammate overlap
    - d_uct_decay: D-UCT staleness decay factor
    - d_uct_threshold: D-UCT staleness threshold (seconds)
    """

    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        conf_dict: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Dec-MCTS planner.

        Args:
            agent_id: This agent's ID
            camera: UAV camera model
            grid_info: Grid information
            conf_dict: Sensor model parameters
            config: Planning configuration
        """
        self.agent_id = agent_id
        self.camera = camera
        self.grid_info = grid_info
        self.conf_dict = conf_dict

        # Configuration with defaults
        config = config or {}
        self.config = {
            "horizon": config.get("horizon", 10),
            "iterations": config.get("iterations", 100),
            "ucb_c": config.get("ucb_c", 1.4),
            "discount_factor": config.get("discount_factor", 0.95),
            "overlap_penalty_weight": config.get("overlap_penalty_weight", 0.3),
            "d_uct_decay": config.get("d_uct_decay", 0.9),
            "d_uct_threshold": config.get("d_uct_threshold", 2.0),
            "parallel": config.get("parallel", 1),
            "timeout": config.get("timeout", 5.0),
        }

        # Current state
        self.belief: Optional[np.ndarray] = None
        self.position: Optional[Tuple[float, float]] = None
        self.altitude: Optional[float] = None

        # Teammate intents
        self._teammate_intents: Dict[int, DecMCTSIntent] = {}

        # Current intent
        self.current_intent: Optional[DecMCTSIntent] = None

        # Statistics
        self._stats = {
            "plans_generated": 0,
            "total_iterations": 0,
            "intent_updates_received": 0,
            "avg_planning_time": 0.0,
        }

        # MCTS action tracking for logging
        self._mcts_action_values: Dict[str, float] = {}
        self._mcts_action_visits: Dict[str, int] = {}

        # RNG
        self._rng = np.random.default_rng()

    def update_state(
        self,
        position: Tuple[float, float],
        altitude: float,
        belief: np.ndarray,
    ) -> None:
        """Update planner state."""
        self.position = position
        self.altitude = altitude
        self.belief = belief.copy()

    def update_teammate_intents(
        self,
        intents: Dict[int, DecMCTSIntent],
    ) -> None:
        """Update stored teammate intents."""
        self._teammate_intents = intents
        self._stats["intent_updates_received"] += 1

    def _compute_teammate_mask(self) -> np.ndarray:
        """
        Compute coverage mask from teammate intents with D-UCT discounting.

        Returns:
            2D mask where values indicate teammate coverage probability
        """
        H_grid, W_grid = self.belief.shape[:2]
        mask = np.zeros((H_grid, W_grid), dtype=float)

        for teammate_id, intent in self._teammate_intents.items():
            if intent.is_stale(max_age=self.config["d_uct_threshold"] * 2):
                continue

            # D-UCT staleness discount
            discount = intent.staleness_discount(
                decay_factor=self.config["d_uct_decay"],
                threshold_sec=self.config["d_uct_threshold"],
            )

            # Add discounted footprints to mask
            for footprint in intent.footprint_sequence:
                imin, imax, jmin, jmax = footprint
                imin = max(0, min(imin, H_grid))
                imax = max(0, min(imax, H_grid))
                jmin = max(0, min(jmin, W_grid))
                jmax = max(0, min(jmax, W_grid))

                mask[imin:imax, jmin:jmax] += discount

        return mask

    def plan(self) -> DecMCTSIntent:
        """
        Run Dec-MCTS planning.

        Returns:
            DecMCTSIntent with planned trajectory
        """
        start_time = time.perf_counter()  # High-resolution timer

        # Build initial state
        uav_pos = uav_position((self.position, self.altitude))
        state = {
            "uav_pos": uav_pos,
            "belief": self.belief.copy(),
            "remaining_steps": self.config["horizon"],
            "teammate_mask": self._compute_teammate_mask(),
        }

        # Create MCTS root
        root = DecMCTSNode(
            state,
            camera=self.camera,
            conf_dict=self.conf_dict,
            config=self.config,
        )

        # Run MCTS iterations
        iterations = 0
        timeout = self.config["timeout"]
        max_iterations = self.config["iterations"]

        while iterations < max_iterations:
            if time.perf_counter() - start_time >= timeout:
                break

            # Selection
            node, path = self._tree_policy(root)

            # Simulation (rollout)
            reward, trajectory = node.rollout(
                rng=self._rng,
                max_depth=self.config["horizon"],
                discount_factor=self.config["discount_factor"],
            )

            # Backpropagation
            node.backpropagate(reward)

            iterations += 1

        end_time = time.perf_counter()
        
        # Extract best action and trajectory
        best_action, action_values = self._extract_best_action(root)
        trajectory = self._extract_trajectory(root)

        # Store action values for logging
        self._mcts_action_values = action_values
        self._mcts_action_visits = {
            action: child.visit_count for action, child in root.children.items()
        }

        # Build intent
        intent = self._build_intent(best_action, trajectory)
        self.current_intent = intent

        # Update stats with timestamps
        planning_time = end_time - start_time
        self._stats["plans_generated"] += 1
        self._stats["total_iterations"] += iterations
        self._stats["last_planning_time_ms"] = planning_time * 1000.0
        self._stats["last_start_ms"] = start_time * 1000.0
        self._stats["last_end_ms"] = end_time * 1000.0
        n = self._stats["plans_generated"]
        self._stats["avg_planning_time"] = (
            self._stats["avg_planning_time"] * (n - 1) + planning_time
        ) / n

        return intent

    def _tree_policy(self, root: DecMCTSNode) -> Tuple[DecMCTSNode, List[DecMCTSNode]]:
        """Tree policy: selection and expansion."""
        node = root
        path = [node]

        while not node.is_terminal():
            if not node.is_fully_expanded():
                child = node.expand()
                path.append(child)
                return child, path
            else:
                node = node.best_child(c_param=self.config["ucb_c"])
                path.append(node)

        return node, path

    def _extract_best_action(self, root: DecMCTSNode) -> Tuple[str, Dict[str, float]]:
        """Extract best action from MCTS tree."""
        if not root.children:
            return "hover", {}

        # Compute average values
        action_values = {}
        for action, child in root.children.items():
            if child.visit_count > 0:
                action_values[action] = child.value / child.visit_count
            else:
                action_values[action] = 0.0

        # Select action with highest average value
        best_action = max(action_values, key=action_values.get)

        return best_action, action_values

    def _extract_trajectory(
        self,
        root: DecMCTSNode,
        max_depth: int = None,
    ) -> List[Dict]:
        """Extract best trajectory from MCTS tree."""
        if max_depth is None:
            max_depth = self.config["horizon"]

        trajectory = []
        node = root
        depth = 0

        while node.children and depth < max_depth:
            # Select best child greedily (c=0)
            best_child = node.best_child(c_param=0.0)
            if best_child is None:
                break

            action = best_child.action_from_parent
            state = best_child.state

            # Get footprint
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=state["uav_pos"].position,
                    altitude=state["uav_pos"].altitude,
                    index_form=True,
                )
            except Exception:
                imin, imax, jmin, jmax = 0, 0, 0, 0

            trajectory.append(
                {
                    "action": action,
                    "position": state["uav_pos"].position,
                    "altitude": state["uav_pos"].altitude,
                    "footprint": (imin, imax, jmin, jmax),
                    "value": best_child.value / max(1, best_child.visit_count),
                }
            )

            node = best_child
            depth += 1

        return trajectory

    def _build_intent(
        self,
        action: str,
        trajectory: List[Dict],
    ) -> DecMCTSIntent:
        """Build intent from action and trajectory."""
        action_sequence = [action]
        position_sequence = [self.position]
        altitude_sequence = [self.altitude]
        footprint_sequence = []
        ig_sequence = []

        # Add trajectory info
        for step in trajectory:
            if step["action"] != action:  # Skip first action (already added)
                action_sequence.append(step["action"])
            position_sequence.append(step["position"])
            altitude_sequence.append(step["altitude"])
            footprint_sequence.append(step["footprint"])
            ig_sequence.append(step.get("value", 0.0))

        total_ig = sum(ig_sequence) if ig_sequence else 0.0

        return DecMCTSIntent(
            agent_id=self.agent_id,
            action_sequence=action_sequence,
            position_sequence=position_sequence,
            altitude_sequence=altitude_sequence,
            footprint_sequence=footprint_sequence,
            ig_sequence=ig_sequence,
            total_expected_ig=total_ig,
            timestamp=time.time(),
        )

    def get_action_values(self) -> Dict[str, float]:
        """Get MCTS action values (for logging)."""
        return self._mcts_action_values.copy()

    def get_action_visits(self) -> Dict[str, int]:
        """Get MCTS action visit counts (for logging)."""
        return self._mcts_action_visits.copy()

    def get_statistics(self) -> Dict:
        """Get planner statistics."""
        return self._stats.copy()


# =============================================================================
# Multi-Agent Coordinator
# =============================================================================


class DecMCTSCoordinator:
    """
    Coordinator for multi-agent Dec-MCTS.

    Manages intent sharing between agents.
    """

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self._intents: Dict[int, DecMCTSIntent] = {}
        self._lock = threading.Lock()

    def share_intent(self, intent: DecMCTSIntent) -> None:
        """Share intent from an agent."""
        with self._lock:
            self._intents[intent.agent_id] = intent

    def get_teammate_intents(self, agent_id: int) -> Dict[int, DecMCTSIntent]:
        """Get intents from teammates (excluding own)."""
        with self._lock:
            return {
                aid: intent for aid, intent in self._intents.items() if aid != agent_id
            }

    def get_all_intents(self) -> Dict[int, DecMCTSIntent]:
        """Get all intents."""
        with self._lock:
            return self._intents.copy()


# =============================================================================
# Logging Functions
# =============================================================================


def log_dec_mcts_decision(
    agent_id: int,
    step: int,
    mcts_action_values: Dict[str, float],
    mcts_action_visits: Dict[str, int],
    selected_action: str,
    trajectory_summary: Dict,
    intents_received: Dict,
) -> None:
    """
    Log Dec-MCTS planning decision.

    Args:
        agent_id: Agent ID
        step: Planning step number
        mcts_action_values: MCTS action values (avg reward per action)
        mcts_action_visits: MCTS visit counts per action
        selected_action: Selected action
        trajectory_summary: Summary of planned trajectory
        intents_received: Info about teammate intents
    """
    logger.info("=" * 70)
    logger.info(f"[DEC-MCTS] Agent {agent_id} | Step {step}")
    logger.info("=" * 70)

    # Log MCTS action values (rollout-based scores)
    logger.info("\n[MCTS ROLLOUT VALUES] (multi-step lookahead):")
    sorted_actions = sorted(
        mcts_action_values.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for action, value in sorted_actions:
        visits = mcts_action_visits.get(action, 0)
        marker = " <-- SELECTED" if action == selected_action else ""
        logger.info(f"  {action:8s}: {value:+8.4f} (visits: {visits:4d}){marker}")

    # Log trajectory
    logger.info(f"\n[TRAJECTORY] {trajectory_summary.get('length', 0)} steps")
    if "total_ig" in trajectory_summary:
        logger.info(f"  Total expected IG: {trajectory_summary['total_ig']:.4f}")

    # Log teammate intents
    if intents_received:
        logger.info(f"\n[TEAMMATE INTENTS]")
        logger.info(f"  Received from: {intents_received.get('teammates', [])}")

    logger.info("=" * 70 + "\n")


def setup_dec_mcts_logger(
    log_dir: str = None,
    experiment_name: str = None,
) -> str:
    """
    Setup logging for Dec-MCTS planner.

    Returns:
        Path to log file
    """
    import os

    # If no explicit log_dir given but an experiment name is provided,
    # place logs under the trials folder for that experiment to avoid
    # creating a top-level `logs/` directory on every run.
    if log_dir is None:
        if experiment_name:
            log_dir = os.path.join("trials", experiment_name, "logs")
        else:
            # Avoid creating top-level `logs/` folders; place legacy logs
            # under `trials/logs/dec_mcts` instead.
            log_dir = os.path.join("trials", "logs", "dec_mcts")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"dec_mcts_{timestamp}.log")

    # Clear existing handlers to prevent duplicate logging
    logger.handlers.clear()

    # Configure logger
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return log_file


# =============================================================================
# Factory Function
# =============================================================================


def create_dec_mcts_planner(
    agent_id: int,
    num_agents: int,
    camera,
    grid_info,
    coordinator: DecMCTSCoordinator = None,
    config: Dict = None,
) -> Tuple[DecMCTSPlanner, DecMCTSCoordinator]:
    """
    Factory function to create Dec-MCTS planner.

    Args:
        agent_id: Agent ID
        num_agents: Total number of agents
        camera: UAV camera model
        grid_info: Grid information
        coordinator: Optional existing coordinator
        config: Planning configuration

    Returns:
        Tuple of (planner, coordinator)
    """
    if coordinator is None:
        coordinator = DecMCTSCoordinator(num_agents)

    planner = DecMCTSPlanner(
        agent_id=agent_id,
        camera=camera,
        grid_info=grid_info,
        config=config,
    )

    return planner, coordinator

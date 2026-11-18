"""
Decentralized Monte Carlo Tree Search (Dec-MCTS) Planner for Multi-UAV Active Sensing

This module implements single-level Dec-MCTS planning as described in:
"Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

The Dec-MCTS approach:
1. MCTS-based trajectory planning with rollouts
2. Multi-agent support with intent sharing
3. Teammate discounting of redundant information gain
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import threading

from helper import (
    uav_position,
    expected_posterior,
    information_gain,
    select_canonical_action,
    teammate_in_range,
    teammate_overlap_discount,
)
from mcts import SearchNodeOps, copy_state



# =============================================================================
# Helper Functions
# =============================================================================


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
    step: int = 0  # planning cycle that produced this intent

# =============================================================================
# Dec-MCTS Node
# =============================================================================


class DecMCTSNode(SearchNodeOps):
    """
    MCTS tree node for Dec-MCTS planning.

    Extended from MCTSNode to support:
    - Teammate intent-aware rewards
    - IGd-style discounting of teammate overlap
    - IG-based rollout rewards
    """

    def __init__(
        self,
        state: Dict,
        camera,
        parent=None,
        action: Optional[str] = None,
        conf_dict: Optional[Dict] = None,
        config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        self.state = copy_state(state)
        self.parent = parent
        self.action_from_parent = action
        self.children: Dict[str, "DecMCTSNode"] = {}
        self.visit_count = 0
        self.value = 0.0
        self.immediate_reward = 0.0
        self.camera = camera
        self.conf_dict = conf_dict
        self.config = config or {}

        # Available actions at this state
        self.untried_actions = sorted(camera.permitted_actions(self.state["uav_pos"]))

        # Use provided seed or inherit from parent for reproducibility
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif parent is not None and hasattr(parent, "_rng"):
            self._rng = np.random.default_rng(parent._rng.integers(2**31))
        else:
            self._rng = np.random.default_rng()

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if node is terminal."""
        if "remaining_steps" not in self.state:
            return False
        return self.state.get("remaining_steps", 1) <= 0

    def best_child(
        self,
        c_param: float = 1.4,
        value_bounds: Optional[Tuple[float, float]] = None,
    ) -> Optional["DecMCTSNode"]:
        """Select best child using UCB1, with mean values scaled to [0, 1] by value_bounds."""
        low, high = value_bounds if value_bounds is not None else (0.0, 1.0)
        span = high - low if high > low else 1.0
        best_node, best_score = None, -float("inf")

        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploitation = (child.value / child.visit_count - low) / span
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
        observed = self.observe(new_state)
        child = DecMCTSNode(
            new_state,
            camera=self.camera,
            parent=self,
            action=action,
            conf_dict=self.conf_dict,
            config=self.config,
            seed=None,  # Will inherit from parent
        )
        child.immediate_reward = observed[0] if observed is not None else 0.0
        self.children[action] = child
        return child

    def observe(self, state: Dict, rng=None) -> Optional[Tuple[float, float]]:
        """Score the footprint at state's position and apply a sampled observation to its belief."""
        try:
            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=state["uav_pos"].position,
                altitude=state["uav_pos"].altitude,
                index_form=True,
            )
        except Exception:
            return None

        ig_reward = self.compute_ig_reward(state, imin, imax, jmin, jmax)
        overlap_penalty = self.compute_overlap_penalty(
            state, imin, imax, jmin, jmax, ig_reward
        )

        prior = state["belief"][imin:imax, jmin:jmax, 1]
        s0, s1 = self._get_sensor_params(state["uav_pos"].altitude)
        Pz0, Pz1, p10, p11 = expected_posterior(prior, s0, s1)
        state["belief"] = self.belief_update(
            state["belief"], imin, imax, jmin, jmax, Pz0, Pz1, p10, p11, rng
        )

        return ig_reward - overlap_penalty, ig_reward

    def _get_sensor_params(self, altitude: float) -> Tuple[float, float]:
        """Return (s0, s1) sensor noise for altitude. See helper.get_sensor_params."""
        from helper import get_sensor_params

        return get_sensor_params(altitude, self.conf_dict)

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

        return float(np.sum(information_gain(prior, s0, s1)))

    def compute_overlap_penalty(
        self,
        state: Dict,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
        ig_reward: Optional[float] = None,
    ) -> float:
        """
        Compute the part of this footprint's IG that teammates' plans make redundant.

        Uses state["teammate_mask"], the teammate plans from _compute_teammate_mask. At weight 1
        the reward IG - penalty equals greedy IGd.
        """
        plans = state.get("teammate_mask")
        if not plans:
            return 0.0

        if ig_reward is None:
            ig_reward = self.compute_ig_reward(state, imin, imax, jmin, jmax)
        redundant = 1.0 - teammate_overlap_discount((imin, imax, jmin, jmax), plans)
        return self.config.get("overlap_penalty_weight", 1.0) * ig_reward * redundant

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

            observed = self.observe(state, rng)
            if observed is None:
                continue
            reward, ig_reward = observed
            total_reward += discount * reward
            discount *= discount_factor

            trajectory.append(
                {
                    "action": action,
                    "position": state["uav_pos"].position,
                    "altitude": state["uav_pos"].altitude,
                    "ig": ig_reward,
                }
            )

        return total_reward, trajectory

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
    - IG-based rollout rewards

    Configuration:
    - horizon: MCTS planning depth
    - iterations: Number of MCTS iterations
    - ucb_c: UCB1 exploration constant
    - discount_factor: Gamma for future rewards
    - overlap_penalty_weight: Share of redundant IG removed for teammate overlap (1 = IGd)
    - discount_teammates: If False, teammate plans do not affect the reward (plain IG)
    - comm_radius: (dx, dy, dz) limit for hearing a teammate; None means unlimited
    """

    def __init__(
        self,
        agent_id: int,
        camera,
        grid_info,
        conf_dict: Optional[Dict] = None,
        config: Optional[Dict] = None,
        seed: Optional[int] = None,
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
            "overlap_penalty_weight": config.get("overlap_penalty_weight", 1.0),
            "discount_teammates": config.get("discount_teammates", True),
            "comm_radius": config.get("comm_radius"),
            "parallel": config.get("parallel", 1),
            "timeout": config.get("timeout", 5.0),  # accepted but unused: iterations stop the search
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

        # RNG with seed
        self._rng = np.random.default_rng(seed)
        self._seed = seed

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

    def _require_state(self) -> Tuple[Tuple[float, float], float, np.ndarray]:
        """Return (position, altitude, belief), which update_state must have set."""
        if self.position is None or self.altitude is None or self.belief is None:
            raise RuntimeError("update_state must be called before planning")
        return self.position, self.altitude, self.belief

    def update_teammate_intents(
        self,
        intents: Dict[int, DecMCTSIntent],
    ) -> None:
        """Update stored teammate intents."""
        self._teammate_intents = intents
        self._stats["intent_updates_received"] += 1

    def _compute_teammate_mask(self) -> List[List[Tuple[int, int, int, int]]]:
        """
        Collect the teammate plans that reduce this agent's reward.

        Returns:
            One footprint list per teammate in communication range, nearest step first.
        """
        if not self.config["discount_teammates"]:
            return []

        position, altitude, _ = self._require_state()
        own_state = (position[0], position[1], altitude)
        plans = []
        for intent in self._teammate_intents.values():
            # position_sequence[0] is the start; index 1 belongs to the first footprint.
            teammate_state = None
            if len(intent.position_sequence) > 1 and len(intent.altitude_sequence) > 1:
                x, y = intent.position_sequence[1]
                teammate_state = (x, y, intent.altitude_sequence[1])
            if not teammate_in_range(own_state, teammate_state, self.config["comm_radius"]):
                continue

            plans.append(list(intent.footprint_sequence))

        return plans

    def plan(self) -> DecMCTSIntent:
        """
        Run Dec-MCTS planning.

        Returns:
            DecMCTSIntent with planned trajectory
        """
        start_time = time.perf_counter()  # High-resolution timer

        # Build initial state
        position, altitude, belief = self._require_state()
        uav_pos = uav_position((position, altitude))
        state = {
            "uav_pos": uav_pos,
            "belief": belief.copy(),
            "remaining_steps": self.config["horizon"],
            "teammate_mask": self._compute_teammate_mask(),
        }

        # Create MCTS root
        root = DecMCTSNode(
            state,
            camera=self.camera,
            conf_dict=self.conf_dict,
            config=self.config,
            seed=self._seed,
        )

        # Run MCTS iterations
        iterations = 0
        self._return_bounds = None
        gamma = self.config["discount_factor"]
        # Every root action is scored once, whatever the budget.
        max_iterations = max(self.config["iterations"], len(root.untried_actions))

        while iterations < max_iterations:
            # Selection
            node, path = self._tree_policy(root)

            # Simulation (rollout)
            reward, trajectory = node.rollout(
                rng=self._rng,
                max_depth=self.config["horizon"],
                discount_factor=self.config["discount_factor"],
            )

            # The whole episode's discounted return is credited to every node on the path.
            episode_return = sum(
                gamma**depth * step.immediate_reward
                for depth, step in enumerate(path[1:])
            ) + gamma ** (len(path) - 1) * reward
            node.backpropagate(episode_return)

            if self._return_bounds is None:
                self._return_bounds = (episode_return, episode_return)
            else:
                low, high = self._return_bounds
                self._return_bounds = (min(low, episode_return), max(high, episode_return))

            iterations += 1

        end_time = time.perf_counter()

        # Extract best action and trajectory
        best_action, action_values = self._extract_best_action(root)
        trajectory = self._extract_trajectory(root, first_action=best_action)

        # Store action values for logging
        self._mcts_action_values = action_values

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
                best_child = node.best_child(
                    c_param=self.config["ucb_c"], value_bounds=self._return_bounds
                )
                if best_child is None:
                    raise RuntimeError("Fully expanded node has no selectable child")
                node = best_child
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

        # Highest average value, with greedy's tie-break order
        best_action = select_canonical_action(
            {action: (value,) for action, value in action_values.items()}
        )

        return best_action, action_values

    def _extract_trajectory(
        self,
        root: DecMCTSNode,
        first_action: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> List[Dict]:
        """Extract best trajectory from MCTS tree, starting with first_action if given."""
        depth_limit: int = self.config["horizon"] if max_depth is None else max_depth

        trajectory = []
        node = root
        depth = 0

        while node.children and depth < depth_limit:
            if depth == 0 and first_action in node.children:
                best_child = node.children[first_action]
            else:
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
        action_sequence = [step["action"] for step in trajectory] or [action]
        position, altitude, _ = self._require_state()
        position_sequence = [position]
        altitude_sequence = [altitude]
        footprint_sequence = []
        ig_sequence = []

        for step in trajectory:
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
            step=self._stats["plans_generated"] + 1,
        )

    def get_action_values(self) -> Dict[str, float]:
        """Get MCTS action values (for logging)."""
        return self._mcts_action_values.copy()

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

# =============================================================================
# Factory Function
# =============================================================================

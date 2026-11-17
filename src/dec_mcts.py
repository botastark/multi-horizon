"""
Decentralized Monte Carlo Tree Search (Dec-MCTS) Planner for Multi-UAV Active Sensing

This module implements single-level Dec-MCTS planning as described in:
"Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

The Dec-MCTS approach:
1. MCTS-based trajectory planning with rollouts
2. Multi-agent support with intent sharing
3. D-UCT staleness discounting of teammate intents (by age in steps)
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
from g2_evaluator import (
    CoverageStepModel,
    auto_g2_weight,
    camera_step_geometry,
    uncovered_from_belief,
)

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
    # Simulation step at which the plan was made
    step: int = 0

    def age(self, current_step: int) -> int:
        return max(0, int(current_step) - int(self.step))

    def is_stale(self, current_step: int, max_age_steps: float) -> bool:
        """True when older than max_age_steps simulation steps."""
        return self.age(current_step) > max_age_steps

    def staleness_discount(
        self, current_step: int, decay_factor: float = 0.9, threshold_steps: float = 2.0
    ) -> float:
        """
        D-UCT weight decay_factor ** (age_steps / threshold_steps):
        1.0 for an intent made this step, smaller for older ones.
        """
        return decay_factor ** (self.age(current_step) / max(threshold_steps, 1e-9))

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
        seed: int = None,
    ):
        self.state = copy_state(state)
        self.parent = parent
        self.action_from_parent = action
        self.children: Dict[str, "DecMCTSNode"] = {}
        # D-UCT statistics: discounted visit count / reward sum, lazily decayed
        # from iteration t_last (gamma = 1 gives plain UCT)
        self.visit_count = 0.0
        self.value = 0.0
        self.t_last = 0
        self.immediate_reward = 0.0
        self.camera = camera
        self.conf_dict = conf_dict
        self.config = config or {}

        # Available actions at this state
        self.untried_actions = sorted(camera.permitted_actions(self.state["uav_pos"]))

        # Thread safety
        self.lock = threading.Lock()
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

    def discounted_count(self, t: Optional[int], gamma: float) -> float:
        if t is None or self.visit_count <= 0:
            return self.visit_count
        return self.visit_count * gamma ** (t - self.t_last)

    def best_child(
        self, c_param: float = 1.4, t: Optional[int] = None, gamma: float = 1.0
    ) -> "DecMCTSNode":
        """Select best child using (D-)UCT; t/gamma enable D-UCT discounting."""
        best_node, best_score = None, -float("inf")
        parent_count = self.discounted_count(t, gamma)

        for action, child in self.children.items():
            child_count = child.discounted_count(t, gamma)
            if child_count <= 1e-12:
                ucb = float("inf")
            else:
                exploitation = child.value / child.visit_count  # mean; decay cancels
                exploration = c_param * np.sqrt(
                    2.0 * np.log(max(1.0, parent_count)) / child_count
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

    def observe(self, state: Dict) -> Optional[Tuple[float, float]]:
        """Score the footprint at state's position and apply its observation to state's belief."""
        try:
            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=state["uav_pos"].position,
                altitude=state["uav_pos"].altitude,
                index_form=True,
            )
        except Exception:
            return None

        ig_reward = self.compute_ig_reward(state, imin, imax, jmin, jmax)
        overlap_penalty = self.compute_overlap_penalty(state, imin, imax, jmin, jmax)

        prior = state["belief"][imin:imax, jmin:jmax, 1]
        s0, s1 = self._get_sensor_params(state["uav_pos"].altitude)
        Pz0, Pz1, p10, p11 = expected_posterior(prior, s0, s1)
        state["belief"] = self.belief_update(
            state["belief"], imin, imax, jmin, jmax, Pz0, Pz1, p10, p11
        )

        return ig_reward - overlap_penalty, ig_reward

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

            observed = self.observe(state)
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

    def backpropagate(self, reward: float) -> None:
        """Add the same reward to this node and all ancestors (legacy helper)."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

    def backpropagate_return(
        self,
        rollout_reward: float,
        gamma: float,
        terminal_bonus: float = 0.0,
        t: Optional[int] = None,
        d_uct_gamma: float = 1.0,
    ) -> None:
        """
        Backpropagate discounted returns.

        Each node on the path is credited with the return from ITS OWN step
        onward: G(node) = immediate_reward(node) + gamma * G(child on path),
        starting from G = rollout reward below the leaf. So a root child's value
        always includes the IG of its own first step plus the discounted
        rewards of the deeper tree steps and the rollout. (The root has no
        action/reward of its own and receives the return of its child.)

        terminal_bonus (the g2 term of the whole plan) is added to every node's
        credited return. With t/d_uct_gamma, older statistics are discounted
        first (D-UCT, paper Eq. 4-5).
        """
        ret = rollout_reward
        node = self
        while node is not None:
            if node.parent is not None:
                ret = node.immediate_reward + gamma * ret
            if t is not None and node.visit_count > 0:
                decay = d_uct_gamma ** (t - node.t_last)
                node.visit_count *= decay
                node.value *= decay
            node.visit_count += 1
            node.value += ret + terminal_bonus
            if t is not None:
                node.t_last = t
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
    - d_uct_threshold_steps: D-UCT staleness threshold (simulation steps);
      teammate intents older than 2x this are ignored
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
            "overlap_penalty_weight": config.get("overlap_penalty_weight", 0.3),
            "d_uct_decay": config.get("d_uct_decay", 0.9),
            "d_uct_threshold_steps": config.get(
                "d_uct_threshold_steps", config.get("d_uct_threshold", 2.0)
            ),
            "parallel": config.get("parallel", 1),
            "timeout": config.get("timeout", 5.0),
            # D-UCT tree discount per iteration (paper Eq. 4-5)
            "d_uct_gamma": config.get("d_uct_gamma", 0.99),
            # g2: same long-horizon coverage term as MH-Dec-MCTS, without an
            # HLP (no region sequences; all remaining work is "unclaimed")
            "use_g2": config.get("use_g2", False),
            "g2_weight": config.get("g2_weight"),
        }
        self._step_cells, self._fp_half = camera_step_geometry(camera)
        self.g2_weight = (
            float(self.config["g2_weight"])
            if self.config["g2_weight"] is not None
            else auto_g2_weight(camera, conf_dict)
        )
        self.coverage_mask: Optional[np.ndarray] = None
        self._g2_ctx: Optional[Dict[str, Any]] = None
        self.last_g2_steps: Optional[float] = None
        self._root: Optional["DecMCTSNode"] = None
        self._tree_t = 0

        # Current state
        self.belief: Optional[np.ndarray] = None
        self.position: Optional[Tuple[float, float]] = None
        self.altitude: Optional[float] = None
        self.current_step: int = 0
        self._step_given = False

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

        # RNG with seed
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def update_state(
        self,
        position: Tuple[float, float],
        altitude: float,
        belief: np.ndarray,
        step: Optional[int] = None,
        coverage_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update planner state.

        step: simulation step (None = count plan() calls).
        coverage_mask: (H, W) bool, cells known to be observed at least once
            (used by g2; None = cells whose belief left 0.5).
        """
        self.position = position
        self.altitude = altitude
        self.belief = belief.copy()
        self.coverage_mask = coverage_mask
        if step is not None:
            self.current_step = int(step)
            self._step_given = True
        elif not self._step_given:
            self.current_step = self._stats["plans_generated"]

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

        threshold = self.config["d_uct_threshold_steps"]
        for teammate_id, intent in self._teammate_intents.items():
            if intent.is_stale(self.current_step, max_age_steps=2 * threshold):
                continue

            # D-UCT staleness discount (age in simulation steps)
            discount = intent.staleness_discount(
                self.current_step,
                decay_factor=self.config["d_uct_decay"],
                threshold_steps=threshold,
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

    # ------------------------------------------------------------- g2 term
    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self.camera.convert_xy_ij(x, y, self.camera.grid.center)
        return float(i), float(j)

    def _footprint(self, uav_pos) -> Tuple[int, int, int, int]:
        [[imin, imax], [jmin, jmax]] = self.camera.get_range(
            position=uav_pos.position, altitude=uav_pos.altitude, index_form=True
        )
        return (imin, imax, jmin, jmax)

    def _prepare_g2_context(self) -> None:
        """Coverage model after teammates' fresh plans + baseline (agent stays put)."""
        uncovered = (
            ~np.asarray(self.coverage_mask, dtype=bool)
            if self.coverage_mask is not None
            else uncovered_from_belief(self.belief)
        )
        tile = (int(round(self._step_cells)),) * 2
        model = CoverageStepModel(uncovered, tile, self._step_cells, self._fp_half)
        threshold = self.config["d_uct_threshold_steps"]
        teammate_fps, teammates = [], {}
        for tid, intent in self._teammate_intents.items():
            if intent.is_stale(self.current_step, max_age_steps=2 * threshold):
                continue
            teammate_fps.extend(intent.footprint_sequence)
            end = intent.position_sequence[-1] if intent.position_sequence else None
            teammates[tid] = (self._world_to_grid(*end) if end is not None else None, [])
        model = model.with_footprints(teammate_fps)
        agents = dict(teammates)
        agents[self.agent_id] = (self._world_to_grid(*self.position), [])
        self._g2_ctx = {
            "model": model,
            "teammates": teammates,
            "baseline": model.team_steps(agents),
        }

    def _g2_steps_for_path(self, leaf: "DecMCTSNode") -> float:
        """g2 after executing the TREE part of the plan (root -> leaf), as in the paper."""
        footprints, node = [], leaf
        while node is not None and node.parent is not None:
            footprints.append(self._footprint(node.state["uav_pos"]))
            node = node.parent
        end = leaf.state["uav_pos"].position
        agents = dict(self._g2_ctx["teammates"])
        agents[self.agent_id] = (self._world_to_grid(end[0], end[1]), [])
        return self._g2_ctx["model"].with_footprints(footprints).team_steps(agents)

    # ------------------------------------------------------- anytime search
    def begin_round(self) -> None:
        """Start a planning round (one simulation step): fresh tree at the current state."""
        uav_pos = uav_position((self.position, self.altitude))
        state = {
            "uav_pos": uav_pos,
            "belief": self.belief.copy(),
            "remaining_steps": self.config["horizon"],
            "teammate_mask": self._compute_teammate_mask(),
        }
        self._root = DecMCTSNode(
            state,
            camera=self.camera,
            conf_dict=self.conf_dict,
            config=self.config,
            seed=self._seed,
        )
        self._tree_t = 0
        self._round_time = 0.0
        self._round_start = None

    def run_iterations(self, n: int, timeout: Optional[float] = None) -> DecMCTSIntent:
        """
        Spend up to n iterations on the current round with the CURRENT teammate
        intents, then publish the current best plan (anytime). New expansions
        use the refreshed teammate mask; D-UCT discounts older episodes.
        """
        t_start = time.perf_counter()
        if self._round_start is None:
            self._round_start = t_start
        root = self._root
        root.state["teammate_mask"] = self._compute_teammate_mask()
        self._g2_ctx = None
        if self.config["use_g2"]:
            self._prepare_g2_context()
        gamma_tree = self.config["d_uct_gamma"]

        for _ in range(int(n)):
            if timeout is not None and time.perf_counter() - t_start >= timeout:
                break
            self._tree_t += 1
            node, path = self._tree_policy(root)
            reward, trajectory = node.rollout(
                rng=self._rng,
                max_depth=self.config["horizon"],
                discount_factor=self.config["discount_factor"],
            )
            bonus = 0.0
            if self._g2_ctx is not None:
                bonus = self.g2_weight * (
                    self._g2_ctx["baseline"] - self._g2_steps_for_path(node)
                )
            node.backpropagate_return(
                reward,
                self.config["discount_factor"],
                terminal_bonus=bonus,
                t=self._tree_t,
                d_uct_gamma=gamma_tree,
            )

        t_end = time.perf_counter()
        self._round_time += t_end - t_start
        self._stats["total_iterations"] += int(n)

        best_action, action_values = self._extract_best_action(root)
        trajectory = self._extract_trajectory(root, first_action=best_action)
        self._mcts_action_values = action_values
        self._mcts_action_visits = {
            action: child.visit_count for action, child in root.children.items()
        }
        intent = self._build_intent(best_action, trajectory)
        self.current_intent = intent
        self.last_g2_steps = None
        if self._g2_ctx is not None and best_action in root.children:
            self.last_g2_steps = self._g2_steps_for_path(root.children[best_action])
        return intent

    def finish_round(self) -> None:
        """Book-keeping at the end of a round (statistics/timing)."""
        planning_time = self._round_time
        self._stats["plans_generated"] += 1
        self._stats["last_planning_time_ms"] = planning_time * 1000.0
        self._stats["last_start_ms"] = (self._round_start or 0.0) * 1000.0
        self._stats["last_end_ms"] = ((self._round_start or 0.0) + planning_time) * 1000.0
        n = self._stats["plans_generated"]
        self._stats["avg_planning_time"] = (
            self._stats["avg_planning_time"] * (n - 1) + planning_time
        ) / n

    def plan(self) -> DecMCTSIntent:
        """
        Synchronous planning: one round with the full iteration budget (stops
        early after `timeout` wall-clock seconds; the asynchronous scheduler
        does not use the timeout).
        """
        self.begin_round()
        intent = self.run_iterations(self.config["iterations"], timeout=self.config["timeout"])
        self.finish_round()
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
                node = node.best_child(
                    c_param=self.config["ucb_c"],
                    t=self._tree_t,
                    gamma=self.config["d_uct_gamma"],
                )
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
        first_action: str = None,
        max_depth: int = None,
    ) -> List[Dict]:
        """Extract best trajectory from MCTS tree, starting with first_action if given."""
        if max_depth is None:
            max_depth = self.config["horizon"]

        trajectory = []
        node = root
        depth = 0

        while node.children and depth < max_depth:
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
        position_sequence = [self.position]
        altitude_sequence = [self.altitude]
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
            step=self.current_step,
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
    Intent exchange for multi-agent Dec-MCTS.

    A shared intent is delivered to the agents within communication range of
    the sender at that moment (neighbor_fn) and kept in each receiver's inbox
    (last intent per teammate). Old intents are aged out by the planner.
    enabled=False (IG / IGd modes) delivers nothing.
    """

    def __init__(self, num_agents: int, neighbor_fn=None, enabled: bool = True):
        self.num_agents = num_agents
        self.neighbor_fn = neighbor_fn
        self.enabled = enabled
        # Deferred delivery (simulated asynchrony): queued until flush()
        self.deferred = False
        self._pending: List[DecMCTSIntent] = []
        self._intents: Dict[int, DecMCTSIntent] = {}
        self._inbox: Dict[int, Dict[int, DecMCTSIntent]] = {
            a: {} for a in range(num_agents)
        }
        self._lock = threading.Lock()

    def flush(self) -> None:
        """Deliver all queued intents (deferred mode)."""
        with self._lock:
            pending, self._pending = self._pending, []
            deferred, self.deferred = self.deferred, False
        try:
            for intent in pending:
                self.share_intent(intent)
        finally:
            self.deferred = deferred

    def share_intent(self, intent: DecMCTSIntent) -> None:
        """Share intent from an agent with the agents currently in range."""
        if self.deferred:
            with self._lock:
                self._pending.append(intent)
            return
        with self._lock:
            self._intents[intent.agent_id] = intent
            if not self.enabled:
                return
            if self.neighbor_fn is None:
                receivers = [a for a in range(self.num_agents) if a != intent.agent_id]
            else:
                receivers = [a for a in self.neighbor_fn(intent.agent_id) if a != intent.agent_id]
            for receiver in receivers:
                self._inbox.setdefault(receiver, {})[intent.agent_id] = intent

    def get_teammate_intents(self, agent_id: int) -> Dict[int, DecMCTSIntent]:
        """Intents this agent has received (last one per teammate)."""
        with self._lock:
            return dict(self._inbox.get(agent_id, {}))

    def get_all_intents(self) -> Dict[int, DecMCTSIntent]:
        """Get all intents."""
        with self._lock:
            return self._intents.copy()


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

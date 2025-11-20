import copy
import numpy as np
from helper import uav_position, H, expected_posterior
from new_camera import Camera
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil, os
from typing import List, Tuple, Dict


# @dataclass
class LLIntent:
    end_xy: Tuple[float, float]
    duration: float
    footprints: List[Tuple[np.ndarray, float]]  # list of (mask_ij, dt)
    ig_per_sec: float


def log_mem(prefix=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"[{prefix}] Memory RSS: {rss:.1f} MB")


def copy_state(state):
    # Deepcopy for dicts, but use .copy() for the NumPy array for efficiency
    new_state = {
        "uav_pos": copy.deepcopy(state["uav_pos"]),
        "belief": state["belief"].copy(),
    }
    if "remaining_steps" in state:
        new_state["remaining_steps"] = int(state["remaining_steps"])
    if "covered_mask" in state:
        new_state["covered_mask"] = state["covered_mask"].copy()
    return new_state


class MCTSNode:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.
    Each node corresponds to a state in the search space and contains information
    about its parent, children, action taken to reach it, visit count, and total reward.
    """

    def __init__(
        self, state, camera, parent=None, action=None, conf_dict=None, plan_cfg=None
    ):
        self.state = copy_state(
            state
        )  # The state represented by this node state = {'uav_pos': uav_pos, 'belief': current_belief_map}  # State representation
        self.parent = parent  # Parent node in the MCTS tree
        self.action_from_parent = action  # Action taken to reach this node
        self.children = {}  # Dictionary of child nodes indexed by action
        self.visit_count = 0  # Number of times this node has been visited.
        self.value = 0.0  # Total reward accumulated from this node's visits
        self.camera = camera  # Camera object for action space and position updates
        self.untried_actions = sorted(camera.permitted_actions(self.state["uav_pos"]))
        self.conf_dict = conf_dict  # Optional configuration dictionary for sensor model
        self.lock = threading.Lock()  # For thread-safe updates
        self._rng = np.random.default_rng()

        self.plan_cfg = plan_cfg or {}

    def _filtered_actions_for_state(self, s):
        # No actions if budget exhausted
        if s.get("remaining_steps", 1) <= 0:
            return []
        actions = sorted(self.camera.permitted_actions(s["uav_pos"]))
        # Enforce h_max: remove lowering alt
        if self.plan_cfg.get("enforce_hmax", False):
            if "down" in actions:
                actions.remove("down")
            if (
                # abs(s["uav_pos"].altitude - self.camera.h_range[-1]) > 1e-1
                # and
                "up"
                in actions
            ):
                actions = ["up"]
        return actions

    def is_fully_expanded(self):
        # all possible actions have been tried
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # select child using UCB:the highest Upper Confidence Bound
        best_node, best_score = None, -float("inf")
        N = max(1, self.visit_count)

        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploitation = child.value / child.visit_count  # the average reward
                exploration = c_param * np.sqrt(
                    2.0 * np.log(self.visit_count) / child.visit_count
                )
                ucb = exploitation + exploration
            if ucb > best_score:
                best_score, best_node = ucb, child
        return best_node

    def is_terminal(self):
        # return False
        if "remaining_steps" not in self.state:
            return False
        return self.state.get("remaining_steps", 1) <= 0

    def apply_action(self, state, action, copy_belief=True):
        if copy_belief:
            next_state = copy_state(state)  # used in EXPANSION
        else:
            next_state = state  # used in ROLLOUT
        next_state["uav_pos"] = uav_position(
            self.camera.x_future(action, x=state["uav_pos"])
        )
        if "remaining_steps" in next_state:
            next_state["remaining_steps"] = max(0, next_state["remaining_steps"] - 1)
        return next_state

    def sensor_model(self, x_future):
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x_future.altitude))

        if self.conf_dict is not None:
            s0, s1 = self.conf_dict[np.round(x_future.altitude, decimals=2)]
        else:
            s0, s1 = sigma, sigma
        return s0, s1

    def expand(self):
        # add child for untried action
        action = self.untried_actions.pop()
        new_state = self.apply_action(self.state, action, copy_belief=True)
        child = MCTSNode(
            new_state,
            camera=self.camera,
            parent=self,
            action=action,
            conf_dict=self.conf_dict,
            plan_cfg=self.plan_cfg,
        )
        self.children[action] = child
        return child

    def belief_update(
        self,
        belief,
        obsd_m_i_min,
        obsd_m_i_max,
        obsd_m_j_min,
        obsd_m_j_max,
        Pz0,
        Pz1,
        p_m1_z0,
        p_m1_z1,
    ):
        """Update belief over the observed area using expected posterior."""
        expected_post = Pz1 * p_m1_z1 + Pz0 * p_m1_z0
        # belief = belief.copy()
        belief[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1] = expected_post
        belief[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 0] = (
            1 - expected_post
        )
        return belief

    def compute_reward(self, obs_submap, Pz0, Pz1, p_m1_z0, p_m1_z1):
        curr_entropy = H(obs_submap)
        expected_entropy = Pz0 * H(p_m1_z0) + Pz1 * H(p_m1_z1)
        return np.sum(curr_entropy - expected_entropy)

    def _coverage_reward(self, state, imin, imax, jmin, jmax):
        # Ensure covered_mask exists
        H, W = state["belief"].shape[:2]
        if "covered_mask" not in state:
            state["covered_mask"] = np.zeros((H, W), dtype=bool)
        sub = state["covered_mask"][imin:imax, jmin:jmax]
        newly = int((~sub).sum())
        state["covered_mask"][imin:imax, jmin:jmax] = True
        # Normalize by field size

        return newly / max(1, H * W)

    def rollout(
        self,
        rng,
        max_depth=10,
        discount_factor=1.0,
    ):
        # TODO currently assumes expected posterior, could be changed to sampled observation
        # simulate random steps from this state, return reward
        # log_mem("rollout start")
        state = copy_state(self.state)
        discount = 1.0
        total_reward = 0
        for t in range(max_depth):
            # 1. Get permitted actions from camera
            actions = self._filtered_actions_for_state(state)
            # actions = self.camera.permitted_actions(state["uav_pos"])
            # actions = sorted(self.camera.permitted_actions(state["uav_pos"]))

            # 2. Select an action randomly TODO could be smarter

            action = actions[rng.integers(len(actions))]
            # action = np.random.choice(actions)

            # 3. Apply the action to get the next state
            state = self.apply_action(state, action, copy_belief=False)
            s0, s1 = self.sensor_model(state["uav_pos"])

            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=state["uav_pos"].position,
                altitude=state["uav_pos"].altitude,
                index_form=True,
            )

            # 4. Simulate observation and
            obs_submap = state["belief"][imin:imax, jmin:jmax, 1]
            a, b, p10, p11 = expected_posterior(obs_submap, s0, s1)

            # 5. Compute immediate reward (e.g., coverage or info gain)
            if self.plan_cfg.get("use_coverage_reward", True):
                reward = self._coverage_reward(state, imin, imax, jmin, jmax)
            else:
                reward = self.compute_reward(obs_submap, a, b, p10, p11)

            discount *= discount_factor
            total_reward += discount * reward

            # 6. Update belief based on observation
            state["belief"] = self.belief_update(
                state["belief"], imin, imax, jmin, jmax, a, b, p10, p11
            )

        return total_reward

    def backpropagate(self, reward):
        # update value and visits back up to root
        node = self
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

    @staticmethod
    def apply_virtual_loss(path, vloss=1.0):
        # No locks needed if only main thread calls this
        for n in path:
            n.visit_count += 1
            n.value -= vloss

    @staticmethod
    def backprop_with_reward(path, reward, vloss=1.0):
        # Cancel the virtual loss, keep the visit that was already counted
        for n in path:
            n.value += vloss + reward

    def print_tree(self, max_depth=2, indent="", action_from_parent=None):
        """Recursively print the tree up to max_depth."""
        if action_from_parent is not None:
            prefix = f"{indent}Action: {action_from_parent} | Visits: {self.visit_count} | Value: {self.value:.2f} | Avg: {self.value/max(1,self.visit_count):.2f}"
        else:
            prefix = (
                f"{indent}ROOT | Visits: {self.visit_count} | Value: {self.value:.2f}"
            )
        print(prefix)
        if max_depth > 0:
            for action, child in self.children.items():
                child.print_tree(
                    max_depth=max_depth - 1,
                    indent=indent + "  ",
                    action_from_parent=action,
                )


class MCTSPlanner:
    def __init__(
        self,
        initial_state,
        uav_camera,
        conf_dict=None,
        discount_factor=1.0,
        max_depth=10,
        parallel=1,
        ucb1_c=1.4,
        seed=None,
        plan_cfg=None,
    ):
        plan_cfg = plan_cfg or {}
        # Initialize coverage mask if not present
        if "covered_mask" not in initial_state:
            H, W = initial_state["belief"].shape[:2]
            initial_state["covered_mask"] = np.zeros((H, W), dtype=bool)

        self.root = MCTSNode(initial_state, uav_camera, conf_dict=conf_dict)
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.parallel = parallel
        self.ucb1_c = ucb1_c
        self._rng = np.random.default_rng(seed)

    def _simulate_only(self, node, seed):
        # Worker thread function: NO TREE TOUCHING
        rng = np.random.default_rng(seed)
        return node.rollout(
            rng=rng, discount_factor=self.discount_factor, max_depth=self.max_depth
        )

    def search(self, num_iterations=100, timeout=None, return_action_scores=False):
        start_time = time.time()
        if self.parallel == 1:
            # Serial execution (default)
            for _ in range(num_iterations):
                if timeout is not None and (time.time() - start_time) >= timeout:
                    break
                node, path = self.tree_policy()
                reward = node.rollout(
                    rng=self._rng,
                    discount_factor=self.discount_factor,
                    max_depth=self.max_depth,
                )
                node.backpropagate(reward)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {}
                for _ in range(num_iterations):
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        break
                    node, path = self.tree_policy()
                    MCTSNode.apply_virtual_loss(path, vloss=1.0)
                    child_seed = int(self._rng.integers(2**32))
                    # Schedule the rollout
                    fut = executor.submit(self._simulate_only, node, child_seed)
                    futures[fut] = path

                for fut in as_completed(futures):
                    try:
                        reward = fut.result()
                    except Exception as e:
                        import traceback

                        print("⚠️ Exception in rollout:", e)
                        traceback.print_exc()
                        continue
                    path = futures[fut]
                    # node.backpropagate(reward)
                    MCTSNode.backprop_with_reward(path, reward, vloss=1.0)

        # def search(self, num_iterations=100, return_action_scores=False, timeout=None):
        #     start_time = time.time()
        #     for _ in range(num_iterations):
        #         if timeout is not None and (time.time() - start_time) >= timeout:
        #             print(
        #                 f"Timeout reached after {(time.time() - start_time):.2f} seconds."
        #             )
        #             break
        #         node = self.tree_policy()
        #         reward = node.rollout(discount_factor=self.discount_factor)
        #         node.backpropagate(reward)

        best_action = self.best_action()
        if return_action_scores:
            action_scores = {
                action: child.value / max(child.visit_count, 1)
                for action, child in self.root.children.items()
            }
            return best_action, action_scores
        return best_action

    def tree_policy(self):
        node = self.root
        path = [node]
        while not node.is_terminal():  # You can define is_terminal if needed
            if not node.is_fully_expanded():
                # return node.expand()
                child = node.expand()
                path.append(child)
                return child, path
            else:
                node = node.best_child(c_param=self.ucb1_c)
                path.append(node)
        return node, path

    # def best_action(self):
    #     # Choose the action from root's children with the highest visit count
    #     best = max(self.root.children.values(), key=lambda n: n.visit_count)
    #     return best.action_from_parent

    def best_action(self):
        # Choose the action from root's children with the highest avg value
        if not self.root.children:
            return None  # no simulations produced children
        best = max(
            self.root.children.values(), key=lambda n: n.value / max(1, n.visit_count)
        )
        return best.action_from_parent

    def extract_solution(self, max_depth=None, return_states=False):
        """
        Traverse the built tree greedily (c=0) to get the final solution.
        Stops at leaf or max_depth. Returns action sequence (and states if requested).
        """
        node = self.root
        actions = []
        states = [copy_state(node.state)] if return_states else None
        depth = 0

        while node.children:
            # print(
            #     f"At d={depth}, node at {node.state['uav_pos']} has {len(node.children)} children."
            # )
            # print(f"children actions: {list(node.children.keys())}")
            if max_depth is not None and depth >= max_depth:
                print(f"Reached max_depth in extract_solution depth={depth}")
            next_node = node.best_child(c_param=0.0)
            actions.append(next_node.action_from_parent)
            node = next_node
            depth += 1
            if return_states:
                states.append(copy_state(node.state))
            if not node.children:
                print(f"next node= {node.state['uav_pos']} has no children to continue")
            # else:
            # print(
            #     f"next node= {node.state['uav_pos']}  has {len(node.children)} children:{list(node.children.keys())}   "
            # )
        return (actions, states) if return_states else actions

    def visualize_tree(self, max_depth=2):
        self.root.print_tree(max_depth=max_depth)


def test_mcts_planner_search_and_best_action():
    class DummyGrid:
        def __init__(self):
            self.x = 60
            self.y = 110
            self.length = 1
            self.center = False
            self.shape = (60, 110)

    grid = DummyGrid()
    camera = Camera(
        grid, fov_angle=60, xy_step=1, h_range=[3, 5, 7, 12], camera_altitude=3
    )
    uav_pos = uav_position(((25, 40), 5))
    state = {"uav_pos": uav_pos, "belief": np.ones((60, 110, 2)) * 0.5}
    print(
        f"Initial state: position=({uav_pos.position},{uav_pos.altitude}), belief shape={state['belief'].shape}"
    )
    planner = MCTSPlanner(
        state, camera, conf_dict=None, discount_factor=1.0, max_depth=20, parallel=4
    )
    num_iterations = 100
    action, score = planner.search(
        num_iterations=num_iterations, return_action_scores=True
    )
    permitted = camera.permitted_actions(uav_pos)
    # planner.visualize_tree(max_depth=6)

    print("Permitted actions:", permitted)
    print("Best action returned by planner:", action)
    print("Action scores:", score)
    assert action in permitted, "Planner should return one of the permitted actions"
    assert (
        planner.root.visit_count == num_iterations
    ), "Root node visit count should equal num_iterations"
    assert planner.root.value >= 0, "Root node value should be non-negative"
    print("test_mcts_planner_search_and_best_action passed.")


# if __name__ == "__main__":
#     test_mcts_planner_search_and_best_action()

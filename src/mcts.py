import copy
import numpy as np
from helper import uav_position, H, expected_posterior
from new_camera import Camera
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class DummyGrid:
    def __init__(self):
        self.x = 60
        self.y = 110
        self.length = 1
        self.center = False
        self.shape = (60, 110)


grid = DummyGrid()


def copy_state(state):
    # Deepcopy for dicts, but use .copy() for the NumPy array for efficiency
    new_state = {
        "uav_pos": copy.deepcopy(state["uav_pos"]),
        "belief": state["belief"].copy(),  # or np.copy(state['belief'])
    }
    return new_state


class MCTSNode:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.
    Each node corresponds to a state in the search space and contains information
    about its parent, children, action taken to reach it, visit count, and total reward.
    """

    def __init__(self, state, camera=None, parent=None, action=None, conf_dict=None):
        self.state = copy_state(
            state
        )  # The state represented by this node state = {'uav_pos': uav_pos, 'belief': current_belief_map}  # State representation
        self.parent = parent  # Parent node in the MCTS tree
        self.action_from_parent = action  # Action taken to reach this node
        self.children = {}  # Dictionary of child nodes indexed by action
        self.visit_count = 0  # Number of times this node has be = 0.0
        self.value = 0.0  # Total reward accumulated from this node's visits
        self.camera = camera  # Camera object for action space and position updates
        self.untried_actions = (
            set(camera.permitted_actions(self.state["uav_pos"]))
            if camera
            else set("up down front back left right hover".split())
        )
        self.conf_dict = conf_dict  # Optional configuration dictionary for sensor model

    def is_fully_expanded(self):
        # all possible actions have been tried
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # select child using UCB:the highest Upper Confidence Bound
        best_node = None
        best_score = -float("inf")
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploitation = child.value / child.visit_count  # the average reward
                exploration = c_param * np.sqrt(
                    2 * np.log(self.visit_count) / child.visit_count
                )  # higher for less-visited nodes
                ucb = exploitation + exploration
            if ucb > best_score:
                best_score = ucb
                best_node = child
        return best_node

    def is_terminal(self):
        return False

    def apply_action(self, state, action):
        next_state = copy_state(state)

        next_state["uav_pos"] = uav_position(
            self.camera.x_future(action, x=state["uav_pos"])
        )

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
        new_state = self.apply_action(self.state, action)
        child = MCTSNode(
            new_state,
            camera=self.camera,
            parent=self,
            action=action,
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
        belief = belief.copy()
        belief[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1] = expected_post
        belief[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 0] = (
            1 - expected_post
        )
        return belief

    def compute_reward(self, obs_submap, Pz0, Pz1, p_m1_z0, p_m1_z1):
        curr_entropy = H(obs_submap)
        expected_entropy = Pz0 * H(p_m1_z0) + Pz1 * H(p_m1_z1)
        return np.sum(curr_entropy - expected_entropy)

    def rollout(self, max_depth=10, discount_factor=1.0):
        # simulate random steps from this state, return reward
        state = copy_state(self.state)
        discount = 1.0
        total_reward = 0
        for t in range(max_depth):
            # 1. Get permitted actions from camera
            actions = self.camera.permitted_actions(state["uav_pos"])

            if not actions:
                break
            # 2. Pick an action (random or simple policy)
            # Deterministic or random action selection can be handled here if needed
            action = np.random.choice(actions)

            # 3. Apply the action to get the next state
            state = self.apply_action(state, action)
            s0, s1 = self.sensor_model(state["uav_pos"])

            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=state["uav_pos"].position,
                altitude=state["uav_pos"].altitude,
                index_form=True,
            )

            obs_submap = state["belief"][imin:imax, jmin:jmax, 1]
            # 4. Simulate observation and

            a, b, p10, p11 = expected_posterior(obs_submap, s0, s1)
            # 5. Compute immediate reward (e.g., info gain)

            reward = self.compute_reward(obs_submap, a, b, p10, p11)
            total_reward += discount * reward
            # 6. Update belief based on observation
            state["belief"] = self.belief_update(
                state["belief"], imin, imax, jmin, jmax, a, b, p10, p11
            )
            discount *= discount_factor

        return total_reward

    def backpropagate(self, reward):
        # update value and visits back up to root
        node = self
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

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
    ):
        self.root = MCTSNode(initial_state, camera=uav_camera, conf_dict=conf_dict)
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.parallel = parallel

    def search(self, num_iterations=100, timeout=None, return_action_scores=False):
        start_time = time.time()
        if self.parallel == 1:
            # Serial execution (default)
            for _ in range(num_iterations):
                if timeout is not None and (time.time() - start_time) >= timeout:
                    print(
                        f"Timeout reached after {(time.time() - start_time):.2f} seconds."
                    )
                    break
                node = self.tree_policy()
                reward = node.rollout(
                    discount_factor=self.discount_factor, max_depth=self.max_depth
                )
                node.backpropagate(reward)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = []
                for _ in range(num_iterations):
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        print(
                            f"Timeout reached after {(time.time() - start_time):.2f} seconds."
                        )
                        break
                    node = self.tree_policy()
                    # Schedule the rollout and backpropagation
                    futures.append(
                        executor.submit(self._rollout_and_backpropagate, node)
                    )
                # Wait for all futures to complete
                for f in as_completed(futures):
                    pass  # Results already handled in _rollout_and_backpropagate

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
        while not node.is_terminal():  # You can define is_terminal if needed
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def best_action(self):
        # Choose the action from root's children with the highest visit count
        best = max(self.root.children.values(), key=lambda n: n.visit_count)
        return best.action_from_parent

    # def best_action(self):
    #     #Choose the action from root's children with the highest avg value
    #     best = max(self.root.children.values(), key=lambda n: n.value / max(1, n.visit_count))
    #     return best.action_from_parent
    def visualize_tree(self, max_depth=2):
        self.root.print_tree(max_depth=max_depth)

    def _rollout_and_backpropagate(self, node):
        reward = node.rollout(
            discount_factor=self.discount_factor, max_depth=self.max_depth
        )
        node.backpropagate(reward)


def test_mcts_node_expand():
    # Simple grid and camera setup

    global grid
    camera = Camera(grid, fov_angle=60, xy_step=1, h_range=[3, 5, 7], camera_altitude=3)

    # Initial belief and state
    belief = np.ones((60, 110, 2)) * 0.5
    uav_pos = uav_position(((5, 10), 3))
    state = {"uav_pos": uav_pos, "belief": belief}

    node = MCTSNode(state, camera=camera)

    # Check that untried_actions matches camera's permitted actions
    expected_actions = set(camera.permitted_actions(uav_pos))
    assert set(node.untried_actions) == expected_actions, "Untried actions mismatch!"

    # Expand a node and check properties
    child = node.expand()
    assert child.parent is node
    assert child.action_from_parent in expected_actions
    assert isinstance(child.state["uav_pos"], uav_position)
    assert not np.shares_memory(
        child.state["belief"], node.state["belief"]
    ), "Belief should be copied"

    print("test_mcts_node_expand passed.")


def test_mcts_node_rollout():
    # Dummy grid, camera, and position classes

    camera = Camera(grid, fov_angle=60, xy_step=1, h_range=[3, 5, 7], camera_altitude=3)
    uav_pos = uav_position(((5, 10), 3))
    belief = np.ones((60, 110, 2)) * 0.5
    state = {"uav_pos": uav_pos, "belief": belief}

    node = MCTSNode(state, camera)
    reward = node.rollout(max_depth=3)
    print("Rollout reward:", reward)
    assert np.isscalar(reward), "Rollout should return a scalar"
    assert reward >= 0, "Reward should be non-negative"
    # Check that original belief is not mutated
    assert np.all(
        state["belief"] == 0.5
    ), "Belief in original state should remain unchanged"
    print("test_mcts_node_rollout passed.")


def test_mcts_planner_search_and_best_action():
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


if __name__ == "__main__":

    test_mcts_node_expand()
    test_mcts_node_rollout()
    test_mcts_planner_search_and_best_action()

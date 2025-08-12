import random
import numpy as np
import math
from helper import uav_position, H, cH
from mcts import MCTSPlanner


class planning:
    def __init__(self, grid_info, uav, strategy, conf_dict=None, optimal_alt=21.6):
        # Initialize belief map (each cell has a default probability of 0.5) and set UAV planning parameters
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None

    def reset(self, conf_dict=None):
        """Reset UAV and planning state, and reinitialize the belief map."""
        self.uav.reset()
        self.conf_dict = conf_dict
        self.last_action = None
        self.M = np.ones_like(self.M) * 0.5

    def info_gain(self, var, x_future):
        """Calculate information gain for a belief state given a future UAV state."""
        ig = H(var) - self._expected_entropy(var, x_future)

        return ig

    def _expected_entropy(self, var, x_future):
        """Compute expected entropy based on future UAV state and sensor model parameters."""
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x_future.altitude))

        if self.conf_dict is not None:
            s0, s1 = self.conf_dict[np.round(x_future.altitude, decimals=2)]
        else:
            s0, s1 = sigma, sigma

        return cH(var, s0, s1)

    def sweep(self, permitted_actions, visited_x):
        """Select a sweeping action based on UAV altitude and visited positions."""
        if (
            self.uav.get_x().altitude < self.optimal_altitude
            and "up" in permitted_actions
        ):
            self.last_action = "up"
            return "up", None

        sweep_actions = []
        for action in permitted_actions:
            x_future = uav_position(self.uav.x_future(action))
            if x_future not in visited_x and action != "up" and action != "down":
                sweep_actions.append(action)

        if self.sweep_direction is None:
            if len(sweep_actions) == 1:
                self.sweep_direction = (
                    "LeftRight"
                    if sweep_actions[0] in ["left", "right"]
                    else "BackFront"
                )
            else:
                self.sweep_direction = random.choice(["LeftRight", "BackFront"])

        # self.sweep_direction = "LeftRight"
        if self.sweep_direction == "LeftRight":
            # give propriority to left or right (if one of them is present in sweep_actions, only one can be present at a time)
            if "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "back" in sweep_actions:
                self.last_action = "back"
            else:
                self.last_action = "hover"
        if self.sweep_direction == "BackFront":
            # give propriority to back or front (if one of them is present in sweep_actions, only one can be present at a time)
            if "back" in sweep_actions:
                self.last_action = "back"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            else:
                self.last_action = "hover"
        return self.last_action, None

    def ig_based(self, permitted_actions):
        """Select an action based on the maximum information gain."""
        info_gain_action = {}
        for action in permitted_actions:
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))
            info_gain_action_a = 0
            [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = (
                self.uav.get_range(
                    position=x_future.position,
                    altitude=x_future.altitude,
                    index_form=True,
                )
            )
            obs_M = self.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1]
            info_gain_action_a = np.sum(self.info_gain(obs_M, x_future))
            info_gain_action[action] = info_gain_action_a

        # Find the maximum information gain
        max_gain = max(info_gain_action.values())

        # Collect actions with the maximum info gain
        max_gain_actions = [
            action for action, gain in info_gain_action.items() if gain == max_gain
        ]

        next_action = random.choice(max_gain_actions)
        # Update previous action for the next step
        self.last_action = next_action
        return next_action, info_gain_action

    def select_action(self, belief, visited_x):
        """Select the next UAV action based on the current belief and the chosen strategy."""
        self.M = belief

        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        if self.strategy == "sweep":
            return self.sweep(permitted_actions, visited_x)
        elif self.strategy == "ig_lookahead":
            return self.ig_lookahead(permitted_actions)
        elif self.strategy == "entropy_guided":
            return self.entropy_guided(permitted_actions)
        elif self.strategy == "entropy_guided_combined":
            return self.entropy_guided_combined(permitted_actions)
        elif self.strategy == "mcts":
            return self.mcts_based()

        return self.ig_based(permitted_actions)

    def ig_lookahead(self, permitted_actions, depth=1, gamma=1):
        """Multi-step IG planner using fixed-depth rollout (loop-based)."""
        action_ig_total = {}

        for a1 in permitted_actions:
            cumulative_ig = 0
            discount = 1.0
            current_state = uav_position(self.uav.x_future(a1))

            # Step 1: apply first action
            obs_idx = self.uav.get_range(
                position=current_state.position,
                altitude=current_state.altitude,
                index_form=True,
            )
            obs = self.M[
                obs_idx[0][0] : obs_idx[0][1], obs_idx[1][0] : obs_idx[1][1], 1
            ]

            ig = np.sum(self.info_gain(obs, current_state))
            cumulative_ig += discount * ig

            # Step 2+: rollout deterministically
            for _ in range(depth - 1):
                discount *= gamma
                permitted_next = self.uav.permitted_actions(current_state)
                best_ig = -1
                best_next = None
                for a_next in permitted_next:

                    next_state = uav_position(
                        self.uav.x_future(a_next, x=current_state)
                    )
                    next_obs_idx = self.uav.get_range(
                        position=next_state.position,
                        altitude=next_state.altitude,
                        index_form=True,
                    )
                    next_obs = self.M[
                        next_obs_idx[0][0] : next_obs_idx[0][1],
                        next_obs_idx[1][0] : next_obs_idx[1][1],
                        1,
                    ]

                    ig = np.sum(self.info_gain(next_obs, next_state))
                    if ig > best_ig:
                        best_ig = ig
                        best_next = next_state
                if best_next is not None:
                    cumulative_ig += discount * best_ig
                    current_state = best_next
                else:
                    break  # No more moves

            action_ig_total[a1] = cumulative_ig

        max_ig = max(action_ig_total.values())
        best_actions = [a for a, val in action_ig_total.items() if val == max_ig]
        next_action = random.choice(best_actions)
        self.last_action = next_action
        return next_action, action_ig_total

    def entropy_guided(self, permitted_actions, beta=0.2):
        """New strategy that steers UAV toward high-entropy regions with local IG."""
        info_score = {}

        # Compute global entropy map and get target i,j
        entropy_map = H(self.M[:, :, 1])
        target_ij = np.unravel_index(np.argmax(entropy_map), entropy_map.shape)

        # Convert target (i,j) to real-world (x,y)
        target_pos = self.uav.ij_to_xy(*target_ij)

        for action in permitted_actions:
            # Simulate next pose
            x_future = uav_position(self.uav.x_future(action))

            # Get indices of visible patch
            [[i_min, i_max], [j_min, j_max]] = self.uav.get_range(
                position=x_future.position,
                altitude=x_future.altitude,
                index_form=True,
            )

            # Compute local IG
            belief_patch = self.M[i_min:i_max, j_min:j_max, 1]
            ig = np.sum(self.info_gain(belief_patch, x_future))

            # Compute distance to global target
            pos = np.array(x_future.position)
            dist = np.linalg.norm(pos - np.array(target_pos))
            distance_bias = -beta * dist

            # Score = IG + entropy attraction
            score = ig + distance_bias
            info_score[action] = score

        # Choose best
        best_actions = [
            a for a, s in info_score.items() if s == max(info_score.values())
        ]
        next_action = random.choice(best_actions)
        self.last_action = next_action
        return next_action, info_score

    def entropy_guided_combined(
        self, permitted_actions, epsilon=1e-3, local_weight=1.0, global_weight=1.0
    ):
        entropy_map = H(self.M[:, :, 1])
        info_score = {}

        rows, cols = entropy_map.shape
        i_indices = np.arange(rows).reshape(-1, 1)  # (rows, 1)
        j_indices = np.arange(cols).reshape(1, -1)  # (1, cols)

        for action in permitted_actions:
            # Simulate future pose
            x_future = uav_position(self.uav.x_future(action))
            x_pos = x_future.position

            # Get (i, j) index of the simulated future position
            i_fut, j_fut = self.uav.convert_xy_ij(*x_pos, self.uav.grid.center)

            # Compute Manhattan distance map from x_future position
            dist_map = np.abs(i_indices - i_fut) + np.abs(j_indices - j_fut) + epsilon
            weighted_entropy_map = entropy_map / dist_map

            # Get sensor footprint
            [[i_min, i_max], [j_min, j_max]] = self.uav.get_range(
                position=x_future.position,
                altitude=x_future.altitude,
                index_form=True,
            )

            # Local IG
            belief_patch = self.M[i_min:i_max, j_min:j_max, 1]
            local_ig = np.sum(self.info_gain(belief_patch, x_future))

            # Global guidance: entropy-weighted closeness
            global_guidance = np.sum(weighted_entropy_map[i_min:i_max, j_min:j_max])

            # Final combined score
            score = local_weight * local_ig + global_weight * global_guidance
            info_score[action] = score

        # Select best action
        best_actions = [
            a for a, s in info_score.items() if s == max(info_score.values())
        ]
        next_action = random.choice(best_actions)
        self.last_action = next_action
        return next_action, info_score

    def mcts_based(self, planning_depth=10):
        uav_pos = self.uav.get_x()

        state = {"uav_pos": uav_pos, "belief": self.M.copy()}
        # print(
        #     f"Initial state: position=({state['uav_pos'].position},{uav_pos.altitude}), belief shape={state['belief'].shape}"
        # )
        mcts_planner = MCTSPlanner(
            state,
            self.uav,
            conf_dict=self.conf_dict,
            discount_factor=1,
            max_depth=planning_depth,
            parallel=8,
        )
        num_iterations = 2000
        action, score = mcts_planner.search(
            num_iterations=num_iterations, return_action_scores=True, timeout=10.0
        )
        return action, score

        # current_state = MCTSState(
        #     uav_position=self.uav.get_x(),
        #     belief_map_summary=self.compress_belief_map(self.M),
        #     steps_remaining=planning_depth,
        # )

        # # All actions treated uniformly
        # mcts = MCTS(current_state, permitted_actions)
        # return mcts.search()

    def compress_belief_map(self, belief_map=None):
        # Compress 60x110 to manageable size for MCTS nodes
        # Option 1: Grid summary (e.g., 12x22 regions)
        if belief_map is None:
            belief_map = self.M

        compressed = np.zeros((12, 22))
        for i in range(12):
            for j in range(22):
                region = belief_map[i * 5 : (i + 1) * 5, j * 5 : (j + 1) * 5, 1]
                compressed[i, j] = np.mean(H(region))  # entropy per region
        return compressed


# class MCTSState:
#     def __init__(self, uav_pos, belief_summary, steps_remaining):
#         self.uav_pos = uav_pos  # uav_pos: uav_position object representing UAV location and altitude
#         self.belief_summary = belief_summary  # Compressed belief state
#         self.steps_remaining = steps_remaining

#     def get_successor_state(self, action, uav):
#         new_position = uav_position(uav.x_future(action))  # Move

#         # Both cases: take new observation and update belief
#         return MCTSState(
#             uav_position=new_position,
#             belief_map_summary=new_belief_summary,
#             steps_remaining=self.steps_remaining - 1,
#         )

#     def is_terminal(self):
#         return self.steps_remaining <= 0

#     def __hash__(self):
#         """Make state hashable for use in dictionaries."""
#         return hash(
#             (self.uav_pos, tuple(self.belief_summary.flatten()), self.remaining_steps)
#         )

#     def __eq__(self, other):
#         """Check equality of states."""
#         if not isinstance(other, MCTSState):
#             return False
#         return (
#             self.uav_pos == other.uav_pos
#             and np.array_equal(self.belief_summary, other.belief_summary)
#             and self.remaining_steps == other.remaining_steps
#         )


# class MCTSNode:
#     def __init__(self, state, parent=None, action=None):
#         self.state = state
#         self.parent = parent
#         self.action = action  # Action that led to this state
#         self.children = {}  # action -> MCTSNode
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_actions = None  # Will be set during expansion

#     def is_fully_expanded(self):
#         return len(self.untried_actions) == 0

#     def is_terminal(self):
#         return self.state.is_terminal()

#     def is_leaf(self):
#         """Check if this is a leaf node (no children)."""
#         return len(self.children) == 0

#     def ucb1_score(self, exploration_constant=1.4):
#         if self.visits == 0:
#             return float("inf")

#         exploitation = self.total_reward / self.visits
#         exploration = exploration_constant * math.sqrt(
#             math.log(self.parent.visits) / self.visits
#         )
#         return exploitation + exploration

#     def add_child(self, action, state):
#         """Add a new child node."""
#         child = MCTSNode(state, parent=self, action=action)
#         self.children[action] = child
#         return child

#     def select_child(self):
#         """Select child with highest UCB1 score."""
#         return max(self.children.values(), key=lambda child: child.ucb1_score())

#     def expand(self, action, child_state):
#         """Add a new child node for the given action."""
#         child = MCTSNode(child_state, parent=self, action=action)
#         self.children[action] = child
#         self.untried_actions.remove(action)
#         return child

#     def update(self, reward):
#         """Update node statistics with reward from simulation."""
#         self.visits += 1
#         self.total_reward += reward

#     def backpropagate(self, reward):
#         """Backpropagate reward up the tree."""
#         self.update(reward)
#         if self.parent is not None:
#             self.parent.backpropagate(reward)


# class MCTS:
#     def __init__(
#         self, root_state, permitted_actions, planner, exploration_constant=1.4
#     ):
#         self.root = MCTSNode(root_state)
#         self.root.untried_actions = permitted_actions.copy()

#         self.exploration_constant = exploration_constant

#     def search(self, iterations=1000):
#         """Run MCTS search and return the best action."""
#         for _ in range(iterations):
#             # Selection and Expansion
#             node = self._select_and_expand()

#             # Simulation
#             reward = self._simulate(node)

#             # Backpropagation
#             node.backpropagate(reward)

#         # Return action with highest visit count (most explored)
#         if not self.root.children:
#             # If no children were expanded, return a random action
#             return random.choice(self.root.untried_actions), {}

#         best_action = max(
#             self.root.children.keys(), key=lambda a: self.root.children[a].visits
#         )

#         # Return action scores for compatibility
#         action_scores = {
#             action: child.total_reward / max(child.visits, 1)
#             for action, child in self.root.children.items()
#         }

#         return best_action, action_scores

#     def _select_and_expand(self):
#         """Selection phase: traverse tree using UCB1, then expand if needed."""
#         node = self.root

#         # Selection: traverse down the tree
#         while not node.is_terminal() and node.is_fully_expanded():
#             node = node.select_child()

#         # Expansion: add a new child if possible
#         if not node.is_terminal() and not node.is_fully_expanded():
#             action = random.choice(node.untried_actions)
#             # Get successor state
#             child_state = node.state.get_successor_state(
#                 action, self.planner.uav, self.planner
#             )
#             # Get permitted actions for the child state
#             child_permitted_actions = self.planner.uav.permitted_actions(
#                 child_state.position
#             )
#             node = node.expand(action, child_state)
#             node.untried_actions = child_permitted_actions.copy()

#         return node

#     def _simulate(self, node):
#         """Simulation phase: run random rollout from current node."""
#         current_state = node.state
#         total_reward = 0.0
#         discount = 1.0

#         # Run simulation until terminal state or max depth
#         while not current_state.is_terminal():
#             # Get permitted actions for current state
#             permitted_actions = self.planner.uav.permitted_actions(
#                 current_state.position
#             )
#             if not permitted_actions:
#                 break

#             # Choose random action
#             action = random.choice(permitted_actions)

#             # Calculate immediate reward
#             reward = self.planner.mcts_reward(current_state, action)
#             total_reward += discount * reward

#             # Get next state
#             current_state = current_state.get_successor_state(
#                 action, self.planner.uav, self.planner
#             )
#             discount *= 0.9  # Discount factor for future rewards

#         return total_reward

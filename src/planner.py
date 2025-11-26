import random
import numpy as np
import math
from helper import uav_position, H, cH
from mcts import MCTSPlanner

from high_planner import build_clusters, make_ll_intent, HighPlanner, RobotHL


class planning:

    def __init__(
        self,
        grid_info,
        uav,
        strategy,
        conf_dict=None,
        optimal_alt=21.6,
        mcts_params=None,
    ):
        # Initialize belief map (each cell has a default probability of 0.5) and set UAV planning parameters
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None
        # MCTS parameters with defaults
        if mcts_params is None:
            mcts_params = {}
        self.mcts_params = {
            "planning_depth": mcts_params.get("planning_depth", 5),
            "num_iterations": mcts_params.get("num_iterations", 10),
            "timeout": mcts_params.get("timeout", 10.0),
            "ucb1_c": mcts_params.get("ucb1_c", 1.4),
            "parallel": mcts_params.get("parallel", 8),
            "discount_factor": mcts_params.get("discount_factor", 1.0),
            "horizon_weights": mcts_params.get("horizon_weights", {}),
        }
        
        # Logging configuration for dual-horizon
        self.log_dir = 'logs/dual_horizon'
        self.experiment_name = None

    def set_experiment_info(self, experiment_name: str, log_dir: str = None):
        """Set experiment information for logging."""
        self.experiment_name = experiment_name
        if log_dir:
            self.log_dir = log_dir
    
    def finalize_episode(self):
        """Finalize episode and log statistics for dual-horizon planner."""
        if self.strategy == "dual_horizon" and hasattr(self, '_dual_horizon_planner'):
            state = {
                'covered_mask': getattr(self, 'covered_mask', None),
                'belief': self.M.copy()
            }
            self._dual_horizon_planner.finalize_episode(state)

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
        elif self.strategy == "mcts":
            return self.mcts_based()
        elif self.strategy == "multi_horizon":
            # Use hierarchical planning with LL MCTS + HP cluster selection
            chosen_cid, ll_action, ll_scores = self.high_level_decision()
            return ll_action, ll_scores
        elif self.strategy == "dual_horizon":
            # Use dual-horizon planner
            return self.dual_horizon_decision()

        return self.ig_based(permitted_actions)

    def dual_horizon_decision(self):
        """
        Run dual-horizon planning that combines short-horizon IG exploitation
        with long-horizon coverage optimization to avoid fragmentation.
        
        Returns:
            Tuple of (selected_action, metrics_dict)
        """
        from dual_horizon_planner import DualHorizonPlanner, setup_dual_horizon_logger
        
        # Initialize logger on first call
        if not hasattr(self, '_dual_horizon_logger_initialized'):
            log_dir = getattr(self, 'log_dir', 'logs/dual_horizon')
            exp_name = getattr(self, 'experiment_name', None)
            log_file = setup_dual_horizon_logger(log_dir=log_dir, experiment_name=exp_name)
            print(f"\n[DUAL HORIZON] Logging to: {log_file}\n")
            self._dual_horizon_logger_initialized = True
        
        # Build state dict for dual-horizon planner
        state = {
            'uav_pos': self.uav.get_x(),
            'belief': self.M.copy(),
            'covered_mask': getattr(self, 'covered_mask', None)
        }
        
        # Initialize covered_mask if not present
        if state['covered_mask'] is None:
            H_dim, W_dim = self.M.shape[:2]
            state['covered_mask'] = np.zeros((H_dim, W_dim), dtype=bool)
            self.covered_mask = state['covered_mask']
        
        # Get horizon weights from mcts_params
        horizon_weights = self.mcts_params.get('horizon_weights', {})
        
        # Create or reuse dual-horizon planner
        if not hasattr(self, '_dual_horizon_planner'):
            self._dual_horizon_planner = DualHorizonPlanner(
                uav_camera=self.uav,
                conf_dict=self.conf_dict,
                mcts_params=self.mcts_params,
                horizon_weights=horizon_weights
            )
        
        planner = self._dual_horizon_planner
        
        # Run dual-horizon planning
        action, metrics = planner.select_action(state, strategy='dual')
        
        # Update covered_mask based on chosen action
        x_future = uav_position(self.uav.x_future(action))
        [[imin, imax], [jmin, jmax]] = self.uav.get_range(
            position=x_future.position,
            altitude=x_future.altitude,
            index_form=True
        )
        self.covered_mask[imin:imax, jmin:jmax] = True
        
        # Return action and scores (combined_scores or action_scores)
        scores = metrics.get('combined_scores', metrics.get('action_scores', {}))
        
        return action, scores

    def mcts_based(self, action_seq=False, **kwargs):
        """
        MCTS-based action selection with configurable parameters for experiments.

        Args:
            planning_depth (int): Maximum depth for MCTS tree search
            num_iterations (int): Number of MCTS iterations to perform
            timeout (float): Maximum time in seconds for MCTS search
            ucb1_c (float): UCB1 exploration constant
            parallel (int): Number of parallel processes/threads
            discount_factor (float): Discount factor for future rewards (gamma)
            action_seq (bool): If True, returns full action sequence instead of just best action

        Returns:
            tuple: (selected_action, action_scores) or (action_sequence, scores) if action_seq=True
        """

        uav_pos = self.uav.get_x()
        # Merge stored parameters with any provided overrides
        params = {**self.mcts_params, **kwargs}

        state = {"uav_pos": uav_pos, "belief": self.M.copy()}
        mcts_planner = MCTSPlanner(
            state,
            self.uav,
            conf_dict=self.conf_dict,
            discount_factor=params["discount_factor"],
            max_depth=params["planning_depth"],
            parallel=params["parallel"],
            ucb1_c=params["ucb1_c"],
        )
        if action_seq:
            action_seq = mcts_planner.extract_solution(
                max_depth=params["planning_depth"], return_states=False
            )
            return action_seq

        action, score = mcts_planner.search(
            num_iterations=params["num_iterations"],
            return_action_scores=True,
            timeout=params["timeout"],
        )
        # print(f"MCTS seq selected action sequence: {action_seq}")
        return action, score

    # def compress_belief_map(self, belief_map=None):
    #     # Compress 60x110 to manageable size for MCTS nodes
    #     # Option 1: Grid summary (e.g., 12x22 regions)
    #     if belief_map is None:
    #         belief_map = self.M

    #     compressed = np.zeros((12, 22))
    #     for i in range(12):
    #         for j in range(22):
    #             region = belief_map[i * 5 : (i + 1) * 5, j * 5 : (j + 1) * 5, 1]
    #             compressed[i, j] = np.mean(H(region))  # entropy per region
    #     return compressed
    def high_level_decision(self, robot_speed=1.0, robot_budget=100, hp_iters=400):
        """
        Runs LL MCTS to get the exploitation plan,
        converts it to an intent, and asks HP for the next cluster to pursue.
        Returns (chosen_cid, ll_selected_action, ll_action_scores)
        """
        # 1) Run LL MCTS
        action_seq = self.mcts_based(action_seq=True)
        ll_action, ll_scores = self.mcts_based(action_seq=False)

        # 3) Make LL intent (uses  UAV+Camera and helper funcs)
        ll_intent = make_ll_intent(self.uav, self.M, action_seq, dt_per_step=1.0)

        # 4) Build/refresh clusters from current belief
        cid_of_cell, clusters = build_clusters(self.M, tile_h=None, tile_w=None)

        # convert cluster dict to simple numeric entropies; keep as is for HP
        # 5) Prepare HP inputs
        robots = [
            RobotHL(
                xy=tuple(self.uav.get_x().position),
                speed=robot_speed,
                budget=robot_budget,
            )
        ]

        # 6) Plan high-level
        hp = HighPlanner(iterations=hp_iters)
        chosen_cid = hp.plan(robots, clusters, cid_of_cell, ll_intent)
        print(f"High-level selected cluster id: {chosen_cid}")

        return chosen_cid, ll_action, ll_scores

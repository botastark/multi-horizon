import numpy as np
from helper import uav_position, H, cH
from mcts import MCTSPlanner


class planning:

    def __init__(
        self,
        grid_info,
        uav,
        strategy,
        conf_dict=None,
        optimal_alt=21.6,
        mcts_params=None,
        agent_id: int = 0,
        coordinator=None,
        seed=None,
        debug_logs=False,
        experiment_config=None,
    ):
        # Initialize belief map (each cell has a default probability of 0.5) and set UAV planning parameters
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.experiment_config = experiment_config or {}  # full experiment config dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None
        self.agent_id = agent_id
        self.coordinator = coordinator
        self.seed = seed
        self.debug_logs = debug_logs
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
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Logging configuration
        self.log_dir = "logs"
        self.experiment_name = None

    def set_experiment_info(self, experiment_name: str, log_dir: str = None):
        """Set experiment information for logging."""
        self.experiment_name = experiment_name
        if log_dir:
            self.log_dir = log_dir

    def finalize_episode(self):
        """Finalize episode and log statistics for planners."""
        # Log Dec-MCTS stats
        if (
            self.debug_logs
            and self.strategy == "dec_mcts"
            and hasattr(self, "_dec_mcts_planner")
        ):
            stats = self._dec_mcts_planner.get_statistics()
            print(f"\n[DEC-MCTS] Final Stats for Agent {self.agent_id}:")
            print(f"  Plans generated: {stats['plans_generated']}")
            print(f"  Total iterations: {stats['total_iterations']}")
            print(f"  Avg planning time: {stats['avg_planning_time']*1000:.1f}ms")
            print(f"  Intent updates received: {stats['intent_updates_received']}\n")

        # Log Multi-Horizon Dec-MCTS stats
        if (
            self.debug_logs
            and self.strategy in ("mh_dec_mcts_efficient", "mh_dec_mcts_full")
            and hasattr(self, "_hierarchical_planner")
        ):
            stats = self._hierarchical_planner.get_statistics()
            print(f"\n[MH DEC-MCTS] Final Stats for Agent {self.agent_id}:")
            print(f"  Planning cycles: {stats['hierarchical']['planning_cycles']}")
            print(
                f"  LL intents broadcast: {stats['hierarchical']['ll_intents_broadcast']}"
            )
            print(
                f"  HL intents broadcast: {stats['hierarchical']['hl_intents_broadcast']}"
            )
            print(
                f"  LLP: plans={stats['llp']['plans_generated']}, iterations={stats['llp']['total_iterations']}"
            )
            print(
                f"  HLP: plans={stats['hlp']['plans_generated']}, intent_updates={stats['hlp']['intent_updates_received']}"
            )
            print(f"  Intent Bus: {stats['intent_bus']}\n")

        # Log Greedy IG stats
        if (
            self.debug_logs
            and self.strategy == "greedy_ig"
            and hasattr(self, "_greedy_ig_planner")
        ):
            stats = self._greedy_ig_planner.get_statistics()
            mode = "IGd" if self._greedy_ig_planner.enable_discounting else "IG"
            print(f"\n[GREEDY {mode}] Final Stats for Agent {self.agent_id}:")
            print(f"  Plans generated: {stats['plans_generated']}")
            print(f"  Total IG: {stats['total_ig']:.4f}")
            if self._greedy_ig_planner.enable_discounting:
                print(f"  Total IGd (discounted): {stats['total_igd']:.4f}")
                print(
                    f"  Teammate state updates received: {stats['teammate_updates_received']}\n"
                )
            else:
                print()

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
        from helper import get_sensor_params

        s0, s1 = get_sensor_params(x_future.altitude, self.conf_dict)
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
                self.sweep_direction = self.rng.choice(["LeftRight", "BackFront"])

        # self.sweep_direction = "LeftRight"
        if self.sweep_direction == "LeftRight":
            # give priority to left or right (if one of them is present in sweep_actions, only one can be present at a time)
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
            # give priority to back or front (if one of them is present in sweep_actions, only one can be present at a time)
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
        eps = 1e-4
        # Find the maximum information gain
        max_gain = max(info_gain_action.values())
        # Collect actions with the maximum info gain + eps tolerance
        max_gain_actions = [
            action
            for action, gain in info_gain_action.items()
            if gain >= max_gain - eps
        ]

        next_action = self.rng.choice(max_gain_actions)
        # Update previous action for the next step
        self.last_action = next_action
        return next_action, info_gain_action

    def greedy_ig_decision(self):
        """
        Run greedy IG planning with multi-agent support.

        Paper's approach (pure belief-based coordination):
        - Single-step IG computation using FUSED belief
        - No overlap penalties or intent-based coordination
        - Coordination emerges from Bayesian news fusion
        - Areas observed by teammates have lower entropy -> lower IG

        IGd variant (when enable_discounting=True):
        - NULL POLICY ASSUMPTION: teammates remain at current positions
        - Discount factor α_ij = 1 - IoU(fp(my_next), fp(teammate_current))
        - Discounted IG: IG_a^d = IG_a × Π_{j ∈ neighbors} α_ij

        Returns:
            Tuple of (selected_action, action_scores_dict)
        """
        from greedy_ig_planner import (
            GreedyIGPlanner,
            log_greedy_ig_decision,
            create_greedy_ig_planner,
        )

        # Create or get greedy IG planner
        if not hasattr(self, "_greedy_ig_planner"):
            # Read greedy_ig config from the full config (via coordinator) not from
            # conf_dict which is the sensor model dict (altitude -> (s0, s1))
            full_config = {}
            if self.coordinator is not None and hasattr(self.coordinator, "config"):
                full_config = self.coordinator.config
            greedy_config = full_config.get("greedy_ig", {})

            enable_discounting = greedy_config.get("enable_discounting", False)
            # Also auto-detect from coordinator mode as a fallback
            if (
                not enable_discounting
                and self.coordinator is not None
                and hasattr(self.coordinator, "mode")
            ):
                if "IGd" in self.coordinator.mode:
                    enable_discounting = True

            config = {
                "intent_discount": 0.0,  # Paper: no intent-based coordination
                "overlap_penalty_weight": 0.0,  # Paper: no penalty terms
                "enable_discounting": enable_discounting,
            }

            self._greedy_ig_planner = create_greedy_ig_planner(
                agent_id=self.agent_id,
                camera=self.uav,
                grid_info=self.uav.grid,
                conf_dict=self.conf_dict,
                config=config,
                seed=0,
            )

            mode = (
                "IGd (null policy, footprint discount)"
                if config["enable_discounting"]
                else "IG"
            )
            if self.debug_logs:
                print(f"\n[GREEDY {mode}] Agent {self.agent_id} initialized")
                print(
                    f"  Paper approach: pure belief-based IG (no penalties, no intents)\n"
                )

        planner = self._greedy_ig_planner

        # Update belief - this should be the FUSED belief from coordinator
        planner.update_belief(self.M.copy())

        # Get teammate current states for IGd null policy (if enabled)
        if self.coordinator is not None and planner.enable_discounting:
            # Get teammate current positions and altitudes (null policy assumption)
            teammate_states = {}
            neighbor_ids = set(self.coordinator.get_neighbors_in_range(self.agent_id))
            other_agents = self.coordinator.get_other_agent_positions(self.agent_id)
            for tid, pos, alt in other_agents:
                if tid in neighbor_ids:
                    teammate_states[tid] = (pos, alt)

            planner.update_teammate_states(teammate_states)

        # Get current position
        uav_pos = self.uav.get_x()

        # Run planning (pure IG or IGd with discounting)
        decision = planner.plan(
            current_position=uav_pos.position,
            current_altitude=uav_pos.altitude,
        )

        # Log decision
        teammate_info = None
        # Get action and scores
        action = decision.action
        action_scores = planner.get_action_scores()

        # Add planning time and timestamps from planner
        stats = planner.get_statistics()
        if "last_planning_time_ms" in stats:
            action_scores["_timing_greedy_ms"] = stats["last_planning_time_ms"]

        # Add timestamps if available
        if hasattr(planner, "_timing_start_ms"):
            action_scores["_timing_greedy_start_ms"] = planner._timing_start_ms
            action_scores["_timing_greedy_end_ms"] = planner._timing_end_ms

        return action, action_scores

    def select_action(self, belief, visited_x):
        """Select the next UAV action based on the current belief and the chosen strategy."""
        self.M = belief

        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        if self.strategy == "sweep":
            return self.sweep(permitted_actions, visited_x)
        elif self.strategy == "mcts":
            return self.mcts_based()
        elif self.strategy == "greedy_ig":
            # Use greedy IG planner (single-step lookahead, multi-agent)
            # Supports IGd variant via enable_discounting config
            return self.greedy_ig_decision()
        elif self.strategy == "dec_mcts":
            # Use Dec-MCTS planner (single-level MCTS, multi-agent)
            return self.dec_mcts_decision()
        elif self.strategy == "mh_dec_mcts_efficient":
            # Use Multi-Horizon Dec-MCTS with LLP/HLP hierarchy (random rollout LLP)
            return self.hierarchical_dec_mcts_decision(visited_x)
        elif self.strategy == "mh_dec_mcts_full":
            # Use Multi-Horizon Dec-MCTS with MCTS for both LLP and HLP
            return self.hierarchical_dec_mcts_decision(visited_x, use_mcts_llp=True)
        elif self.strategy == "ig":
            # Legacy single-agent IG (backward compatibility)
            return self.ig_based(permitted_actions)

        # Default fallback to legacy IG
        return self.ig_based(permitted_actions)

    def dec_mcts_decision(self):
        """
        Run Dec-MCTS planning (single-level MCTS with multi-agent support).

        This is a flat MCTS planner without the LLP/HLP hierarchy:
        - Single MCTS tree for action selection
        - Intent sharing for decentralized coordination
        - D-UCT discounting for async operation
        - IG-based rollout rewards

        Use as comparison between greedy IG and multi-horizon planning.

        Returns:
            Tuple of (selected_action, action_scores)
        """
        from dec_mcts import (
            DecMCTSPlanner,
            DecMCTSCoordinator,
        )

        # Create or get Dec-MCTS planner
        if not hasattr(self, "_dec_mcts_planner"):
            # Check if coordinator has a shared Dec-MCTS coordinator
            dec_mcts_coordinator = None
            if self.coordinator is not None:
                if not hasattr(self.coordinator, "_dec_mcts_coordinator"):
                    num_agents = getattr(self.coordinator, "num_agents", 1)
                    self.coordinator._dec_mcts_coordinator = DecMCTSCoordinator(
                        num_agents=num_agents
                    )
                dec_mcts_coordinator = self.coordinator._dec_mcts_coordinator

            # Extract config
            dec_mcts_config = self.experiment_config.get("dec_mcts", {})
            dec_config = self.experiment_config.get("decentralized", {})
            d_uct_config = dec_config.get("d_uct", {})

            config = {
                "horizon": dec_mcts_config.get(
                    "horizon", self.mcts_params.get("planning_depth", 10)
                ),
                "iterations": dec_mcts_config.get(
                    "iterations", self.mcts_params.get("num_iterations", 100)
                ),
                "ucb_c": dec_mcts_config.get(
                    "ucb_c", self.mcts_params.get("ucb1_c", 1.4)
                ),
                "discount_factor": dec_mcts_config.get(
                    "discount_factor", self.mcts_params.get("discount_factor", 0.95)
                ),
                "overlap_penalty_weight": dec_config.get("overlap_penalty_weight", 0.3),
                "d_uct_decay": d_uct_config.get("decay_factor", 0.9),
                "d_uct_threshold": d_uct_config.get("threshold_sec", 2.0),
                "parallel": dec_mcts_config.get(
                    "parallel", self.mcts_params.get("parallel", 1)
                ),
                "timeout": dec_mcts_config.get(
                    "timeout", self.mcts_params.get("timeout", 5.0)
                ),
            }

            self._dec_mcts_planner = DecMCTSPlanner(
                agent_id=self.agent_id,
                camera=self.uav,
                grid_info=self.uav.grid,
                conf_dict=self.conf_dict,
                config=config,
                seed=self.seed,
            )
            self._dec_mcts_coordinator = dec_mcts_coordinator

            if self.debug_logs:
                print(f"\n[DEC-MCTS] Agent {self.agent_id} initialized")
                print(
                    f"  Horizon: {config['horizon']}, Iterations: {config['iterations']}"
                )
                print(
                    f"  D-UCT decay: {config['d_uct_decay']}, threshold: {config['d_uct_threshold']}s\n"
                )

        planner = self._dec_mcts_planner

        # Update state
        uav_pos = self.uav.get_x()
        planner.update_state(
            position=uav_pos.position,
            altitude=uav_pos.altitude,
            belief=self.M.copy(),
        )

        # Get teammate intents
        if self._dec_mcts_coordinator is not None:
            teammate_intents = self._dec_mcts_coordinator.get_teammate_intents(
                self.agent_id
            )
            planner.update_teammate_intents(teammate_intents)
        else:
            teammate_intents = {}

        # Run planning
        intent = planner.plan()

        # Share intent with coordinator
        if self._dec_mcts_coordinator is not None:
            self._dec_mcts_coordinator.share_intent(intent)

        # Get stats for timing
        stats = planner.get_statistics()

        # Build action scores from MCTS values
        action_scores = planner.get_action_values()

        # Add timing information from planner stats
        if "last_planning_time_ms" in stats:
            action_scores["_timing_dec_mcts_ms"] = stats["last_planning_time_ms"]
        if "last_start_ms" in stats:
            action_scores["_timing_dec_mcts_start_ms"] = stats["last_start_ms"]
            action_scores["_timing_dec_mcts_end_ms"] = stats["last_end_ms"]

        action = intent.action_sequence[0] if intent.action_sequence else "hover"
        return action, action_scores

    def hierarchical_dec_mcts_decision(self, visited_x, use_mcts_llp=False):
        """
        Run Multi-Horizon Dec-MCTS planning with shared beliefs and intents.

        Implements the full MH Dec-MCTS framework from the Multi-Horizon paper:
        - Two-level planning (LLP + HLP) per agent
        - Asynchronous intent sharing between agents
        - Reward decomposition: g = g1(LL intents) + g2(all intents)

        Args:
            visited_x: List of visited positions
            use_mcts_llp: If True, use MCTS tree search for LLP; if False, use random rollouts

        Returns:
            Tuple of (selected_action, action_scores)
        """
        from hierarchical_dec_mcts import (
            HierarchicalDecMCTSPlanner,
            IntentBus,
            create_hierarchical_planner,
        )

        # Create or get hierarchical planner
        if not hasattr(self, "_hierarchical_planner"):
            # Check if coordinator has a shared intent bus
            intent_bus = None
            if self.coordinator is not None:
                if not hasattr(self.coordinator, "_intent_bus"):
                    # Create shared intent bus on coordinator
                    num_agents = getattr(self.coordinator, "num_agents", 1)
                    self.coordinator._intent_bus = IntentBus(num_agents=num_agents)
                intent_bus = self.coordinator._intent_bus
            else:
                # Single agent mode - create local intent bus
                intent_bus = IntentBus(num_agents=1)

            # Extract config - prefer hierarchical_dec_mcts section, fallback to mcts_params
            hier_config = self.experiment_config.get("hierarchical_dec_mcts", {})
            horizon_weights = self.mcts_params.get("horizon_weights", {})

            config = {
                "llp_horizon": hier_config.get(
                    "llp_horizon", horizon_weights.get("short_horizon_depth", 5)
                ),
                "llp_iterations": hier_config.get(
                    "llp_iterations", self.mcts_params.get("num_iterations", 50)
                ),
                "llp_discount_factor": hier_config.get("llp_discount_factor", 0.95),
                "hlp_horizon": hier_config.get(
                    "hlp_horizon", horizon_weights.get("long_horizon_depth", 3) // 5
                ),  # Regions, not steps
                "hlp_iterations": hier_config.get("hlp_iterations", 30),
                "tile_size": hier_config.get(
                    "tile_size",
                    hier_config.get(
                        "hlp_tile_size", horizon_weights.get("tile_size", [100, 100])
                    ),
                ),
                "hlp_replan_interval": hier_config.get("hlp_replan_interval", 1.0),
                "llp_ucb_c": hier_config.get("llp_ucb_c", 1.41),
                "hlp_ucb_c": hier_config.get("hlp_ucb_c", 1.0),
                "use_mcts_llp": use_mcts_llp,
                "use_g2": hier_config.get("use_g2", False),
                "g2_mode": hier_config.get("g2_mode", "hl_aware"),
            }

            num_agents = 1
            if self.coordinator is not None:
                num_agents = getattr(self.coordinator, "num_agents", 1)

            self._hierarchical_planner = create_hierarchical_planner(
                agent_id=self.agent_id,
                num_agents=num_agents,
                camera=self.uav,
                grid_info=self.uav.grid,
                intent_bus=intent_bus,
                config=config,
                conf_dict=self.conf_dict,
                seed=self.seed,
            )

            llp_mode = "MCTS tree search" if use_mcts_llp else "random rollouts"
            if self.debug_logs:
                print(f"\n[HIERARCHICAL DEC-MCTS] Agent {self.agent_id} initialized")
                print(
                    f"  LLP: {llp_mode}, horizon: {config['llp_horizon']}, iterations: {config['llp_iterations']}"
                )
                print(
                    f"  HLP: MCTS, horizon: {config['hlp_horizon']}, tile_size: {config['tile_size']}\n"
                )

        planner = self._hierarchical_planner

        # Update state from belief and position
        uav_pos = self.uav.get_x()
        planner.update_state(
            position=uav_pos.position,
            altitude=uav_pos.altitude,
            belief=self.M.copy(),
        )

        # Run hierarchical planning (includes intent sharing)
        action, metrics = planner.plan()

        # Build action scores from LL intent
        action_scores = {}
        ll_intent = metrics.get("ll_intent")
        if ll_intent and ll_intent.action_sequence:
            for i, act in enumerate(ll_intent.action_sequence):
                if i < len(ll_intent.ig_sequence):
                    action_scores[act] = ll_intent.ig_sequence[i]

        # Ensure current action has a score
        if action not in action_scores:
            action_scores[action] = metrics.get("expected_ig", 0.0)

        # Add timing breakdown from metrics
        if "hlp_time_ms" in metrics:
            action_scores["_timing_hlp_ms"] = metrics["hlp_time_ms"]
            action_scores["_timing_llp_ms"] = metrics["llp_time_ms"]
            action_scores["_timing_hlp_start_ms"] = metrics.get("hlp_start_ms")
            action_scores["_timing_hlp_end_ms"] = metrics.get("hlp_end_ms")
            action_scores["_timing_llp_start_ms"] = metrics.get("llp_start_ms")
            action_scores["_timing_llp_end_ms"] = metrics.get("llp_end_ms")
            action_scores["_timing_hlp_replanned"] = (
                1.0 if metrics.get("hlp_replanned", False) else 0.0
            )

        return action, action_scores

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
            seed=self.seed,
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

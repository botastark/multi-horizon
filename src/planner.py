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
    ):
        # Initialize belief map (each cell has a default probability of 0.5) and set UAV planning parameters
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None
        self.agent_id = agent_id
        self.coordinator = coordinator
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
        if self.strategy == "dec_mcts" and hasattr(self, "_dec_mcts_planner"):
            stats = self._dec_mcts_planner.get_statistics()
            print(f"\n[DEC-MCTS] Final Stats for Agent {self.agent_id}:")
            print(f"  Plans generated: {stats['plans_generated']}")
            print(f"  Total iterations: {stats['total_iterations']}")
            print(f"  Avg planning time: {stats['avg_planning_time']*1000:.1f}ms")
            print(f"  Intent updates received: {stats['intent_updates_received']}\n")

        # Log Multi-Horizon Dec-MCTS stats
        if self.strategy == "hierarchical_dec_mcts" and hasattr(
            self, "_hierarchical_planner"
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
        if self.strategy == "greedy_ig" and hasattr(self, "_greedy_ig_planner"):
            stats = self._greedy_ig_planner.get_statistics()
            mode = "IGd" if self._greedy_ig_planner.enable_discounting else "IG"
            print(f"\n[GREEDY {mode}] Final Stats for Agent {self.agent_id}:")
            print(f"  Plans generated: {stats['plans_generated']}")
            print(f"  Total IG: {stats['total_ig']:.4f}")
            if self._greedy_ig_planner.enable_discounting:
                print(f"  Total IGd (discounted): {stats['total_igd']:.4f}")
            print(f"  Intent updates received: {stats['intent_updates_received']}\n")

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
            # Try exact lookup first (keys are rounded to 2 decimals).
            key = np.round(x_future.altitude, decimals=2)
            if key in self.conf_dict:
                s0, s1 = self.conf_dict[key]
            else:
                # Fallback: find nearest altitude key available in conf_dict
                try:
                    keys = np.array(list(self.conf_dict.keys()), dtype=float)
                    idx = np.argmin(np.abs(keys - x_future.altitude))
                    nearest = keys[idx]
                    s0, s1 = self.conf_dict[nearest]
                except Exception:
                    s0, s1 = sigma, sigma
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
                self.sweep_direction = self.rng.choice(["LeftRight", "BackFront"])

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
        - Discount factor α_ij = 1 - IoU(fp(x_i), fp(x_j))
        - Discounted IG: IG_a^d = IG_a × Π_{j ∈ neighbors} α_ij

        Returns:
            Tuple of (selected_action, action_scores_dict)
        """
        from greedy_ig_planner import (
            GreedyIGPlanner,
            GreedyIGCoordinator,
            log_greedy_ig_decision,
            create_greedy_ig_planner,
        )

        # Create or get greedy IG planner
        if not hasattr(self, "_greedy_ig_planner"):
            # Extract config - paper approach uses 0.0 for penalties
            greedy_config = (
                self.conf_dict.get("greedy_ig", {}) if self.conf_dict else {}
            )

            # Auto-detect IGd mode from coordinator's mode attribute
            enable_discounting = greedy_config.get("enable_discounting", False)
            if self.coordinator is not None and hasattr(self.coordinator, "mode"):
                # If mode contains "IGd", enable discounting
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
            )

            # Create or get coordinator
            if self.coordinator is not None:
                if not hasattr(self.coordinator, "_greedy_ig_coordinator"):
                    num_agents = getattr(self.coordinator, "num_agents", 1)
                    self.coordinator._greedy_ig_coordinator = GreedyIGCoordinator(
                        num_agents=num_agents
                    )
                self._greedy_ig_coordinator = self.coordinator._greedy_ig_coordinator
            else:
                self._greedy_ig_coordinator = None

            mode = (
                "IGd (footprint discounting)" if config["enable_discounting"] else "IG"
            )
            print(f"\n[GREEDY {mode}] Agent {self.agent_id} initialized")
            print(f"  Paper approach: pure belief-based IG (no penalties)\n")

        planner = self._greedy_ig_planner

        # Update belief - this should be the FUSED belief from coordinator
        planner.update_belief(self.M.copy())

        # Get teammate intents for discounting (if enabled)
        if self._greedy_ig_coordinator is not None:
            teammate_intents = self._greedy_ig_coordinator.get_teammate_intents(
                self.agent_id
            )
            planner.update_teammate_intents(teammate_intents)
        else:
            teammate_intents = {}

        # Get current position
        uav_pos = self.uav.get_x()

        # Run planning (pure IG or IGd with discounting)
        intent = planner.plan(
            current_position=uav_pos.position,
            current_altitude=uav_pos.altitude,
        )

        # Share intent with coordinator
        if self._greedy_ig_coordinator is not None:
            self._greedy_ig_coordinator.share_intent(intent)

        # Log decision
        log_greedy_ig_decision(
            agent_id=self.agent_id,
            step=planner._stats["plans_generated"],
            raw_ig_scores=planner._raw_ig_scores,
            overlap_penalties=planner._overlap_penalties,
            final_scores=planner._action_scores,
            selected_action=intent.action,
            intents_received={"teammates": list(teammate_intents.keys())},
            discount_factors=(
                planner._discount_factors if planner.enable_discounting else None
            ),
        )

        # Get action and scores
        action = intent.action
        action_scores = planner.get_action_scores()

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
        elif self.strategy == "hierarchical_dec_mcts":
            # Use Multi-Horizon Dec-MCTS with LLP/HLP hierarchy
            return self.hierarchical_dec_mcts_decision(visited_x)
        elif self.strategy == "ig":
            # Legacy single-agent IG (backward compatibility)compatibility
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
            log_dec_mcts_decision,
            setup_dec_mcts_logger,
        )

        # Initialize logger on first call
        if not hasattr(self, "_dec_mcts_logger_initialized"):
            exp_name = getattr(self, "experiment_name", None)
            log_dir = getattr(self, "log_dir", None)
            import os

            # Prefer placing logs under the trial folder if an experiment
            # name is available. Fall back to legacy `logs/dec_mcts` only
            # when neither an explicit `log_dir` nor `experiment_name`
            # is provided.
            if log_dir is None:
                if exp_name:
                    log_dir = os.path.join("trials", exp_name, "logs")
                else:
                    # Avoid creating top-level `logs/` directories for legacy
                    # dec_mcts logging; place them under `trials/logs/dec_mcts`.
                    log_dir = os.path.join("trials", "logs", "dec_mcts")

            os.makedirs(log_dir, exist_ok=True)
            log_file = setup_dec_mcts_logger(log_dir=log_dir, experiment_name=exp_name)
            print(f"\n[DEC-MCTS] Logging to: {log_file}\n")
            self._dec_mcts_logger_initialized = True

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
            dec_mcts_config = (
                self.conf_dict.get("dec_mcts", {}) if self.conf_dict else {}
            )
            dec_config = (
                self.conf_dict.get("decentralized", {}) if self.conf_dict else {}
            )
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
            )
            self._dec_mcts_coordinator = dec_mcts_coordinator

            print(f"\n[DEC-MCTS] Agent {self.agent_id} initialized")
            print(f"  Horizon: {config['horizon']}, Iterations: {config['iterations']}")
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

        # Log decision
        stats = planner.get_statistics()
        log_dec_mcts_decision(
            agent_id=self.agent_id,
            step=stats["plans_generated"],
            mcts_action_values=planner.get_action_values(),
            mcts_action_visits=planner.get_action_visits(),
            selected_action=(
                intent.action_sequence[0] if intent.action_sequence else "hover"
            ),
            trajectory_summary={
                "length": len(intent.action_sequence),
                "total_ig": intent.total_expected_ig,
            },
            intents_received={"teammates": list(teammate_intents.keys())},
        )

        # Build action scores from MCTS values
        action_scores = planner.get_action_values()

        action = intent.action_sequence[0] if intent.action_sequence else "hover"
        return action, action_scores

    def hierarchical_dec_mcts_decision(self, visited_x):
        """
        Run Multi-Horizon Dec-MCTS planning with shared beliefs and intents.

        Implements the full MH Dec-MCTS framework from the Multi-Horizon paper:
        - Two-level planning (LLP + HLP) per agent
        - Asynchronous intent sharing between agents
        - Reward decomposition: g = g1(LL intents) + g2(all intents)

        Also known as "mh_dec_mcts" strategy.

        Args:
            visited_x: List of visited positions

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
            hier_config = (
                self.conf_dict.get("hierarchical_dec_mcts", {})
                if self.conf_dict
                else {}
            )
            horizon_weights = self.mcts_params.get("horizon_weights", {})

            config = {
                "llp_horizon": hier_config.get(
                    "llp_horizon", horizon_weights.get("short_horizon_depth", 5)
                ),
                "llp_iterations": hier_config.get(
                    "llp_iterations", self.mcts_params.get("num_iterations", 50)
                ),
                "hlp_horizon": hier_config.get(
                    "hlp_horizon", horizon_weights.get("long_horizon_depth", 3) // 5
                ),  # Regions, not steps
                "hlp_iterations": hier_config.get("hlp_iterations", 30),
                "tile_size": hier_config.get(
                    "tile_size", horizon_weights.get("tile_size", [100, 100])
                ),
                "hlp_replan_interval": hier_config.get("hlp_replan_interval", 1.0),
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
            )

            print(f"\n[HIERARCHICAL DEC-MCTS] Agent {self.agent_id} initialized")
            print(
                f"  LLP horizon: {config['llp_horizon']}, iterations: {config['llp_iterations']}"
            )
            print(
                f"  HLP horizon: {config['hlp_horizon']}, tile_size: {config['tile_size']}\n"
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

        # Log detailed planning decision
        from hierarchical_dec_mcts import log_planning_decision, log_intent_sharing

        stats = planner.get_statistics()
        log_planning_decision(
            agent_id=self.agent_id,
            step=stats["hierarchical"]["planning_cycles"],
            llp_action_scores=metrics.get("llp_action_scores", {}),
            hlp_region_scores=metrics.get("hlp_region_scores", {}),
            selected_action=action,
            target_region=metrics.get("target_region"),
            intents_received=metrics.get("intents_received", {}),
        )

        # Log intent sharing
        ll_intent = metrics.get("ll_intent")
        hl_intent = metrics.get("hl_intent")
        log_intent_sharing(
            agent_id=self.agent_id,
            ll_intent_summary={
                "actions": ll_intent.action_sequence if ll_intent else [],
                "total_ig": ll_intent.total_expected_ig if ll_intent else 0,
            },
            hl_intent_summary={
                "target_region": hl_intent.current_target_region if hl_intent else None,
                "region_sequence": hl_intent.region_sequence if hl_intent else [],
            },
        )

        # Build action scores from LL intent
        action_scores = {}
        if ll_intent and ll_intent.action_sequence:
            for i, act in enumerate(ll_intent.action_sequence):
                if i < len(ll_intent.ig_sequence):
                    action_scores[act] = ll_intent.ig_sequence[i]

        # Ensure current action has a score
        if action not in action_scores:
            action_scores[action] = metrics.get("expected_ig", 0.0)

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

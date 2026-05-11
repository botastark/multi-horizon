import numpy as np
import time
import csv
import os
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from helper import (
    H,
    compute_metrics,
    footprint_dict_to_bounds,
    footprint_iou,
    observed_m_ids,
    select_argmax_action,
    uav_position,
)
from experiment_utils import (
    extract_region_metadata,
    finalize_planners,
)
from viewer import plot_metrics, plot_terrain
from experiment_config import get_tqdm_file


class Simulator:
    def __init__(
        self,
        agents,
        map_obj,
        ground_truth_map,
        conf_dict,
        grid_info,
        n_steps,
        results_folder,
        corr_type,
        e_margin,
        grf_r,
        iter_idx,
        enable_stepwise_plotting,
        enable_logging,
        action_strategy,
        coordinator=None,
        multi_agent_logger=None,
        run_id=None,
        debug_logs=False,
    ):
        self.agents = agents
        self.map_obj = map_obj
        self.ground_truth_map = ground_truth_map
        self.conf_dict = conf_dict
        self.grid_info = grid_info
        self.n_steps = n_steps
        self.results_folder = results_folder
        self.corr_type = corr_type
        self.e_margin = e_margin
        self.grf_r = grf_r
        self.iter_idx = iter_idx
        self.enable_stepwise_plotting = enable_stepwise_plotting
        self.enable_logging = enable_logging
        self.action_strategy = action_strategy
        self.debug_logs = debug_logs
        self.coordinator = coordinator
        self.multi_agent_logger = multi_agent_logger
        self.run_id = run_id
        self.run_number = iter_idx

        self.fused_entropy_history = []

        # Timing statistics aggregation (Priority 1 metrics)
        self.timing_stats = {
            "all_records": [],  # Collect all timing records across agents
            "by_planner": {},  # Aggregate by planner type
            "outliers": [],  # Records with latency > mean + 3*std
        }

        # Open timestamp CSV files for direct writing per step
        if self.multi_agent_logger is not None:
            log_folder = os.path.dirname(self.multi_agent_logger.path)
        else:
            log_folder = os.path.join(results_folder, "txt")
            os.makedirs(log_folder, exist_ok=True)

        # Determine file mode: write header on first run, append on subsequent runs
        timestamp_path = os.path.join(log_folder, "timestamps.csv")
        file_exists = os.path.exists(timestamp_path)
        file_mode = "a" if file_exists else "w"

        # Create CSV writers for timestamps with run_id and run_number columns
        fieldnames = ["run_id", "run_number", "step", "agent_id", "start_ms", "end_ms"]
        self._timestamp_file = open(timestamp_path, file_mode, newline="")
        self._timestamp_writer = csv.DictWriter(
            self._timestamp_file, fieldnames=fieldnames
        )
        if not file_exists:
            self._timestamp_writer.writeheader()
        self._timestamp_file.flush()

        # For MH planners, create separate HLP and LLP timestamp files
        if action_strategy in (
            "hierarchical_dec_mcts",
            "mh_dec_mcts",
            "mh_dec_mcts_both",
            "mh_dec_mcts_full",
            "mh_dec_mcts_efficient",
        ):
            hlp_path = os.path.join(log_folder, "timestamps_hlp.csv")
            llp_path = os.path.join(log_folder, "timestamps_llp.csv")

            hlp_exists = os.path.exists(hlp_path)
            llp_exists = os.path.exists(llp_path)

            self._timestamp_hlp_file = open(
                hlp_path, "a" if hlp_exists else "w", newline=""
            )
            self._timestamp_hlp_writer = csv.DictWriter(
                self._timestamp_hlp_file, fieldnames=fieldnames
            )
            if not hlp_exists:
                self._timestamp_hlp_writer.writeheader()
            self._timestamp_hlp_file.flush()

            self._timestamp_llp_file = open(
                llp_path, "a" if llp_exists else "w", newline=""
            )
            self._timestamp_llp_writer = csv.DictWriter(
                self._timestamp_llp_file, fieldnames=fieldnames
            )
            if not llp_exists:
                self._timestamp_llp_writer.writeheader()
            self._timestamp_llp_file.flush()
        else:
            self._timestamp_hlp_file = None
            self._timestamp_llp_file = None

        self.fused_mse_history = []
        self.combined_coverage_history = []

        # Initialize display_belief with prior (0.5 probability for all cells)
        self.display_belief = np.full(
            (self.grid_info.shape[0], self.grid_info.shape[1], 2), 0.5
        )

    def run(self):
        """Run the simulation loop."""
        for step in tqdm(
            range(0, self.n_steps),
            desc="steps",
            position=3,
            leave=False,
            file=get_tqdm_file(),
        ):
            # 1. Compute metrics and log (BEFORE any observations, including step 0)
            # This ensures step 0 records the pure prior state (all beliefs = 0.5)
            self._compute_metrics(step)
            self._log_step(step)

            if step == 0 and self.debug_logs:
                print(f"Map Sum: {self.ground_truth_map.sum()}")

            if self.debug_logs:
                print(f"step {step}: Processing agent observations...")

            # 2. Observe at the current position, then fuse. This mirrors the
            # PA reference loop: metrics -> observe/update/fuse -> plan -> move.
            agent_observations = self._process_agent_observations()

            if self.coordinator is not None:
                dec_config = self.coordinator.config.get("decentralized", {})
                news_sharing = dec_config.get("news_sharing", True)
                self._perform_belief_fusion(
                    agent_observations,
                    step,
                    news_sharing=news_sharing,
                )

            self._plot_step(step, agent_observations)

            if self.debug_logs:
                print(f"step {step}: Planning...")

            # 3. Plan using the post-observation belief.
            self._select_agent_actions(step)

            # 4. Act (update positions). The new position will be observed at
            # the beginning of the next iteration, matching PA.
            self._update_agent_positions()

            if self.debug_logs:
                print(f"{'─'*40}")

        # Cleanup and finalize
        if self.coordinator is not None:
            self.coordinator.comm_bus.clear_all_queues()
            coord_stats = self.coordinator.get_statistics()
        else:
            coord_stats = {}

        finalize_planners(self.agents)
        if self.enable_logging and self.multi_agent_logger is not None:
            self.multi_agent_logger.close()

        # Print Priority 1 timing statistics (only if debug logging enabled)
        if self.debug_logs:
            self.print_timing_summary()
        # Persist detailed timing records for analysis (silent unless debug)
        self.write_timing_log_csv()

        return {
            "agents": [
                {
                    "agent_id": a["agent_id"],
                    "entropy": a["entropy"],
                    "mse": a["mse"],
                    "coverage": a["coverage"],
                    "height": a["height"],
                    "uav_positions": a["uav_positions"],
                    "actions": a["actions"],
                }
                for a in self.agents
            ],
            "coordination_stats": coord_stats,
        }

    def _process_agent_observations(self) -> Dict:
        """
        Phase 1: Process observations for all agents.
        """
        agent_observations = {}

        for agent in self.agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            occupancy_map = agent["occupancy_map"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]

            # Process coordination messages
            if self.coordinator is not None:
                self.coordinator.process_messages(agent_id)

            # Get observations with sensor model
            sigmas = None
            if self.conf_dict is not None:
                s0, s1 = self.conf_dict[np.round(uav_pos.altitude, decimals=2)]
                sigmas = [s0, s1]

            fp_vertices_ij, submap = self.map_obj.get_observations(uav_pos, sigmas)

            # Update local belief with OG (Bayesian update)
            occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)

            # Run local LBP propagation (decentralized per agent)
            lbp_iters = 1
            if self.coordinator is not None:
                lbp_iters = self.coordinator.lbp_iterations

            occupancy_map.propagate_messages(
                fp_vertices_ij,
                submap,
                max_iterations=lbp_iters,
                reset_msgs=True,
            )

            # Update agent's belief map (this is the LOCAL belief before any fusion)
            belief_map[:, :, 1] = occupancy_map.get_belief().copy()
            belief_map[:, :, 0] = 1 - belief_map[:, :, 1]
            agent["belief_map"] = belief_map

            # Store a copy of the pure local belief (before fusion with other agents)
            agent["local_belief_map"] = belief_map.copy()

            # Store observation info
            agent_observations[agent_id] = {
                "fp_ij": fp_vertices_ij,
                "submap": submap,
                "sigmas": sigmas,
                "camera": camera,
                "uav_pos": uav_pos,
            }

        return agent_observations

    def _perform_belief_fusion(
        self,
        agent_observations: Dict,
        step: int = 0,
        news_sharing: bool = True,
    ) -> None:
        """
        Phase 2: Perform synchronous belief fusion across agents.
        """
        # Access the MultiAgentMapper through coordinator
        mapper = self.coordinator.map

        # CRITICAL: Sync local beliefs from agent["occupancy_map"] to mapper.maps[]
        for agent in self.agents:
            agent_id = agent["agent_id"]
            local_belief = agent["occupancy_map"].get_belief()
            mapper.maps[agent_id].map_beliefs = local_belief.copy()

        # Only perform news fusion if news_sharing is enabled
        if news_sharing:
            # Build neighbor map
            neighbor_map = {}
            for agent_id in range(self.coordinator.num_agents):
                neighbor_map[agent_id] = self.coordinator.get_neighbors_in_range(
                    agent_id
                )

            # Debug: Print neighbor connectivity at step 0
            if self.debug_logs and step == 0:
                total_connections = sum(
                    len(neighbors) for neighbors in neighbor_map.values()
                )
                print(
                    f"[Step {step}] Neighbor connectivity: {total_connections} total connections"
                )
                for agent_id, neighbors in neighbor_map.items():
                    if neighbors:
                        print(
                            f"  Agent {agent_id}: {len(neighbors)} neighbors {neighbors}"
                        )

            # Phase 2: Update news and fuse (combined step)
            mapper.update_news_and_fuse(agent_observations, neighbor_map)

            # Log fusion stats periodically
            if self.debug_logs and (step == 0 or step % 20 == 0):
                fusion_stats = mapper.get_stats()
                print(
                    f"[Step {step}] Belief Fusion: news_fusions={fusion_stats.get('news_fusions', 0)}, "
                    f"mode={fusion_stats.get('news_mode', 'N/A')}"
                )
        else:
            # No news sharing: either IG (no position sharing) or IGd (discounted IG with IOU )
            if step == 0:
                mode_label = "IG"
                try:
                    dec_cfg = (
                        self.coordinator.config.get("decentralized", {})
                        if self.coordinator is not None
                        else {}
                    )
                    pos_share = dec_cfg.get("position_sharing", False)
                    if pos_share:
                        mode_label = "IGd"
                except Exception:
                    mode_label = "IG"
                print(f"[Step {step}] News sharing disabled ({mode_label} mode)")

        # Feed fused beliefs back to agents
        for agent in self.agents:
            agent_id = agent["agent_id"]
            fused_belief = mapper.get_agent_belief(agent_id)
            if fused_belief is not None:
                agent["belief_map"][:, :, 1] = fused_belief
                agent["belief_map"][:, :, 0] = 1 - fused_belief
                # CRITICAL: Update the local OccupancyMap with the fused belief
                # so that subsequent local updates start from the fused state.
                agent["occupancy_map"].map_beliefs = fused_belief.copy()

    def _select_agent_actions(self, step: int = 0) -> None:
        """
        Phase 3: Compute metrics, select actions for all agents.
        """
        if (
            self.action_strategy == "greedy_ig"
            and self.coordinator is not None
            and getattr(self.coordinator, "mode", None) in {"IG_BS", "IGd_BM"}
        ):
            self._select_greedy_baseline_actions(step)
            return

        for agent in self.agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            planner = agent["planner"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]

            # Compute metrics
            agent["observed_ids"].update(observed_m_ids(camera, uav_pos))
            entropy_val, mse_val, coverage_val = compute_metrics(
                self.ground_truth_map, belief_map, agent["observed_ids"], self.grid_info
            )
            agent["entropy"].append(entropy_val)
            agent["mse"].append(mse_val)
            agent["coverage"].append(coverage_val)
            agent["height"].append(uav_pos.altitude)

            # LLP/Planning Blocking Latency: Wall-clock time from action request to action ready
            # This includes ALL overhead:
            #   - Reading teammate intents/states from coordinator
            #   - Reading HLP guidance (for hierarchical planners)
            #   - All planner-specific computation (MCTS, IG, etc.)
            #   - Broadcasting intents back to coordinator
            #   - Returning action and metrics
            # This is the critical real-time metric for onboard deployment.
            # Use high-resolution timer for accurate latency measurement
            llp_start_time = time.perf_counter()
            next_action, info_gain_action = planner.select_action(
                belief_map, agent["uav_positions"]
            )
            llp_end_time = time.perf_counter()
            llp_latency_ms = (llp_end_time - llp_start_time) * 1000.0

            agent["info_gain_action"] = info_gain_action

            # Initialize legacy arrays on first step
            if "planning_times" not in agent:
                agent["planning_times"] = []  # Legacy list format
                agent["hlp_times"] = []
                agent["llp_times"] = []
                agent["hlp_replans"] = []

            # Store legacy format (backward compatibility)
            agent["planning_times"].append(llp_latency_ms)

            # Write timestamps directly to CSV (main file for all planners)
            self._timestamp_writer.writerow(
                {
                    "run_id": self.run_id,
                    "run_number": self.run_number,
                    "step": step,
                    "agent_id": agent_id,
                    "start_ms": llp_start_time * 1000.0,
                    "end_ms": llp_end_time * 1000.0,
                }
            )
            self._timestamp_file.flush()

            # For MH planners, write separate HLP and LLP timestamps
            if "_timing_hlp_ms" in info_gain_action:
                hlp_start_ms = info_gain_action.get("_timing_hlp_start_ms")
                hlp_end_ms = info_gain_action.get("_timing_hlp_end_ms")
                llp_start_ms = info_gain_action.get("_timing_llp_start_ms")
                llp_end_ms = info_gain_action.get("_timing_llp_end_ms")

                # Write HLP timestamps
                if hlp_start_ms is not None and self._timestamp_hlp_writer is not None:
                    self._timestamp_hlp_writer.writerow(
                        {
                            "run_id": self.run_id,
                            "run_number": self.run_number,
                            "step": step,
                            "agent_id": agent_id,
                            "start_ms": hlp_start_ms,
                            "end_ms": hlp_end_ms,
                        }
                    )
                    self._timestamp_hlp_file.flush()

                # Write LLP timestamps
                if llp_start_ms is not None and self._timestamp_llp_writer is not None:
                    self._timestamp_llp_writer.writerow(
                        {
                            "run_id": self.run_id,
                            "run_number": self.run_number,
                            "step": step,
                            "agent_id": agent_id,
                            "start_ms": llp_start_ms,
                            "end_ms": llp_end_ms,
                        }
                    )
                    self._timestamp_llp_file.flush()

                # Store in legacy arrays
                hlp_time = info_gain_action["_timing_hlp_ms"]
                llp_time = info_gain_action["_timing_llp_ms"]
                hlp_replanned = bool(info_gain_action["_timing_hlp_replanned"])
                agent["hlp_times"].append(hlp_time)
                agent["llp_times"].append(llp_time)
                agent["hlp_replans"].append(1.0 if hlp_replanned else 0.0)

            # Apply collision avoidance if enabled
            if self.coordinator and self.coordinator.collision_distance > 0:
                next_action = self._apply_collision_avoidance(
                    agent, next_action, info_gain_action, camera
                )

            # Log selected action
            if self.debug_logs:
                print(
                    f"[Agent {agent_id}] Step {step}: action={next_action} | "
                    f"pos=({uav_pos.position[0]:.1f}, {uav_pos.position[1]:.1f})"
                )

            # Store action for later position update
            agent["_next_action"] = next_action

    def _apply_collision_avoidance(
        self,
        agent: Dict,
        next_action,
        info_gain_action: Dict,
        camera,
    ) -> Any:
        """
        Apply collision avoidance penalty to action selection.
        """
        agent_id = agent["agent_id"]
        best_action = next_action
        best_score = float("-inf")

        # Get max IG for normalization
        valid_scores = [
            s for s in info_gain_action.values() if isinstance(s, (int, float))
        ]
        max_ig = max(max(valid_scores), 1.0) if valid_scores else 1.0

        config = self.coordinator.config if self.coordinator else {}

        for action, ig_score in info_gain_action.items():
            if not isinstance(ig_score, (int, float)):
                continue
            proposed_state = camera.x_future(action)
            if proposed_state is None:
                continue
            proposed_pos = uav_position(proposed_state)
            proposed_row, proposed_col = camera.convert_xy_ij(
                proposed_pos.position[0],
                proposed_pos.position[1],
                camera.grid.center,
            )

            # Get collision penalty (0-1 range)
            penalty = self.coordinator.get_collision_penalty(
                agent_id, (proposed_row, proposed_col)
            )

            # Adjusted score with collision penalty
            collision_weight = config.get("agents", {}).get(
                "collision_penalty_weight", 1.0
            )
            adjusted_score = ig_score - penalty * max_ig * collision_weight

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_action = action

        return best_action

    def _update_agent_positions(self) -> None:
        """
        Update all agent positions based on selected actions.
        """
        for agent in self.agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            next_action = agent.get("_next_action")

            if next_action is None:
                continue

            future_state = camera.x_future(next_action)

            # Update UAV position
            uav_pos = uav_position(future_state)
            agent["actions"].append(next_action)
            agent["uav_positions"].append(uav_pos)
            agent["uav_pos"] = uav_pos
            camera.set_altitude(uav_pos.altitude)
            camera.set_position(uav_pos.position)

            # Update coordinator with new position
            if self.coordinator:
                self.coordinator.update_agent_state(
                    agent_id=agent_id,
                    position=(uav_pos.position[0], uav_pos.position[1]),
                    altitude=uav_pos.altitude,
                )

            # Clean up temporary action
            agent.pop("_next_action", None)

    def _greedy_ig_planner_for_agent(self, agent, enable_discounting=False):
        planner = agent["planner"]
        if hasattr(planner, "_greedy_ig_planner"):
            return planner._greedy_ig_planner

        from greedy_ig_planner import create_greedy_ig_planner

        planner._greedy_ig_planner = create_greedy_ig_planner(
            agent_id=agent["agent_id"],
            camera=agent["camera"],
            grid_info=self.grid_info,
            conf_dict=self.conf_dict,
            config={
                "intent_discount": 0.0,
                "overlap_penalty_weight": 0.0,
                "enable_discounting": enable_discounting,
            },
            seed=0,
        )
        return planner._greedy_ig_planner

    def _greedy_ig_selfish_data(self):
        actions = []
        data = []
        enable_discounting = getattr(self.coordinator, "mode", None) == "IGd_BM"

        for agent in self.agents:
            camera = agent["camera"]
            uav_pos = agent["uav_pos"]
            greedy_planner = self._greedy_ig_planner_for_agent(
                agent,
                enable_discounting=enable_discounting,
            )
            greedy_planner.update_belief(agent["belief_map"])
            scored_actions = greedy_planner.score_admissible_actions(
                uav_pos.position,
                uav_pos.altitude,
            )
            admissible_action_to_ig = {
                action: [action_data["ig"]]
                for action, action_data in scored_actions.items()
            }
            admissible_action_to_fp = {
                action: action_data["footprint_ij"]
                for action, action_data in scored_actions.items()
            }

            selected = select_argmax_action(camera.rng, admissible_action_to_ig)
            actions.append(selected)
            data.append(
                {
                    "admissible_action_to_IG": admissible_action_to_ig,
                    "admissible_action_to_fp_ij": admissible_action_to_fp,
                }
            )

        return actions, data

    def _select_greedy_baseline_actions(self, step: int = 0) -> None:
        for agent in self.agents:
            camera = agent["camera"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]
            agent["observed_ids"].update(observed_m_ids(camera, uav_pos))
            entropy_val, mse_val, coverage_val = compute_metrics(
                self.ground_truth_map, belief_map, agent["observed_ids"], self.grid_info
            )
            agent["entropy"].append(entropy_val)
            agent["mse"].append(mse_val)
            agent["coverage"].append(coverage_val)
            agent["height"].append(uav_pos.altitude)

        selfish_actions, data = self._greedy_ig_selfish_data()
        mode = getattr(self.coordinator, "mode", None)

        if mode == "IGd_BM" and len(self.agents) > 1:
            if not hasattr(self, "_baseline_agent_decision_order_rng"):
                self._baseline_agent_decision_order_rng = np.random.default_rng(17)

            predicted_states = np.array(
                [
                    [
                        agent["uav_pos"].position[0],
                        agent["uav_pos"].position[1],
                        agent["uav_pos"].altitude,
                    ]
                    for agent in self.agents
                ],
                dtype=float,
            )
            selected_actions = ["hover"] * len(self.agents)
            decision_order = self._baseline_agent_decision_order_rng.permutation(
                len(self.agents)
            )
            h_disp = self.agents[0]["camera"].xy_step
            v_disp = self.agents[0]["camera"].h_step
            radius_multiplier = self.coordinator.config.get("decentralized", {}).get(
                "radius_multiplier", 5
            )
            max_distances = radius_multiplier * np.array(
                [h_disp, h_disp, v_disp], dtype=float
            )
            action_to_direction = {
                "up": np.array([0, 0, v_disp], dtype=float),
                "down": np.array([0, 0, -v_disp], dtype=float),
                "front": np.array([0, h_disp, 0], dtype=float),
                "back": np.array([0, -h_disp, 0], dtype=float),
                "right": np.array([h_disp, 0, 0], dtype=float),
                "left": np.array([-h_disp, 0, 0], dtype=float),
                "hover": np.array([0, 0, 0], dtype=float),
            }

            for agent_id in decision_order:
                check = np.prod(
                    np.where(
                        np.abs(predicted_states - predicted_states[agent_id])
                        > max_distances,
                        0,
                        1,
                    ),
                    axis=1,
                )
                neighbors = [idx for idx in np.flatnonzero(check) if idx != agent_id]
                for neighbor_id in neighbors:
                    neighbor_state = predicted_states[neighbor_id]
                    neighbor_fp = self.agents[neighbor_id][
                        "camera"
                    ].get_footprint_vertices_ij(
                        (neighbor_state[0], neighbor_state[1]),
                        neighbor_state[2],
                    )
                    for action, agent_fp in data[agent_id][
                        "admissible_action_to_fp_ij"
                    ].items():
                        overlap = footprint_iou(
                            footprint_dict_to_bounds(neighbor_fp),
                            footprint_dict_to_bounds(agent_fp),
                        )
                        data[agent_id]["admissible_action_to_IG"][action][
                            0
                        ] *= 1.0 - overlap

                action = select_argmax_action(
                    self.agents[agent_id]["camera"].rng,
                    data[agent_id]["admissible_action_to_IG"],
                )
                selected_actions[agent_id] = action
                predicted_states[agent_id, :] += action_to_direction[action]
        else:
            selected_actions = selfish_actions

        for agent, action, action_data in zip(self.agents, selected_actions, data):
            agent["info_gain_action"] = {
                k: v[0] for k, v in action_data["admissible_action_to_IG"].items()
            }
            agent["_next_action"] = action

    def _compute_metrics(self, step):
        combined_observed_ids = set()
        for agent in self.agents:
            combined_observed_ids.update(agent["observed_ids"])

        per_agent_mses = []
        for agent in self.agents:
            agent_belief = agent["belief_map"][:, :, 1]
            agent_mse = np.mean((self.ground_truth_map - agent_belief) ** 2)
            per_agent_mses.append(agent_mse)

        fused_local = np.mean(
            [agent["belief_map"][:, :, 1] for agent in self.agents], axis=0
        )
        ma_config = (
            self.coordinator.config.get("multi_agent", {})
            if self.coordinator is not None
            else {}
        )
        use_fused_mean_metrics = (
            self.action_strategy == "greedy_ig"
            and self.coordinator is not None
            and getattr(self.coordinator, "mode", None) in {"IG_BS", "IGd_BM"}
            and ma_config.get("metric_aggregation") == "fused_mean"
        )
        if ma_config.get("clip_metric_beliefs", not use_fused_mean_metrics):
            fused_local = np.clip(fused_local, 0.001, 0.999)

        if use_fused_mean_metrics:
            fused_mse_val = float(np.mean((self.ground_truth_map - fused_local) ** 2))
        else:
            fused_mse_val = np.mean(per_agent_mses)

        if self.debug_logs and (step == 0 or step % 10 == 0):
            flat = fused_local.ravel()
            print(
                f"[DEBUG] step={step} fused_mean={flat.mean():.4f} min={flat.min():.4f} max={flat.max():.4f}",
                flush=True,
            )
            print(
                f"[DEBUG] per_agent_mses={[f'{m:.4f}' for m in per_agent_mses]} avg={fused_mse_val:.4f}",
                flush=True,
            )

        self.display_belief = np.zeros(
            (self.grid_info.shape[0], self.grid_info.shape[1], 2)
        )
        self.display_belief[:, :, 1] = fused_local
        self.display_belief[:, :, 0] = 1 - fused_local

        if use_fused_mean_metrics:
            fused_entropy_val = float(np.sum(H(fused_local)))
            decided = (fused_local > 0.55) | (fused_local < 0.45)
            combined_coverage_val = float(decided.mean())
        else:
            fused_entropy_val, _, combined_coverage_val = compute_metrics(
                self.ground_truth_map,
                self.display_belief,
                combined_observed_ids,
                self.grid_info,
            )

        # Get action of first agent
        action = self.agents[0]["actions"][-1] if self.agents[0]["actions"] else "None"
        if self.debug_logs:
            print(
                f"Step {step}: Action={action}, Entropy={fused_entropy_val:.4f}, MSE={fused_mse_val:.4f}"
            )

        self.fused_entropy_history.append(fused_entropy_val)
        self.fused_mse_history.append(fused_mse_val)
        self.combined_coverage_history.append(combined_coverage_val)

    def _log_step(self, step):
        if self.enable_logging and self.multi_agent_logger is not None:
            # At step 0, before any actions, height/actions lists are empty
            heights = [
                (
                    agent["height"][-1]
                    if len(agent["height"]) > 0
                    else agent["uav_pos"].altitude
                )
                for agent in self.agents
            ]
            actions = [
                agent["actions"][-1] if len(agent["actions"]) > 0 else None
                for agent in self.agents
            ]
            igs = [
                (
                    agent["info_gain_action"].get(agent["actions"][-1])
                    if len(agent["actions"]) > 0
                    else None
                )
                for agent in self.agents
            ]

            # Get planning times
            planning_times = [
                (
                    agent["planning_times"][-1]
                    if "planning_times" in agent and len(agent["planning_times"]) > 0
                    else None
                )
                for agent in self.agents
            ]

            # Get HLP/LLP timing breakdown (only for MH-Dec-MCTS)
            hlp_times = [
                (
                    agent["hlp_times"][-1]
                    if "hlp_times" in agent and len(agent["hlp_times"]) > 0
                    else None
                )
                for agent in self.agents
            ]
            llp_times = [
                (
                    agent["llp_times"][-1]
                    if "llp_times" in agent and len(agent["llp_times"]) > 0
                    else None
                )
                for agent in self.agents
            ]
            hlp_replans = [
                (
                    agent["hlp_replans"][-1]
                    if "hlp_replans" in agent and len(agent["hlp_replans"]) > 0
                    else None
                )
                for agent in self.agents
            ]

            self.multi_agent_logger.log_multi_agent_data(
                entropy=self.fused_entropy_history[-1],
                mse=self.fused_mse_history[-1],
                coverage=self.combined_coverage_history[-1],
                heights=heights,
                actions=actions,
                igs=igs,
                step=step,
                planning_times=planning_times,
                hlp_times=hlp_times,
                llp_times=llp_times,
                hlp_replans=hlp_replans,
            )

    def _plot_step(self, step, agent_observations):
        if self.enable_stepwise_plotting:
            # For step 0, include the starting position in trajectory
            # For step > 0, exclude the last position (current) to show path up to previous step
            if step == 0:
                all_uav_positions = [agent["uav_positions"][:] for agent in self.agents]
            else:
                all_uav_positions = [
                    agent["uav_positions"][:-1] for agent in self.agents
                ]
            first_agent_obs = agent_observations.get(0, {})
            plot_submap = first_agent_obs.get("submap", np.array([]))
            plot_fp_ij = first_agent_obs.get("fp_ij", [[0, 0], [0, 0]])

            per_agent_data = []
            for agent in self.agents:
                agent_id = agent["agent_id"]
                obs_data = agent_observations.get(agent_id, {})
                agent_camera = agent["camera"]
                agent_uav_pos = obs_data.get("uav_pos", agent["uav_pos"])
                agent_planner = agent["planner"]
                agent_region_metadata, agent_selected_region, agent_region_scores = (
                    extract_region_metadata(agent_planner, self.action_strategy)
                )
                per_agent_data.append(
                    {
                        "agent_id": agent_id,
                        "submap": obs_data.get("submap", np.array([])),
                        "fp_ij": obs_data.get(
                            "fp_ij",
                            {"ul": (0, 0), "bl": (0, 0), "ur": (0, 0), "br": (0, 0)},
                        ),
                        "belief_map": agent.get(
                            "local_belief_map", agent["belief_map"]
                        ).copy(),
                        "obs_range": agent_camera.get_range(
                            position=(
                                agent_uav_pos.position
                                if hasattr(agent_uav_pos, "position")
                                else agent["uav_pos"].position
                            ),
                            altitude=(
                                agent_uav_pos.altitude
                                if hasattr(agent_uav_pos, "altitude")
                                else agent["uav_pos"].altitude
                            ),
                            index_form=False,
                        ),
                        "region_metadata": agent_region_metadata,
                        "selected_region_id": agent_selected_region,
                        "region_scores": agent_region_scores,
                    }
                )

            plot_terrain(
                f"{self.results_folder}/plots/{self.iter_idx}/steps/step_{step}.png",
                self.display_belief,
                self.grid_info,
                all_uav_positions,
                self.ground_truth_map,
                plot_submap,
                self.agents[0]["camera"].get_range(index_form=False),
                plot_fp_ij,
                self.agents[0]["camera"].get_hrange(),
                region_metadata=None,
                selected_region_id=None,
                region_scores=None,
                multi_agent=True,
                per_agent_data=per_agent_data,
            )

            per_agent_heights = [agent["height"][: step + 1] for agent in self.agents]
            plot_metrics(
                f"{self.results_folder}/plots/iter_{self.iter_idx}.png",
                self.fused_entropy_history,
                self.fused_mse_history,
                self.combined_coverage_history,
                per_agent_heights,
                height_range=self.agents[0]["camera"].get_hrange(),
            )

    def analyze_timing_statistics(self) -> Dict[str, Any]:
        """
        Analyze Priority 1 timing statistics after experiment completion.

        Returns comprehensive statistics including:
        - Mean, std, median, p95, max latency per planner
        - Outlier detection (> mean + 3*std)
        - For MH planners: separate stats by HLP replan status
        - First step vs subsequent steps comparison

        Returns:
            Dictionary with statistical analysis results
        """
        import numpy as np

        # Aggregate all timing records from agents
        all_records = []
        for agent in self.agents:
            if "timing_log" in agent:
                all_records.extend(agent["timing_log"])

        if not all_records:
            if self.debug_logs:
                print("[WARNING] No timing records found")
            return {}

        self.timing_stats["all_records"] = all_records

        # Group by planner type
        by_planner = {}
        for record in all_records:
            planner_type = record.get("planner_type", "unknown")
            if planner_type not in by_planner:
                by_planner[planner_type] = []
            by_planner[planner_type].append(record)

        self.timing_stats["by_planner"] = by_planner

        # Compute statistics per planner
        results = {}

        for planner_type, records in by_planner.items():
            latencies = np.array([r["llp_latency_ms"] for r in records])

            # Overall statistics
            stats = {
                "count": len(latencies),
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            }

            # Outlier detection (> mean + 3*std)
            outlier_threshold = stats["mean_ms"] + 3 * stats["std_ms"]
            outliers = [r for r in records if r["llp_latency_ms"] > outlier_threshold]
            stats["outlier_count"] = len(outliers)
            stats["outlier_percentage"] = 100.0 * len(outliers) / len(latencies)

            # First step vs subsequent steps
            first_step_latencies = [
                r["llp_latency_ms"] for r in records if r["is_first_step"]
            ]
            other_latencies = [
                r["llp_latency_ms"] for r in records if not r["is_first_step"]
            ]

            if first_step_latencies:
                stats["first_step_mean_ms"] = float(np.mean(first_step_latencies))
                stats["first_step_max_ms"] = float(np.max(first_step_latencies))

            if other_latencies:
                stats["other_steps_mean_ms"] = float(np.mean(other_latencies))

            # MH-specific: separate by HLP replan status
            if any("hlp_replanned" in r for r in records):
                replan_records = [r for r in records if r.get("hlp_replanned", False)]
                cached_records = [
                    r for r in records if not r.get("hlp_replanned", False)
                ]

                if replan_records:
                    replan_latencies = [r["llp_latency_ms"] for r in replan_records]
                    stats["hlp_replan_count"] = len(replan_records)
                    stats["hlp_replan_frequency"] = len(replan_records) / len(records)
                    stats["hlp_replan_mean_ms"] = float(np.mean(replan_latencies))
                    stats["hlp_replan_max_ms"] = float(np.max(replan_latencies))

                if cached_records:
                    cached_latencies = [r["llp_latency_ms"] for r in cached_records]
                    stats["hlp_cached_mean_ms"] = float(np.mean(cached_latencies))
                    stats["hlp_cached_max_ms"] = float(np.max(cached_latencies))

                # Validation: replan should be slower
                if replan_records and cached_records:
                    stats["replan_slowdown_factor"] = (
                        stats["hlp_replan_mean_ms"] / stats["hlp_cached_mean_ms"]
                    )

                    if stats["replan_slowdown_factor"] < 1.2:
                        print(
                            f"[WARNING] {planner_type}: HLP replan doesn't significantly increase latency (factor: {stats['replan_slowdown_factor']:.2f}x)"
                        )

            # Store outlier records
            self.timing_stats["outliers"].extend(outliers)

            results[planner_type] = stats

        return results

    def print_timing_summary(self):
        """Print formatted summary of timing statistics."""
        print("\n" + "=" * 80)
        print("PRIORITY 1: LLP BLOCKING LATENCY STATISTICS")
        print("=" * 80)

        stats = self.analyze_timing_statistics()

        if not stats:
            print("No timing data available")
            return

        for planner_type, planner_stats in stats.items():
            print(f"\n{planner_type.upper()} Planner:")
            print(f"  Total measurements: {planner_stats['count']}")
            print(
                f"  Mean:   {planner_stats['mean_ms']:7.2f} ms  (± {planner_stats['std_ms']:.2f} ms)"
            )
            print(f"  Median: {planner_stats['median_ms']:7.2f} ms")
            print(f"  P95:    {planner_stats['p95_ms']:7.2f} ms")
            print(
                f"  Range:  [{planner_stats['min_ms']:.2f}, {planner_stats['max_ms']:.2f}] ms"
            )

            # Outliers
            if planner_stats["outlier_count"] > 0:
                print(
                    f"  Outliers: {planner_stats['outlier_count']} ({planner_stats['outlier_percentage']:.1f}%) > {planner_stats['mean_ms'] + 3*planner_stats['std_ms']:.2f} ms"
                )

            # First step analysis
            if "first_step_mean_ms" in planner_stats:
                print(
                    f"  First step: {planner_stats['first_step_mean_ms']:.2f} ms (max: {planner_stats['first_step_max_ms']:.2f} ms)"
                )
                if "other_steps_mean_ms" in planner_stats:
                    slowdown = (
                        planner_stats["first_step_mean_ms"]
                        / planner_stats["other_steps_mean_ms"]
                    )
                    print(f"  First step slowdown: {slowdown:.2f}x")

            # HLP replan analysis (MH planners)
            if "hlp_replan_frequency" in planner_stats:
                print(f"\n  HLP Replan Analysis:")
                print(
                    f"    Replan frequency: {planner_stats['hlp_replan_frequency']*100:.1f}% ({planner_stats['hlp_replan_count']} steps)"
                )
                if "hlp_replan_mean_ms" in planner_stats:
                    print(
                        f"    With replan:  {planner_stats['hlp_replan_mean_ms']:7.2f} ms (max: {planner_stats['hlp_replan_max_ms']:.2f} ms)"
                    )
                if "hlp_cached_mean_ms" in planner_stats:
                    print(
                        f"    Cached (LLP): {planner_stats['hlp_cached_mean_ms']:7.2f} ms (max: {planner_stats['hlp_cached_max_ms']:.2f} ms)"
                    )
                if "replan_slowdown_factor" in planner_stats:
                    print(
                        f"    Replan overhead: {planner_stats['replan_slowdown_factor']:.2f}x slower"
                    )

        # Overall outlier summary
        total_outliers = len(self.timing_stats["outliers"])
        if total_outliers > 0:
            print(f"\n  TOTAL OUTLIERS ACROSS ALL PLANNERS: {total_outliers}")
            print(f"  (Steps with latency > mean + 3*std)")

        print("\n" + "=" * 80)

    def write_timing_log_csv(self):
        """Close timestamp CSV files."""
        try:
            if hasattr(self, "_timestamp_file") and self._timestamp_file:
                self._timestamp_file.close()
                if self.debug_logs:
                    print(f"[INFO] Timestamps saved to: {self._timestamp_file.name}")

            if hasattr(self, "_timestamp_hlp_file") and self._timestamp_hlp_file:
                self._timestamp_hlp_file.close()
                if self.debug_logs:
                    print(
                        f"[INFO] HLP timestamps saved to: {self._timestamp_hlp_file.name}"
                    )

            if hasattr(self, "_timestamp_llp_file") and self._timestamp_llp_file:
                self._timestamp_llp_file.close()
                if self.debug_logs:
                    print(
                        f"[INFO] LLP timestamps saved to: {self._timestamp_llp_file.name}"
                    )
        except Exception as exc:
            if self.debug_logs:
                print(f"[ERROR] Failed to close timestamp files: {exc}")

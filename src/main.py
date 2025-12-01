import os
import json
import random
import numpy as np
from tqdm import tqdm
import argparse
import copy


from helper import (
    FastLogger,
    compute_metrics,
    observed_m_ids,
    uav_position,
)
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning

from uav_camera import Camera
from multi_agent_coordinator import (
    MultiAgentCoordinator,
    generate_multi_agent_starts,
)

# from new_camera import Camera  # Updated import for new camera model
from viewer import plot_metrics, plot_terrain, plot_terrain_2d

from helper import create_run_folder, make_param_tag
import matplotlib

matplotlib.use("Agg")


# -----------------------------------------------------------------------------
# Load Experiment Configuration from JSON File
# -----------------------------------------------------------------------------
def load_config(config_file):
    """Load experiment configuration from a JSON file and filter out comment keys."""
    with open(config_file, "r") as f:
        config = json.load(f)
    # Remove any keys starting with "_" (used for comments)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    return config


# -----------------------------------------------------------------------------
# Build Global Folder Paths from Config
# -----------------------------------------------------------------------------
def load_global_paths(config):
    """
    Build global path variables using the base 'project_path' directory provided
    in the config.
    """
    PROJECT_PATH = config["project_path"].rstrip("/")  # Ensure no trailing slash
    ANNOTATION_PATH = os.path.join(PROJECT_PATH, "data", "annotation.txt")
    ORTHOMAP_PATH = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
    TILE_PIXEL_PATH = os.path.join(PROJECT_PATH, "data", "tiles_to_pixels.txt")
    MODEL_PATH = os.path.join(
        PROJECT_PATH,
        "binary_classifier",
        "models",
        "best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth",
    )
    CACHE_DIR = os.path.join(PROJECT_PATH, "data", "predictions_cache")
    return (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    )


# -----------------------------------------------------------------------------
# Parse Command-Line Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active sensing experiments using a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main Experiment Code
# -----------------------------------------------------------------------------


def run_single_agent_experiment(
    config,
    grid_info,
    camera,
    map_obj,
    ground_truth_map,
    conf_dict,
    occupancy_map,
    planner,
    uav_pos,
    results_folder,
    corr_type,
    e_margin,
    grf_r,
    iter_idx,
    n_steps,
    ENABLE_STEPWISE_PLOTTING,
    ENABLE_LOGGING,
    mcts_params,
    action_strategy,
):
    """
    Run a single-agent experiment iteration.

    This is the original experiment loop extracted into a function for reuse.
    """
    # Initialize belief map with a uniform probability (0.5)
    belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)

    uav_positions, actions = [uav_pos], []
    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)

    observed_ids = set()
    entropy, mse, height, coverage = [], [], [], []

    logger = None
    if ENABLE_LOGGING:
        log_folder = os.path.join(results_folder, "txt")
        logger = FastLogger(
            log_folder,
            strategy=action_strategy,
            pairwise=corr_type,
            grid=grid_info,
            init_x=uav_pos,
            r=grf_r,
            n_agent=iter_idx,
            e=e_margin,
            conf_dict=conf_dict,
            header_extras=[("mcts_params", json.dumps(mcts_params, sort_keys=True))],
        )

    if ENABLE_STEPWISE_PLOTTING:
        os.makedirs(
            results_folder
            + f"/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter_idx}/steps/",
            exist_ok=True,
        )

    info_gain_action = {}

    for step in tqdm(range(0, n_steps), desc="steps", position=3, leave=False):
        sigmas = None
        if conf_dict is not None:
            s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
            sigmas = [s0, s1]

        fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)
        observed_field_range = camera.get_range(index_form=False)

        occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
        occupancy_map.propagate_messages(fp_vertices_ij, submap, max_iterations=1)

        belief_map[:, :, 1] = occupancy_map.get_belief().copy()
        belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

        observed_ids.update(observed_m_ids(camera, uav_pos))
        entropy_val, mse_val, coverage_val = compute_metrics(
            ground_truth_map, belief_map, observed_ids, grid_info
        )
        entropy.append(entropy_val)
        mse.append(mse_val)
        coverage.append(coverage_val)
        height.append(uav_pos.altitude)

        if ENABLE_LOGGING and logger is not None:
            logger.log_data(
                entropy[-1],
                mse[-1],
                height[-1],
                coverage[-1],
                step=step,
                action=actions[-1] if len(actions) > 0 else None,
                ig=info_gain_action.get(actions[-1]) if len(actions) > 0 else None,
            )

        if ENABLE_STEPWISE_PLOTTING:
            plot_metrics(
                f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/iter_{iter_idx}.png",
                entropy,
                mse,
                coverage,
                height,
                height_range=camera.get_hrange(),
            )

        next_action, info_gain_action = planner.select_action(belief_map, uav_positions)

        print(f"Step {step}: Selected action {next_action}")
        print(f"Current UAV position: {uav_pos}")
        igs_sorted = dict(
            sorted(info_gain_action.items(), key=lambda kv: kv[1], reverse=True)
        )
        for a, ig in igs_sorted.items():
            print(f"{a}\t - {ig:.4f}")
        print("________________________________")

        uav_pos = uav_position(camera.x_future(next_action))
        actions.append(next_action)
        uav_positions.append(uav_pos)
        camera.set_altitude(uav_pos.altitude)
        camera.set_position(uav_pos.position)

        if ENABLE_STEPWISE_PLOTTING:
            region_metadata = None
            selected_region_id = None
            region_scores = None

            if action_strategy == "dual_horizon" and hasattr(
                planner, "_dual_horizon_planner"
            ):
                if hasattr(planner._dual_horizon_planner, "current_region_metadata"):
                    region_metadata = (
                        planner._dual_horizon_planner.current_region_metadata
                    )
                    selected_region_id = getattr(
                        planner._dual_horizon_planner, "current_selected_region", None
                    )
                    region_scores = getattr(
                        planner._dual_horizon_planner, "current_region_scores", None
                    )
            elif action_strategy == "threaded_dual_horizon" and hasattr(
                planner, "_threaded_dual_horizon_planner"
            ):
                if hasattr(
                    planner._threaded_dual_horizon_planner, "current_region_metadata"
                ):
                    region_metadata = (
                        planner._threaded_dual_horizon_planner.current_region_metadata
                    )
                    selected_region_id = getattr(
                        planner._threaded_dual_horizon_planner,
                        "current_selected_region",
                        None,
                    )
                    region_scores = getattr(
                        planner._threaded_dual_horizon_planner,
                        "current_region_scores",
                        None,
                    )

            plot_terrain(
                f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter_idx}/steps/step_{step}.png",
                belief_map,
                grid_info,
                uav_positions[0:-1],
                ground_truth_map,
                submap,
                observed_field_range,
                fp_vertices_ij,
                camera.get_hrange(),
                region_metadata=region_metadata,
                selected_region_id=selected_region_id,
                region_scores=region_scores,
            )

    return {
        "entropy": entropy,
        "mse": mse,
        "coverage": coverage,
        "height": height,
        "uav_positions": uav_positions,
        "actions": actions,
    }


def run_multi_agent_experiment(
    config,
    grid_info,
    map_obj,
    ground_truth_map,
    conf_dict,
    results_folder,
    corr_type,
    e_margin,
    grf_r,
    iter_idx,
    n_steps,
    ENABLE_STEPWISE_PLOTTING,
    ENABLE_LOGGING,
    mcts_params,
    action_strategy,
    min_alt,
    overlap,
    optimal_alt,
    seed,
):
    """
    Run a multi-agent experiment iteration with decentralized coordination.

    Each agent has its own camera, planner, and tracks its own path.
    Agents coordinate through the MultiAgentCoordinator.
    """
    ma_config = config.get("multi_agent", {})
    num_agents = ma_config.get("num_agents", 1)
    start_position = config.get("start_position", "corner")

    print(f"\n{'='*60}")
    print(f"MULTI-AGENT EXPERIMENT: {num_agents} agents")
    print(f"{'='*60}\n")

    # Create RNG
    rng = np.random.default_rng(seed)

    # Initialize coordinator
    coordinator = MultiAgentCoordinator(
        num_agents=num_agents,
        grid_shape=grid_info.shape,
        config=config,
    )

    # Generate start positions for all agents
    start_positions = generate_multi_agent_starts(
        num_agents=num_agents,
        grid_info=grid_info,
        start_position=start_position,
        min_distance=10.0,
    )

    # Initialize per-agent state
    agents = []
    for agent_id in range(num_agents):
        # Create camera for this agent
        camera = Camera(
            grid_info,
            60,
            rng=np.random.default_rng(seed + agent_id),
            camera_altitude=min_alt,
            f_overlap=overlap,
            s_overlap=overlap,
        )

        # Create planner for this agent (with coordinator for decentralized HLP)
        agent_planner = planning(
            grid_info,
            camera,
            action_strategy,
            conf_dict=conf_dict,
            optimal_alt=optimal_alt,
            mcts_params=mcts_params,
            agent_id=agent_id,
            coordinator=coordinator,
        )

        # Initialize UAV position
        start_pos = start_positions[agent_id]
        uav_pos = uav_position((start_pos, camera.get_hrange()[0]))
        camera.set_altitude(uav_pos.altitude)
        camera.set_position(uav_pos.position)

        # Create occupancy map for this agent
        occupancy_map = OM(
            grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
        )

        agent_state = {
            "agent_id": agent_id,
            "camera": camera,
            "planner": agent_planner,
            "occupancy_map": occupancy_map,
            "uav_pos": uav_pos,
            "uav_positions": [uav_pos],
            "actions": [],
            "belief_map": np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5),
            "observed_ids": set(),
            "entropy": [],
            "mse": [],
            "coverage": [],
            "height": [],
            "info_gain_action": {},
        }
        agents.append(agent_state)

        # Update coordinator with initial position
        current_row, current_col = camera.convert_xy_ij(
            uav_pos.position[0], uav_pos.position[1], camera.grid.center
        )
        coordinator.update_agent_state(
            agent_id=agent_id,
            position=(current_row, current_col),
            altitude=uav_pos.altitude,
        )

        print(
            f"Agent {agent_id}: Start position {start_pos}, grid ({current_row:.0f}, {current_col:.0f})"
        )

    # Setup logging
    loggers = {}
    if ENABLE_LOGGING:
        log_folder = os.path.join(results_folder, "txt")
        for agent in agents:
            agent_id = agent["agent_id"]
            loggers[agent_id] = FastLogger(
                log_folder,
                strategy=action_strategy,
                pairwise=corr_type,
                grid=grid_info,
                init_x=agent["uav_pos"],
                r=grf_r,
                n_agent=f"{iter_idx}_agent{agent_id}",
                e=e_margin,
                conf_dict=conf_dict,
                header_extras=[
                    ("mcts_params", json.dumps(mcts_params, sort_keys=True)),
                    ("num_agents", str(num_agents)),
                    ("agent_id", str(agent_id)),
                ],
            )

    # Setup step-wise plotting directory
    if ENABLE_STEPWISE_PLOTTING:
        os.makedirs(
            results_folder
            + f"/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter_idx}/steps/",
            exist_ok=True,
        )

    # Main multi-agent loop
    for step in tqdm(range(0, n_steps), desc="steps", position=3, leave=False):
        # =====================================================================
        # PHASE 1: All agents observe and update local belief (SYNCHRONOUS)
        # Each agent:
        #   1. Gets observation from environment
        #   2. Updates local OG belief
        #   3. Runs local LBP propagation (decentralized within their view)
        # =====================================================================
        agent_observations = {}  # Store observations for Phase 2

        for agent in agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            occupancy_map = agent["occupancy_map"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]

            # Process coordination messages (if any from previous step)
            coordinator.process_messages(agent_id)

            # Get observations
            sigmas = None
            if conf_dict is not None:
                s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                sigmas = [s0, s1]

            fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

            # Update local belief with OG (Bayesian update)
            occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)

            # Run local LBP propagation (decentralized/async per agent)
            occupancy_map.propagate_messages(fp_vertices_ij, submap, max_iterations=1)

            # Update agent's belief map
            belief_map[:, :, 1] = occupancy_map.get_belief().copy()
            belief_map[:, :, 0] = 1 - belief_map[:, :, 1]
            agent["belief_map"] = belief_map

            # Store observation info for synchronous fusion
            agent_observations[agent_id] = {
                "fp_ij": fp_vertices_ij,
                "submap": submap,
                "sigmas": sigmas,
                "camera": camera,
                "uav_pos": uav_pos,
            }

        # =====================================================================
        # PHASE 2: Synchronous News Belief Update + Fusion (Paper-compliant)
        # Step A: ALL agents update their news beliefs from observations
        # Step B: ALL agents fuse with their neighbors
        # This ensures proper decentralized coordination without double-counting
        # =====================================================================
        if coordinator.lbp_fusion is not None:
            # Use the coordinator's synchronous fusion method
            # Step A: Update all news beliefs
            coordinator.update_all_news(agent_observations)

            # Step B: Fuse all agents with their neighbors
            coordinator.fuse_all_news()

            # NOTE: We keep using each agent's LOCAL occupancy_map belief for
            # planning and visualization. The coordinator fusion updates its
            # internal map_beliefs for multi-agent coordination/consensus,
            # but each agent's local belief (from occupancy_map) is more accurate
            # for that agent's own observations.
        else:
            # Fallback to simple belief sharing (weighted averaging)
            for agent in agents:
                agent_id = agent["agent_id"]
                camera = agent["camera"]
                uav_pos = agent["uav_pos"]
                belief_map = agent["belief_map"]

                if coordinator.should_coordinate(agent_id):
                    observed_mask = np.zeros(grid_info.shape, dtype=bool)
                    [[imin, imax], [jmin, jmax]] = camera.get_range(
                        position=uav_pos.position,
                        altitude=uav_pos.altitude,
                        index_form=True,
                    )
                    observed_mask[imin:imax, jmin:jmax] = True
                    coordinator.share_belief(agent_id, belief_map, observed_mask)

        # =====================================================================
        # PHASE 3: Compute metrics, select actions, and move (per-agent)
        # =====================================================================
        for agent in agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            planner = agent["planner"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]

            # Compute metrics
            agent["observed_ids"].update(observed_m_ids(camera, uav_pos))
            entropy_val, mse_val, coverage_val = compute_metrics(
                ground_truth_map, belief_map, agent["observed_ids"], grid_info
            )
            agent["entropy"].append(entropy_val)
            agent["mse"].append(mse_val)
            agent["coverage"].append(coverage_val)
            agent["height"].append(uav_pos.altitude)

            # Log
            if ENABLE_LOGGING and agent_id in loggers:
                loggers[agent_id].log_data(
                    agent["entropy"][-1],
                    agent["mse"][-1],
                    agent["height"][-1],
                    agent["coverage"][-1],
                    step=step,
                    action=agent["actions"][-1] if len(agent["actions"]) > 0 else None,
                    ig=(
                        agent["info_gain_action"].get(agent["actions"][-1])
                        if len(agent["actions"]) > 0
                        else None
                    ),
                )

            # Select action
            next_action, info_gain_action = planner.select_action(
                belief_map, agent["uav_positions"]
            )
            agent["info_gain_action"] = info_gain_action

            # Apply collision avoidance penalty
            if coordinator.collision_distance > 0:
                # Get proposed position for each action
                best_action = next_action
                best_score = float("-inf")

                for action, ig_score in info_gain_action.items():
                    proposed_state = camera.x_future(action)
                    if proposed_state is None:
                        continue
                    proposed_pos = uav_position(proposed_state)
                    proposed_row, proposed_col = camera.convert_xy_ij(
                        proposed_pos.position[0],
                        proposed_pos.position[1],
                        camera.grid.center,
                    )

                    # Get collision penalty
                    penalty = coordinator.get_collision_penalty(
                        agent_id, (proposed_row, proposed_col)
                    )

                    # Adjusted score = IG - collision_penalty
                    adjusted_score = ig_score - penalty * 0.5

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_action = action

                next_action = best_action

            print(
                f"[Agent {agent_id}] Step {step}: {next_action} | pos={uav_pos.position}"
            )

            # Update UAV position
            uav_pos = uav_position(camera.x_future(next_action))
            agent["actions"].append(next_action)
            agent["uav_positions"].append(uav_pos)
            agent["uav_pos"] = uav_pos
            camera.set_altitude(uav_pos.altitude)
            camera.set_position(uav_pos.position)

            # Update coordinator with new position
            current_row, current_col = camera.convert_xy_ij(
                uav_pos.position[0], uav_pos.position[1], camera.grid.center
            )
            coordinator.update_agent_state(
                agent_id=agent_id,
                position=(current_row, current_col),
                altitude=uav_pos.altitude,
                coverage=coverage_val,
            )

        # Stepwise plotting (combines all agents)
        if ENABLE_STEPWISE_PLOTTING:
            # Collect all agent paths
            all_uav_positions = []
            for agent in agents:
                all_uav_positions.append(agent["uav_positions"][:-1])

            # Compute fused belief from all agents' local beliefs (product-of-experts)
            # This combines each agent's occupancy_map belief
            all_beliefs = np.stack(
                [agent["belief_map"][:, :, 1] for agent in agents], axis=-1
            )
            prod_occupied = np.prod(all_beliefs, axis=-1)
            prod_free = np.prod(1.0 - all_beliefs, axis=-1)
            epsilon = 1e-20
            fused_local = prod_occupied / (prod_occupied + prod_free + epsilon)
            fused_local = np.clip(fused_local, 0.001, 0.999)

            display_belief = np.zeros((grid_info.shape[0], grid_info.shape[1], 2))
            display_belief[:, :, 1] = fused_local
            display_belief[:, :, 0] = 1 - fused_local

            # Get region metadata from first agent's planner
            region_metadata = None
            selected_region_id = None
            region_scores = None
            first_planner = agents[0]["planner"]

            if action_strategy == "threaded_dual_horizon" and hasattr(
                first_planner, "_threaded_dual_horizon_planner"
            ):
                if hasattr(
                    first_planner._threaded_dual_horizon_planner,
                    "current_region_metadata",
                ):
                    region_metadata = (
                        first_planner._threaded_dual_horizon_planner.current_region_metadata
                    )
                    selected_region_id = getattr(
                        first_planner._threaded_dual_horizon_planner,
                        "current_selected_region",
                        None,
                    )
                    region_scores = getattr(
                        first_planner._threaded_dual_horizon_planner,
                        "current_region_scores",
                        None,
                    )

            # Get observation data from first agent for plotting
            first_agent_obs = agent_observations.get(0, {})
            plot_submap = first_agent_obs.get("submap", np.array([]))
            plot_fp_ij = first_agent_obs.get("fp_ij", [[0, 0], [0, 0]])

            # Collect per-agent data for multi-agent visualization
            per_agent_data = []
            for agent in agents:
                agent_id = agent["agent_id"]
                obs_data = agent_observations.get(agent_id, {})
                agent_camera = agent["camera"]
                agent_uav_pos = obs_data.get("uav_pos", agent["uav_pos"])
                agent_planner = agent["planner"]

                # Get per-agent HLP region data
                agent_region_metadata = None
                agent_selected_region = None
                agent_region_scores = None

                if action_strategy == "dual_horizon" and hasattr(
                    agent_planner, "_dual_horizon_planner"
                ):
                    dhp = agent_planner._dual_horizon_planner
                    if hasattr(dhp, "current_region_metadata"):
                        agent_region_metadata = dhp.current_region_metadata
                        agent_selected_region = getattr(
                            dhp, "current_selected_region", None
                        )
                        agent_region_scores = getattr(
                            dhp, "current_region_scores", None
                        )
                elif action_strategy == "threaded_dual_horizon" and hasattr(
                    agent_planner, "_threaded_dual_horizon_planner"
                ):
                    tdhp = agent_planner._threaded_dual_horizon_planner
                    if hasattr(tdhp, "current_region_metadata"):
                        agent_region_metadata = tdhp.current_region_metadata
                        agent_selected_region = getattr(
                            tdhp, "current_selected_region", None
                        )
                        agent_region_scores = getattr(
                            tdhp, "current_region_scores", None
                        )

                per_agent_data.append(
                    {
                        "agent_id": agent_id,
                        "submap": obs_data.get("submap", np.array([])),
                        "fp_ij": obs_data.get(
                            "fp_ij",
                            {"ul": (0, 0), "bl": (0, 0), "ur": (0, 0), "br": (0, 0)},
                        ),
                        "belief_map": agent["belief_map"].copy(),
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

            # Plot with multi-agent paths (no region_metadata in main call - it's per-agent now)
            plot_terrain(
                f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter_idx}/steps/step_{step}.png",
                display_belief,
                grid_info,
                all_uav_positions,  # Pass list of paths
                ground_truth_map,
                plot_submap,
                agents[0]["camera"].get_range(index_form=False),
                plot_fp_ij,
                agents[0]["camera"].get_hrange(),
                region_metadata=None,  # Regions shown per-agent now
                selected_region_id=None,
                region_scores=None,
                multi_agent=True,  # New flag for multi-agent visualization
                per_agent_data=per_agent_data,  # Pass per-agent observation/belief/HLP data
            )

        print(f"--- Step {step} complete ---\n")

    # Finalize planners
    for agent in agents:
        if hasattr(agent["planner"], "finalize_episode"):
            agent["planner"].finalize_episode()

    # Print coordination statistics
    coord_stats = coordinator.get_statistics()
    print(f"\n{'='*60}")
    print("MULTI-AGENT COORDINATION STATISTICS")
    print(f"{'='*60}")
    print(f"Belief fusions: {coord_stats.get('belief_fusions', 0)}")
    print(f"Region allocations: {coord_stats.get('region_allocations', 0)}")
    print(f"Collision avoidances: {coord_stats.get('collision_avoidances', 0)}")
    print(f"Communication stats: {coord_stats.get('comm_bus', {})}")

    if "lbp_fusion" in coord_stats:
        print(f"LBP Fusion stats: {coord_stats['lbp_fusion']}")

    if "coverage_per_agent" in coord_stats:
        print(f"Coverage per agent: {coord_stats['coverage_per_agent']}")

    print(f"{'='*60}\n")

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
            for a in agents
        ],
        "coordination_stats": coord_stats,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Extract configuration parameters
    (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    ) = load_global_paths(config)
    # base_dir = create_run_folder(os.path.join(PROJECT_PATH, "results"))
    base_dir = os.path.join(PROJECT_PATH, "trials")
    run_base = f"{config['field_type'].lower()}_{config['start_position']}"
    if config.get("action_strategy") == "mcts" and config.get("params_in_path", True):
        run_base = run_base + "__" + make_param_tag(config.get("mcts_params", {}))
    results_folder = os.path.join(base_dir, run_base)

    ENABLE_STEPWISE_PLOTTING = config["enable_plotting"]
    ENABLE_LOGGING = config["enable_logging"]
    mcts_params = config.get("mcts_params", {})

    field_type = config["field_type"]
    start_position = config["start_position"]
    action_strategy = config["action_strategy"]
    correlation_types = config["correlation_types"]
    n_steps = config["n_steps"]
    iters = config["iters"]

    # Multi-agent configuration
    ma_config = config.get("multi_agent", {})
    num_agents = ma_config.get("num_agents", 1)

    if isinstance(iters, int):
        iters = [0, iters]
    error_margins = [None if e == "None" else e for e in config["error_margins"]]
    if action_strategy == "sweep":
        error_margins = [None]
        iters = [0, 1]

    # -----------------------------------------------------------------------------
    # Setup Grid and Field Parameters Based on Field Type
    # -----------------------------------------------------------------------------

    if field_type == "Ortomap":
        grf_r = "orto"
        min_alt = 19.5
        overlap = 0.8
        optimal_alt = min_alt

        class grid_info:
            x = 60
            y = 110
            length = 1
            shape = (int(y / length), int(x / length))
            center = True

        use_sensor_model = False
    else:
        grf_r = 4
        field_type = grf_r
        min_alt = None
        overlap = None
        optimal_alt = 21.5

        class grid_info:
            x = 50
            y = 50
            length = 0.125
            shape = (int(y / length), int(x / length))
            center = True

        use_sensor_model = True

    seed = 123
    rng = np.random.default_rng(seed)

    # Create initial camera (for single-agent or field initialization)
    camera1 = Camera(
        grid_info,
        60,
        rng=rng,
        camera_altitude=min_alt,
        f_overlap=overlap,
        s_overlap=overlap,
    )
    map_obj = Field(
        grid_info,
        field_type,
        sweep=action_strategy,
        h_range=camera1.get_hrange(),
        annotation_path=ANNOTATION_PATH,
        ortomap_path=ORTHOMAP_PATH,
        tile_pixel_path=TILE_PIXEL_PATH,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
    )

    # -----------------------------------------------------------------------------
    # Main Experiment Loop
    # -----------------------------------------------------------------------------

    for corr_type in tqdm(correlation_types, desc="Pairwise", position=0):
        for e_margin in tqdm(
            error_margins, desc=f"Error Margins (pairwise = {corr_type})", position=1
        ):
            for iter_idx in tqdm(
                range(iters[0], iters[-1]),
                desc=f"Iters (e={e_margin})",
                position=2,
                leave=False,
            ):
                map_obj.reset()
                ground_truth_map = map_obj.get_ground_truth()

                if e_margin is not None:
                    conf_dict = map_obj.init_s0_s1(
                        e=e_margin,
                        sensor=use_sensor_model,
                    )
                else:
                    conf_dict = None

                # Decide between single-agent and multi-agent experiment
                if num_agents > 1:
                    # Multi-agent experiment
                    result = run_multi_agent_experiment(
                        config=config,
                        grid_info=grid_info,
                        map_obj=map_obj,
                        ground_truth_map=ground_truth_map,
                        conf_dict=conf_dict,
                        results_folder=results_folder,
                        corr_type=corr_type,
                        e_margin=e_margin,
                        grf_r=grf_r,
                        iter_idx=iter_idx,
                        n_steps=n_steps,
                        ENABLE_STEPWISE_PLOTTING=ENABLE_STEPWISE_PLOTTING,
                        ENABLE_LOGGING=ENABLE_LOGGING,
                        mcts_params=mcts_params,
                        action_strategy=action_strategy,
                        min_alt=min_alt,
                        overlap=overlap,
                        optimal_alt=optimal_alt,
                        seed=seed,
                    )
                    print(f"Multi-agent experiment completed with {num_agents} agents")
                else:
                    # Single-agent experiment (original behavior)
                    occupancy_map = OM(
                        grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
                    )

                    planner = planning(
                        grid_info,
                        camera1,
                        action_strategy,
                        conf_dict=conf_dict,
                        optimal_alt=optimal_alt,
                        mcts_params=mcts_params,
                    )

                    # Select initial UAV starting position
                    if start_position == "edge":
                        real_border = [
                            (
                                -grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),
                            (
                                grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                grid_info.y / 2,
                            ),
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                -grid_info.y / 2,
                            ),
                        ]
                        start_pos = random.choice(real_border)
                    elif start_position == "corner":
                        start_pos = random.choice(
                            [
                                (-grid_info.x / 2, -grid_info.y / 2),
                                (-grid_info.x / 2, grid_info.y / 2),
                                (grid_info.x / 2, -grid_info.y / 2),
                                (grid_info.x / 2, grid_info.y / 2),
                            ]
                        )

                    uav_pos = uav_position((start_pos, camera1.get_hrange()[0]))

                    result = run_single_agent_experiment(
                        config=config,
                        grid_info=grid_info,
                        camera=camera1,
                        map_obj=map_obj,
                        ground_truth_map=ground_truth_map,
                        conf_dict=conf_dict,
                        occupancy_map=occupancy_map,
                        planner=planner,
                        uav_pos=uav_pos,
                        results_folder=results_folder,
                        corr_type=corr_type,
                        e_margin=e_margin,
                        grf_r=grf_r,
                        iter_idx=iter_idx,
                        n_steps=n_steps,
                        ENABLE_STEPWISE_PLOTTING=ENABLE_STEPWISE_PLOTTING,
                        ENABLE_LOGGING=ENABLE_LOGGING,
                        mcts_params=mcts_params,
                        action_strategy=action_strategy,
                    )

                    # Finalize planner
                    if hasattr(planner, "finalize_episode"):
                        planner.finalize_episode()


if __name__ == "__main__":
    main()

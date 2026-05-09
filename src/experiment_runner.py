import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from helper import (
    FastLogger,
    uav_position,
)
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera
from multi_agent_coordinator import MultiAgentCoordinator
from experiment_utils import initialize_agent
from simulator import Simulator


def _build_logged_hyperparams(config, action_strategy, mcts_params):
    """Build a flat, method-specific hyperparameter dictionary for run.log headers."""
    hyperparams = {}

    # Keep legacy mcts_params fields when present
    if isinstance(mcts_params, dict):
        for key in ["horizon", "iterations", "ucb_c", "discount_factor", "timeout"]:
            if key in mcts_params:
                hyperparams[key] = mcts_params[key]

    decentralized_cfg = config.get("decentralized", {})
    for key in ["overlap_penalty_weight", "radius_multiplier"]:
        if key in decentralized_cfg:
            hyperparams[key] = decentralized_cfg[key]

    if action_strategy == "greedy_ig":
        greedy_cfg = config.get("greedy_ig", {})
        if "overlap_penalty_weight" in greedy_cfg:
            hyperparams["overlap_penalty_weight"] = greedy_cfg["overlap_penalty_weight"]

    elif action_strategy == "dec_mcts":
        dec_cfg = config.get("dec_mcts", {})
        for key in ["horizon", "iterations", "ucb_c", "discount_factor", "timeout"]:
            if key in dec_cfg:
                hyperparams[key] = dec_cfg[key]

    elif action_strategy in [
        "hierarchical_dec_mcts",
        "mh_dec_mcts",
        "mh_dec_mcts_both",
        "mh_dec_mcts_full",
        "mh_dec_mcts_efficient",
    ]:
        hier_cfg = config.get("hierarchical_dec_mcts", {})
        llp_cfg = hier_cfg.get("llp", {})
        hlp_cfg = hier_cfg.get("hlp", {})

        if llp_cfg:
            for key in ["horizon", "iterations", "ucb_c", "discount_factor"]:
                if key in llp_cfg:
                    hyperparams[f"llp_{key}"] = llp_cfg[key]
        if hlp_cfg:
            for key in ["horizon", "iterations", "ucb_c", "discount_factor"]:
                if key in hlp_cfg:
                    hyperparams[f"hlp_{key}"] = hlp_cfg[key]
            if "replan_interval" in hlp_cfg:
                hyperparams["hlp_replan_interval"] = hlp_cfg["replan_interval"]

        # flattened compatibility keys
        for key, value in hier_cfg.items():
            if key.startswith("llp_") or key.startswith("hlp_"):
                hyperparams[key] = value

        if "use_mcts_llp" in hier_cfg:
            hyperparams["use_mcts_llp"] = hier_cfg["use_mcts_llp"]

    return hyperparams


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
    """Run a single-agent experiment using the Simulator class."""
    # Ensure camera state matches provided uav_pos
    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)

    # Build agent state compatible with initialize_agent output
    agent_state = {
        "agent_id": 0,
        "camera": camera,
        "planner": planner,
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

    agents = [agent_state]

    # Generate unique run ID for this iteration (timestamp-based for linking logs/plots)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    simulator = Simulator(
        agents=agents,
        map_obj=map_obj,
        ground_truth_map=ground_truth_map,
        conf_dict=conf_dict,
        grid_info=grid_info,
        n_steps=n_steps,
        results_folder=results_folder,
        corr_type=corr_type,
        e_margin=e_margin,
        grf_r=grf_r,
        iter_idx=iter_idx,
        enable_stepwise_plotting=ENABLE_STEPWISE_PLOTTING,
        enable_logging=ENABLE_LOGGING,
        action_strategy=action_strategy,
        coordinator=None,
        multi_agent_logger=None,
        run_id=run_id,
        debug_logs=False,
    )

    result = simulator.run()

    # Extract single-agent formatted result
    agent_res = result["agents"][0]
    return {
        "entropy": agent_res["entropy"],
        "mse": agent_res["mse"],
        "coverage": agent_res["coverage"],
        "height": agent_res["height"],
        "uav_positions": agent_res["uav_positions"],
        "actions": agent_res["actions"],
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
    camera_hrange=None,
    news_mode=None,
    init_camera=None,
    init_planner=None,
    init_occupancy_map=None,
    debug_logs=False,
):
    """
    Run a multi-agent experiment iteration with decentralized coordination.
    """
    ma_config = config.get("multi_agent", {})
    num_agents = config.get("num_agents", ma_config.get("num_agents", 1))
    start_position = config.get("start_position", "corner")

    if debug_logs:
        from experiment_config import get_main_logger

        logger = get_main_logger()
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-AGENT EXPERIMENT: {num_agents} agents")
        logger.info(f"{'='*60}\n")

    # Create RNG
    rng = np.random.default_rng(seed)

    # If single-agent, avoid creating coordinator and any messaging.
    agents = []
    if num_agents == 1:
        # Compute start position locally (replicates previous single-agent logic)
        if start_position == "edge":
            real_border = [
                (
                    -grid_info.x / 2,
                    rng.uniform(-grid_info.y / 2, grid_info.y / 2),
                ),
                (
                    grid_info.x / 2,
                    rng.uniform(-grid_info.y / 2, grid_info.y / 2),
                ),
                (
                    rng.uniform(-grid_info.x / 2, grid_info.x / 2),
                    grid_info.y / 2,
                ),
                (
                    rng.uniform(-grid_info.x / 2, grid_info.x / 2),
                    -grid_info.y / 2,
                ),
            ]
            start_xy = real_border[rng.integers(len(real_border))]
        elif start_position == "corner":
            corners = [
                (-grid_info.x / 2, -grid_info.y / 2),
                (-grid_info.x / 2, grid_info.y / 2),
                (grid_info.x / 2, -grid_info.y / 2),
                (grid_info.x / 2, grid_info.y / 2),
            ]
            start_xy = corners[rng.integers(len(corners))]
        elif start_position == "center":
            start_xy = (0.0, 0.0)
        else:
            raise ValueError(
                f"Invalid start_position: {start_position}. Choose from 'edge', 'corner', or 'center'."
            )

        # Altitude from camera_hrange if provided, else fallback to min_alt
        start_z = camera_hrange[0] if camera_hrange is not None else min_alt

        if (
            init_camera is not None
            and init_planner is not None
            and init_occupancy_map is not None
        ):
            init_camera.set_altitude(start_z)
            init_camera.set_position(start_xy)
            agent_state = {
                "agent_id": 0,
                "camera": init_camera,
                "planner": init_planner,
                "occupancy_map": init_occupancy_map,
                "uav_pos": uav_position((start_xy, start_z)),
                "uav_positions": [uav_position((start_xy, start_z))],
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
        else:
            agent_state = initialize_agent(
                agent_id=0,
                grid_info=grid_info,
                start_position=start_xy,
                action_strategy=action_strategy,
                conf_dict=conf_dict,
                corr_type=corr_type,
                mcts_params=mcts_params,
                optimal_alt=optimal_alt,
                min_alt=min_alt,
                overlap=overlap,
                seed=seed,
                coordinator=None,
                start_altitude=start_z,
                debug_logs=debug_logs,
            )
            agents.append(agent_state)

        coordinator = None
    else:
        # Multi-agent: create full coordinator and initialize agents
        coord_news_mode = None
        if news_mode:
            if "_BS" in news_mode:
                coord_news_mode = "BS"
            elif "_BM" in news_mode:
                coord_news_mode = "BM"
            elif news_mode in ["BS", "BM"]:
                coord_news_mode = news_mode
        coordinator = MultiAgentCoordinator(
            grid_shape=grid_info.shape,
            config=config,
            conf_dict=conf_dict,
            correlation_type=corr_type,
            news_mode=coord_news_mode,
            mode=news_mode,
            grid_info=grid_info,
            debug_logs=debug_logs,
        )

        # Generate start positions for all agents
        start_positions = coordinator.reset_start_position(
            grid_info=grid_info,
            start_position=start_position,
            min_distance=10.0,
            seed=seed,
            camera_hrange=camera_hrange,
        )
        if debug_logs:
            logger.info(f"Start positions: {start_positions}")

        for agent_id in range(num_agents):
            sp = start_positions[agent_id]
            if isinstance(sp, (list, tuple)) and len(sp) == 3:
                start_xy = (sp[0], sp[1])
                start_z = sp[2]

            agent_state = initialize_agent(
                agent_id=agent_id,
                grid_info=grid_info,
                start_position=start_xy,
                action_strategy=action_strategy,
                conf_dict=conf_dict,
                corr_type=corr_type,
                mcts_params=mcts_params,
                optimal_alt=optimal_alt,
                min_alt=min_alt,
                overlap=overlap,
                seed=seed,
                coordinator=coordinator,
                start_altitude=start_z,
                debug_logs=debug_logs,
            )
            agents.append(agent_state)

            # Update coordinator with initial position
            camera = agent_state["camera"]
            uav_pos = agent_state["uav_pos"]
            current_row, current_col = camera.convert_xy_ij(
                uav_pos.position[0], uav_pos.position[1], camera.grid.center
            )
            coordinator.update_agent_state(
                agent_id=agent_id,
                position=(current_row, current_col),
                altitude=uav_pos.altitude,
            )

    # Get news_mode and sharing options for logging
    dec_config = config.get("decentralized", {})
    news_sharing = dec_config.get("news_sharing", True)
    position_sharing = dec_config.get("position_sharing", True)

    if news_mode is None:
        if not position_sharing and not news_sharing:
            news_mode = "IG"
        elif position_sharing and not news_sharing:
            greedy_cfg = config.get("greedy_ig", {})
            if "multi_agent" in config and isinstance(config["multi_agent"], dict):
                greedy_cfg = {
                    **greedy_cfg,
                    **config["multi_agent"].get("greedy_ig", {}),
                }
            enable_discounting = greedy_cfg.get("enable_discounting", False)
            news_mode = "IGd" if enable_discounting else "IG"
        else:
            news_mode = ma_config.get("news_mode", dec_config.get("news_mode", "BM"))

    # Setup logging
    multi_agent_logger = None
    try:
        coordinator.mode = news_mode
    except Exception:
        pass

    # Generate unique run ID for this iteration (timestamp-based for linking logs/plots)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if ENABLE_LOGGING:
        log_folder = os.path.join(results_folder, "txt")
        init_positions = [agent["uav_pos"] for agent in agents]
        logged_hyperparams = _build_logged_hyperparams(
            config=config,
            action_strategy=action_strategy,
            mcts_params=mcts_params,
        )

        # Determine if hierarchical timing columns are needed
        use_hierarchical_timing = action_strategy in (
            "hierarchical_dec_mcts",
            "mh_dec_mcts",
            "mh_dec_mcts_both",
            "mh_dec_mcts_full",
            "mh_dec_mcts_efficient",
        )

        multi_agent_logger = FastLogger(
            log_folder,
            strategy=action_strategy,
            pairwise=corr_type,
            grid=grid_info,
            init_x=init_positions,
            r=grf_r,
            iteration=iter_idx,
            num_agents=num_agents,
            e=e_margin,
            conf_dict=conf_dict,
            filename="run.log",
            multi_agent=True,
            news_mode=news_mode,
            use_hierarchical_timing=use_hierarchical_timing,
            run_id=run_id,
            header_extras=[
                ("hyperparams", json.dumps(logged_hyperparams, sort_keys=True)),
                ("mcts_params", json.dumps(mcts_params, sort_keys=True)),
            ],
        )

    # Setup step-wise plotting directory
    if ENABLE_STEPWISE_PLOTTING:
        # Use "plots" as folder name instead of correlation type pattern
        plot_folder = os.path.join(results_folder, "plots")
        os.makedirs(
            f"{plot_folder}/{iter_idx}/steps/",
            exist_ok=True,
        )

    simulator = Simulator(
        agents=agents,
        map_obj=map_obj,
        ground_truth_map=ground_truth_map,
        conf_dict=conf_dict,
        grid_info=grid_info,
        n_steps=n_steps,
        results_folder=results_folder,
        corr_type=corr_type,
        e_margin=e_margin,
        grf_r=grf_r,
        iter_idx=iter_idx,
        enable_stepwise_plotting=ENABLE_STEPWISE_PLOTTING,
        enable_logging=ENABLE_LOGGING,
        action_strategy=action_strategy,
        coordinator=coordinator,
        multi_agent_logger=multi_agent_logger,
        run_id=run_id,
        debug_logs=debug_logs,
    )

    result = simulator.run()

    # Log coordination statistics when available (only if debug_logs enabled)
    if debug_logs:
        coord_stats = result.get("coordination_stats", {})
        if coord_stats:
            logger.info(f"\n{'='*60}")
            logger.info("MULTI-AGENT COORDINATION STATISTICS")
            logger.info(f"{'='*60}")
            logger.info(f"Belief fusions: {coord_stats.get('belief_fusions', 0)}")
            logger.info(
                f"Region allocations: {coord_stats.get('region_allocations', 0)}"
            )
            logger.info(
                f"Collision avoidances: {coord_stats.get('collision_avoidances', 0)}"
            )
            logger.info(f"Communication stats: {coord_stats.get('comm_bus', {})}")
            if "lbp_fusion" in coord_stats:
                logger.info(f"LBP Fusion stats: {coord_stats['lbp_fusion']}")
            if "coverage_per_agent" in coord_stats:
                logger.info(f"Coverage per agent: {coord_stats['coverage_per_agent']}")
            logger.info(f"{'='*60}\n")

    return result

import os
import sys
import json
import random
import logging
import numpy as np
from tqdm import tqdm
import argparse
import copy
from datetime import datetime


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
from async_runner import AsyncMultiAgentRunner, create_async_runner_from_config

from viewer import plot_metrics, plot_terrain

from helper import create_run_folder, make_param_tag
from config_loader import load_config  # Use new config loader with backward compat
from experiment_utils import (
    initialize_agent,
    compute_agent_metrics,
    update_agent_observation,
    compute_multi_agent_fused_metrics,
    finalize_planners,
    get_results_folder,
    process_agent_observations,
    perform_belief_fusion,
    select_agent_actions,
    update_agent_positions,
    extract_region_metadata,
)

import matplotlib

matplotlib.use("Agg")


# =============================================================================
# Logging Setup - Redirect all output to file
# =============================================================================

# Store original stdout for tqdm
_original_stdout = sys.stdout
_original_stderr = sys.stderr


def setup_main_logger(log_dir: str = "logs", experiment_name: str = None) -> str:
    """
    Set up main logger to redirect all output to a file.

    This captures:
    - All logging.* calls
    - All print() statements (via stdout redirect)

    Args:
        log_dir: Directory for log files
        experiment_name: Optional experiment name for log filename

    Returns:
        Path to the created log file
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_suffix = f"_{experiment_name}" if experiment_name else ""
    log_filename = f"main{exp_suffix}_{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            # Optionally keep console output: logging.StreamHandler()
        ],
    )

    # Create a custom stream to redirect print() to logger
    class LoggerWriter:
        """Redirect stdout/stderr to logger and file."""

        def __init__(self, logger, level, log_file_handle):
            self.logger = logger
            self.level = level
            self.log_file = log_file_handle
            self.buffer = ""

        def write(self, message):
            if message and message.strip():
                self.logger.log(self.level, message.strip())

        def flush(self):
            if self.log_file:
                self.log_file.flush()

    # Open log file for print redirect
    log_file_handle = open(log_file, "a")

    # Get main logger
    main_logger = logging.getLogger("main")

    # Redirect print statements to logger
    sys.stdout = LoggerWriter(main_logger, logging.INFO, log_file_handle)
    sys.stderr = LoggerWriter(main_logger, logging.ERROR, log_file_handle)

    # Log initialization (use original stdout for immediate feedback)
    _original_stdout.write(f"Logging to: {log_file}\n")
    _original_stdout.flush()

    main_logger.info("=" * 80)
    main_logger.info("MAIN EXPERIMENT LOG")
    main_logger.info(f"Experiment: {experiment_name if experiment_name else 'default'}")
    main_logger.info(f"Log file: {log_file}")
    main_logger.info("=" * 80)

    return log_file


def get_main_logger():
    """Get the main logger instance."""
    return logging.getLogger("main")


def get_tqdm_file():
    """Get file handle for tqdm progress bars (uses original stderr)."""
    return _original_stderr


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
            iteration=iter_idx,
            num_agents=1,
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

    for step in tqdm(
        range(0, n_steps), desc="steps", position=3, leave=False, file=get_tqdm_file()
    ):
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
            # Get region metadata from hierarchical planners
            region_metadata, selected_region_id, region_scores = (
                extract_region_metadata(planner, action_strategy)
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


def run_async_multi_agent(
    config,
    agents,
    coordinator,
    map_obj,
    ground_truth_map,
    grid_info,
    n_steps,
    results_folder,
    corr_type,
    action_strategy,
    e_margin,
    grf_r,
    iter_idx,
    conf_dict,
    ENABLE_STEPWISE_PLOTTING,
    ENABLE_LOGGING,
    multi_agent_logger,
    fused_entropy_history,
    fused_mse_history,
    combined_coverage_history,
):
    """
    Run multi-agent experiment with true asynchronous execution.

    Each agent runs in its own thread with independent planning cycles.
    Agents communicate via async message queues with realistic delays.
    D-UCT discounting handles stale intents from asynchronous drift.

    Args:
        config: Configuration dict
        agents: List of pre-initialized agent state dicts
        coordinator: MultiAgentCoordinator instance
        map_obj: Map/environment object for observations
        ground_truth_map: Ground truth for metrics
        grid_info: Grid information
        n_steps: Target number of steps
        ... (other args for logging/plotting)

    Returns:
        Results dict compatible with sync version
    """
    async_config = config.get("decentralized", {}).get("async", {})
    num_agents = len(agents)

    print(f"\n{'='*60}")
    print(f"ASYNC MULTI-AGENT EXPERIMENT: {num_agents} agents")
    print(f"Planning rate: {async_config.get('planning_rate_hz', 5.0)} Hz")
    print(f"HLP rate: {async_config.get('hlp_rate_hz', 2.0)} Hz")
    print(f"Comm delay: {async_config.get('comm_delay_ms', 50.0)} ms")
    print(f"Max intent age: {async_config.get('max_intent_age_sec', 5.0)} sec")
    print(f"{'='*60}\n")

    # Create async runner
    async_runner = AsyncMultiAgentRunner(
        num_agents=num_agents,
        planning_rate_hz=async_config.get("planning_rate_hz", 5.0),
        hlp_rate_hz=async_config.get("hlp_rate_hz", 2.0),
        comm_delay_ms=async_config.get("comm_delay_ms", 50.0),
        drop_probability=async_config.get("drop_probability", 0.0),
    )

    # Set coordinator for collision avoidance and belief fusion
    async_runner.set_coordinator(coordinator)

    # Add agents to the async runner
    for agent in agents:
        agent_id = agent["agent_id"]
        async_runner.add_agent(
            agent_id=agent_id,
            camera=agent["camera"],
            planner=agent["planner"],
            belief_map=agent["belief_map"],
            occupancy_map=agent["occupancy_map"],
            initial_position=agent["uav_pos"].position,
            initial_altitude=agent["uav_pos"].altitude,
            config=config,
            coordinator=coordinator,
        )

    # Checkpoint callback for logging and visualization
    checkpoint_count = [0]

    def on_checkpoint(agent_states):
        """Called periodically during async execution."""
        nonlocal checkpoint_count
        step = checkpoint_count[0]

        # Log progress
        positions = [(s.position, s.altitude) for s in agent_states.values()]
        steps = [s.step for s in agent_states.values()]
        cycles = [s.planning_cycles for s in agent_states.values()]

        print(f"[Async Checkpoint {step}] Steps: {steps}, Planning cycles: {cycles}")

        # Compute metrics from current beliefs (simplified for async)
        # In async mode, we log at checkpoints rather than per-step
        checkpoint_count[0] += 1

    # Set checkpoint callback
    checkpoint_interval = (
        1.0 / async_config.get("observation_rate_hz", 10.0) * 10
    )  # Every 10 obs cycles
    async_runner.set_checkpoint_callback(on_checkpoint, interval=checkpoint_interval)

    # Calculate runtime based on expected step duration
    # Estimate: n_steps at planning_rate_hz
    planning_rate = async_config.get("agent_planning_rate_hz", 5.0)
    estimated_duration = n_steps / planning_rate
    timeout = estimated_duration * 2.0  # 2x buffer for safety

    print(
        f"Running async experiment for ~{estimated_duration:.1f}s (timeout: {timeout:.1f}s)"
    )

    # Run until all agents complete target steps
    async_runner.run_until_steps(target_steps=n_steps, timeout_sec=timeout)

    # Collect results
    all_states = async_runner.get_all_states()
    all_stats = async_runner.get_all_statistics()
    all_logs = async_runner.get_all_logs()
    network_stats = async_runner.get_network_statistics()

    print(f"\n{'='*60}")
    print("ASYNC EXECUTION STATISTICS")
    print(f"{'='*60}")
    for agent_id, stats in all_stats.items():
        print(
            f"Agent {agent_id}: LLP cycles={stats['llp_cycles']}, "
            f"HLP cycles={stats['hlp_cycles']}, "
            f"actions={stats['actions_executed']}, "
            f"intents_sent={stats['intents_broadcast']}, "
            f"intents_rcvd={stats['intents_received']}"
        )
    print(
        f"Network: sent={network_stats['messages_sent']}, "
        f"delivered={network_stats['messages_delivered']}, "
        f"dropped={network_stats['messages_dropped']}"
    )
    print(f"{'='*60}\n")

    # Convert async logs to compatible format
    # Extract action sequences and positions from logs
    results_agents = []
    for agent_id in range(num_agents):
        agent_log = all_logs.get(agent_id, [])
        agent_state = all_states.get(agent_id)

        actions = [entry["action"] for entry in agent_log]
        positions = [entry["position"] for entry in agent_log]

        # Reconstruct uav_positions from log
        uav_positions = [
            uav_position((entry["position"], entry["altitude"])) for entry in agent_log
        ]
        # Add initial position
        if uav_positions:
            uav_positions.insert(0, agents[agent_id]["uav_positions"][0])

        # Metrics would need to be computed from final state
        # For now, return empty lists (async doesn't track per-step metrics the same way)
        results_agents.append(
            {
                "agent_id": agent_id,
                "entropy": [],  # Would need reconstruction
                "mse": [],
                "coverage": [],
                "height": [entry["altitude"] for entry in agent_log],
                "uav_positions": uav_positions,
                "actions": actions,
            }
        )

    coord_stats = coordinator.get_statistics()
    coord_stats["async_execution"] = {
        "agent_stats": all_stats,
        "network_stats": network_stats,
    }

    # Close logger
    if ENABLE_LOGGING and multi_agent_logger is not None:
        multi_agent_logger.close()

    return {
        "agents": results_agents,
        "coordination_stats": coord_stats,
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
    num_agents = config.get("num_agents", ma_config.get("num_agents", 1))
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
        start_pos = start_positions[agent_id]

        agent_state = initialize_agent(
            agent_id=agent_id,
            grid_info=grid_info,
            start_position=start_pos,
            action_strategy=action_strategy,
            conf_dict=conf_dict,
            corr_type=corr_type,
            mcts_params=mcts_params,
            optimal_alt=optimal_alt,
            min_alt=min_alt,
            overlap=overlap,
            seed=seed,
            coordinator=coordinator,
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

        print(
            f"Agent {agent_id}: Start position {start_pos}, grid ({current_row:.0f}, {current_col:.0f})"
        )

    # Setup logging - single unified logger for multi-agent
    multi_agent_logger = None
    if ENABLE_LOGGING:
        log_folder = os.path.join(results_folder, "txt")
        # Collect all agent init positions
        init_positions = [agent["uav_pos"] for agent in agents]
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
            filename=f"multi_agent_iter{iter_idx}.log",
            multi_agent=True,
            header_extras=[
                ("mcts_params", json.dumps(mcts_params, sort_keys=True)),
            ],
        )

    # Setup step-wise plotting directory
    if ENABLE_STEPWISE_PLOTTING:
        os.makedirs(
            results_folder
            + f"/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter_idx}/steps/",
            exist_ok=True,
        )

    # Initialize fused metrics lists for plotting
    fused_entropy_history = []
    fused_mse_history = []
    combined_coverage_history = []

    # =========================================================================
    # CHECK FOR ASYNC EXECUTION MODE
    # =========================================================================
    async_config = config.get("decentralized", {}).get("async", {})
    async_enabled = async_config.get("enabled", False)

    if async_enabled:
        # Run in true async mode using threaded agents
        return run_async_multi_agent(
            config=config,
            agents=agents,
            coordinator=coordinator,
            map_obj=map_obj,
            ground_truth_map=ground_truth_map,
            grid_info=grid_info,
            n_steps=n_steps,
            results_folder=results_folder,
            corr_type=corr_type,
            action_strategy=action_strategy,
            e_margin=e_margin,
            grf_r=grf_r,
            iter_idx=iter_idx,
            conf_dict=conf_dict,
            ENABLE_STEPWISE_PLOTTING=ENABLE_STEPWISE_PLOTTING,
            ENABLE_LOGGING=ENABLE_LOGGING,
            multi_agent_logger=multi_agent_logger,
            fused_entropy_history=fused_entropy_history,
            fused_mse_history=fused_mse_history,
            combined_coverage_history=combined_coverage_history,
        )

    # =========================================================================
    # SYNCHRONOUS EXECUTION MODE (DEFAULT)
    # =========================================================================
    # Main multi-agent loop
    for step in tqdm(
        range(0, n_steps), desc="steps", position=3, leave=False, file=get_tqdm_file()
    ):
        # =====================================================================
        # PHASE 1: All agents observe and update local belief (SYNCHRONOUS)
        # =====================================================================
        agent_observations = process_agent_observations(
            agents, map_obj, conf_dict, coordinator
        )

        # Log sigmas once at the start (step 0 only)
        if step == 0:
            sample_sigmas = agent_observations.get(0, {}).get("sigmas")
            if sample_sigmas:
                print(
                    f"Sensor model sigmas: s0={sample_sigmas[0]:.4f}, s1={sample_sigmas[1]:.4f}"
                )
            else:
                print("Sensor model: No sigmas (perfect observations)")

        # =====================================================================
        # PHASE 2: Synchronous belief fusion across agents
        # =====================================================================
        perform_belief_fusion(agents, coordinator, agent_observations, grid_info, step)

        # =====================================================================
        # PHASE 3: Compute metrics, select actions, and update positions
        # =====================================================================
        select_agent_actions(
            agents, ground_truth_map, grid_info, coordinator, config, step
        )
        update_agent_positions(agents, coordinator)

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
            region_metadata, selected_region_id, region_scores = (
                extract_region_metadata(agents[0]["planner"], action_strategy)
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
                (
                    agent_region_metadata,
                    agent_selected_region,
                    agent_region_scores,
                ) = extract_region_metadata(agent_planner, action_strategy)

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

            # Plot aggregated metrics for multi-agent iteration
            # Use fused belief for entropy/MSE, OR-based coverage, per-agent heights

            # Compute combined observed_ids (union of all agents' observed_ids)
            combined_observed_ids = set()
            for agent in agents:
                combined_observed_ids.update(agent["observed_ids"])

            # Compute metrics from fused belief (display_belief)
            fused_entropy_val, fused_mse_val, combined_coverage_val = compute_metrics(
                ground_truth_map, display_belief, combined_observed_ids, grid_info
            )

            # Append to running history lists
            fused_entropy_history.append(fused_entropy_val)
            fused_mse_history.append(fused_mse_val)
            combined_coverage_history.append(combined_coverage_val)

            # Log multi-agent data (common metrics + per-agent lists)
            if ENABLE_LOGGING and multi_agent_logger is not None:
                heights = [agent["height"][-1] for agent in agents]
                actions = [
                    agent["actions"][-1] if len(agent["actions"]) > 0 else None
                    for agent in agents
                ]
                igs = [
                    (
                        agent["info_gain_action"].get(agent["actions"][-1])
                        if len(agent["actions"]) > 0
                        else None
                    )
                    for agent in agents
                ]
                multi_agent_logger.log_multi_agent_data(
                    entropy=fused_entropy_val,
                    mse=fused_mse_val,
                    coverage=combined_coverage_val,
                    heights=heights,
                    actions=actions,
                    igs=igs,
                    step=step,
                )

            # Per-agent heights (list of lists)
            per_agent_heights = [agent["height"][: step + 1] for agent in agents]

            plot_metrics(
                f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/iter_{iter_idx}.png",
                fused_entropy_history,
                fused_mse_history,
                combined_coverage_history,
                per_agent_heights,
                height_range=agents[0]["camera"].get_hrange(),
            )

        print(f"{'─'*40}")

    # Drain any remaining messages from queues (cleanup old messages)
    coordinator.comm_bus.clear_all_queues()

    # Finalize planners using utility function
    finalize_planners(agents)

    # Close logger
    if ENABLE_LOGGING and multi_agent_logger is not None:
        multi_agent_logger.close()

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
    run_base = f"{config['action_strategy']}_{config['field_type'].lower()}_{config['start_position']}"
    if config.get("action_strategy") == "mcts" and config.get("params_in_path", True):
        run_base = run_base + "__" + make_param_tag(config.get("mcts_params", {}))
    results_folder = os.path.join(base_dir, run_base)

    # Setup logging to file (must be early, before any print statements)
    log_dir = os.path.join(results_folder, "logs")
    log_file = setup_main_logger(log_dir=log_dir, experiment_name=run_base)
    logger = get_main_logger()
    logger.info(f"Results folder: {results_folder}")
    logger.info(f"Config loaded from: {args.config}")

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
    num_agents = config.get("num_agents", ma_config.get("num_agents", 1))

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

    seed = config.get("seed", 42)

    # Create initial camera (for single-agent or field initialization)
    camera1 = Camera(
        grid_info,
        60,
        camera_altitude=min_alt,
        f_overlap=overlap,
        s_overlap=overlap,
        seed=seed,
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

    for corr_type in tqdm(
        correlation_types, desc="Pairwise", position=0, file=get_tqdm_file()
    ):
        for e_margin in tqdm(
            error_margins,
            desc=f"Error Margins (pairwise = {corr_type})",
            position=1,
            file=get_tqdm_file(),
        ):
            for iter_idx in tqdm(
                range(iters[0], iters[-1]),
                desc=f"Iters (e={e_margin})",
                position=2,
                leave=False,
                file=get_tqdm_file(),
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
                        seed=seed,
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
                    elif start_position == "center":
                        start_pos = (0.0, 0.0)
                    else:
                        raise ValueError(
                            f"Invalid start_position: {start_position}. Choose from 'edge', 'corner', or 'center'."
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
                    print("Single-agent experiment completed")

                    # Finalize planner
                    if hasattr(planner, "finalize_episode"):
                        planner.finalize_episode()


if __name__ == "__main__":
    main()

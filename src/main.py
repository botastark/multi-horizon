import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")

from helper import make_param_tag
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera
from config_loader import load_config, flatten_hierarchical_config

from experiment_config import (
    setup_main_logger,
    get_main_logger,
    get_tqdm_file,
    load_global_paths,
    parse_args,
)
from experiment_runner import run_multi_agent_experiment


def run_experiment_with_config(config: dict, args):
    """Run a single experiment with the given configuration."""
    # Flatten hierarchical configs for backward compatibility
    config = flatten_hierarchical_config(config)

    # Extract configuration parameters
    (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    ) = load_global_paths(config)

    # Check if output_dir is provided in config (from web interface)
    exp_config = config.get("experiment", {})
    output_dir = exp_config.get("output_dir")

    if output_dir:
        # Use provided output directory (from web interface or explicit config)
        base_dir = output_dir
        results_folder = base_dir
        run_base = os.path.basename(base_dir)
        # Skip folder name generation, already done by caller
        skip_folder_generation = True
    else:
        # Use experiments/temp/ for command-line runs (organized structure)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(PROJECT_PATH, "experiments", "temp")
        os.makedirs(base_dir, exist_ok=True)
        skip_folder_generation = False
        # Store timestamp for later use in moving to runs/
        temp_timestamp = timestamp

    # Get multi-agent and news mode configuration for folder naming
    ma_config = config.get("multi_agent", {})
    dec_config = config.get("decentralized", {})
    num_agents = config.get("num_agents", ma_config.get("num_agents", 1))

    # Determine news mode based on sharing options:
    # - IG: No sharing (position_sharing=false, news_sharing=false)
    # - IG_d: Position sharing only (position_sharing=true, news_sharing=false)
    # - BS/BM: Position + news sharing (position_sharing=true, news_sharing=true)
    position_sharing = dec_config.get("position_sharing", True)
    news_sharing = dec_config.get("news_sharing", True)

    # Allow explicit mode label override from strategy-specific config or shared config
    strategy_cfg = config.get(config.get("action_strategy", ""), {})
    explicit_label = strategy_cfg.get("mode_labels", strategy_cfg.get("mode_label"))

    # If not in strategy config, check shared config
    if explicit_label is None:
        explicit_label = config.get("mode_labels")

    # Compute inferred label from sharing flags and news_mode
    inferred_label = "IG"
    if num_agents > 1:
        if not position_sharing and not news_sharing:
            inferred_label = "IG"
        elif position_sharing and not news_sharing:
            inferred_label = "IGd"
        else:
            # News sharing enabled: use BS or BM mode
            news_mode_setting = ma_config.get(
                "news_mode", dec_config.get("news_mode", "BM")
            )
            # Determine label based on position_sharing + news_mode
            # position_sharing=False, news_sharing=True → IG_BS or IG_BM
            # position_sharing=True, news_sharing=True → IGd_BS or IGd_BM
            if news_mode_setting in ["BS", "BM"]:
                inferred_label = (
                    f"IG_{news_mode_setting}"
                    if not position_sharing
                    else f"IGd_{news_mode_setting}"
                )
            else:
                inferred_label = news_mode_setting
    else:
        inferred_label = "IG"

    # Get radius (grf_r) - Read from config or use defaults
    field_type_raw = config.get("field_type", "Gaussian")
    if field_type_raw == "Ortomap":
        grf_r_for_name = "orto"
    else:
        grf_r_for_name = config.get("cluster_radius", 5)

    # If strategy-specific `mode_label` is a list, run experiments for each label.
    # Otherwise resolve a single mode_label and run once.
    valid_labels = {"IG", "IG_BS", "IG_BM", "IGd", "IGd_BS", "IGd_BM"}

    if isinstance(explicit_label, list):
        # Filter valid labels from the list
        mode_labels = [
            l for l in explicit_label if isinstance(l, str) and l in valid_labels
        ]
        if not mode_labels:
            # No valid labels in list, fall back to inferred
            if inferred_label in valid_labels:
                mode_labels = [inferred_label]
            else:
                mode_labels = ["IG"]
    elif (
        explicit_label
        and isinstance(explicit_label, str)
        and explicit_label in valid_labels
    ):
        # Single explicit label provided
        mode_labels = [explicit_label]
    else:
        # Use inferred label
        if inferred_label in valid_labels:
            mode_labels = [inferred_label]
        else:
            # try to normalize forms like 'BM' or 'BS'
            if inferred_label in ["BM", "BS"]:
                mode_labels = [f"IG_{inferred_label}"]
            else:
                mode_labels = [inferred_label]

    # Loop over mode labels (allows batch runs via greedy_ig.mode_label = [...])
    for news_mode in mode_labels:
        # Parse the mode label to set config flags for this run
        # IG = no sharing (position_sharing=False, news_sharing=False)
        # IGd = position only (position_sharing=True, news_sharing=False)
        # IG_BS/IG_BM = IG with news sharing (position_sharing=False, news_sharing=True, news_mode=BS/BM)
        # IGd_BS/IGd_BM = IGd with news sharing (position_sharing=True, news_sharing=True, news_mode=BS/BM)

        if news_mode == "IG":
            # No sharing
            config["decentralized"]["position_sharing"] = False
            config["decentralized"]["news_sharing"] = False
            actual_news_mode = None
        elif news_mode == "IGd":
            # Position sharing only
            config["decentralized"]["position_sharing"] = True
            config["decentralized"]["news_sharing"] = False
            actual_news_mode = None
        elif news_mode in ["IG_BS", "IG_BM"]:
            # IG with news sharing
            config["decentralized"]["position_sharing"] = False
            config["decentralized"]["news_sharing"] = True
            actual_news_mode = news_mode.split("_")[1]  # "BS" or "BM"
            if "multi_agent" not in config:
                config["multi_agent"] = {}
            config["multi_agent"]["news_mode"] = actual_news_mode

            # Limited testing mode: IG_BS uses infinite communication
            if config.get("limited_testing", False) and news_mode == "IG_BS":
                if (
                    "radius_multiplier" in dec_config
                    or config.get("action_strategy") == "greedy_ig"
                ):
                    config["decentralized"]["radius_multiplier"] = -1
                else:
                    config["decentralized"]["communication_range"] = -1

        elif news_mode in ["IGd_BS", "IGd_BM"]:
            # IGd with news sharing
            config["decentralized"]["position_sharing"] = True
            config["decentralized"]["news_sharing"] = True
            actual_news_mode = news_mode.split("_")[1]  # "BS" or "BM"
            if "multi_agent" not in config:
                config["multi_agent"] = {}
            config["multi_agent"]["news_mode"] = actual_news_mode

            # Limited testing mode: IGd_BM uses limited communication (5x multiplier)
            if config.get("limited_testing", False) and news_mode == "IGd_BM":
                if (
                    "radius_multiplier" in dec_config
                    or config.get("action_strategy") == "greedy_ig"
                ):
                    config["decentralized"]["radius_multiplier"] = 5
                else:
                    config["decentralized"]["communication_range"] = (
                        5 * 3.125
                    )  # 5x grid_step
        else:
            # Unknown mode, use defaults
            actual_news_mode = news_mode

        pa_reference_greedy_ig = (
            config.get("action_strategy") == "greedy_ig"
            and config.get("multi_agent", {}).get("pa_reference_compat", False)
        )
        if pa_reference_greedy_ig:
            config.setdefault("multi_agent", {})
            config.setdefault("greedy_ig", {})
            if news_mode == "IG_BS":
                config["decentralized"]["position_sharing"] = False
                config["decentralized"]["news_sharing"] = True
                config["multi_agent"]["news_mode"] = "BS"
                config["multi_agent"]["news_inference_type"] = "OG"
                actual_news_mode = "BS"
            elif news_mode == "IGd_BM":
                config["decentralized"]["position_sharing"] = True
                config["decentralized"]["news_sharing"] = False
                config["multi_agent"]["news_mode"] = "BM"
                config["greedy_ig"]["enable_discounting"] = True
                actual_news_mode = "BM"

        # Get communication range for folder naming
        # Check if radius_multiplier is specified (preferred, matches reference paper)
        # Read from config dict directly to get modified values
        radius_multiplier = config.get("decentralized", {}).get(
            "radius_multiplier",
            config.get("multi_agent", {}).get("radius_multiplier", None),
        )

        if radius_multiplier is not None:
            # Use radius_multiplier notation (e.g., "R5" for multiplier=5, "Rinf" for -1)
            comm_range_str = (
                "Rinf" if radius_multiplier == -1 else f"R{radius_multiplier}"
            )
        else:
            # Fallback to direct communication_range in meters
            comm_range = config.get("decentralized", {}).get(
                "communication_range",
                config.get("multi_agent", {}).get("communication_range", -1),
            )
            comm_range_str = "inf" if comm_range == -1 else str(comm_range)

        # Only build folder structure if not using provided output_dir
        if not skip_folder_generation:
            # Build run_base with all parameters: strategy_field_r{radius}_start_N{agents}_{mode}_commR{range}
            run_base = (
                f"{config['action_strategy']}_{config['field_type'].lower()}_"
                f"r{grf_r_for_name}_{config['start_position']}_N{num_agents}_{news_mode}_comm{comm_range_str}"
            )

            # Add experiment name suffix to distinguish different hyperparameter configs
            exp_name = config.get("experiment", {}).get("name")
            if exp_name:
                run_base = f"{run_base}__{exp_name}"
            elif config.get("action_strategy") == "mcts" and config.get(
                "params_in_path", True
            ):
                run_base = (
                    run_base + "__" + make_param_tag(config.get("mcts_params", {}))
                )
            results_folder = os.path.join(base_dir, run_base)
            # Store run_base for later moving to runs/ directory
            temp_run_base = run_base
        else:
            temp_run_base = None

        if results_folder:
            os.makedirs(results_folder, exist_ok=True)
            try:
                with open(
                    os.path.join(results_folder, "config_resolved.json"), "w"
                ) as config_file:
                    json.dump(config, config_file, indent=2)
            except Exception:
                pass

        # Check if debug logging is enabled (default: False)
        debug_logs = config.get(
            "debug_logs", config.get("shared", {}).get("debug_logs", False)
        )

        # Setup logging to file only if debug_logs is enabled
        if debug_logs:
            log_dir = os.path.join(results_folder, "logs")
            log_file = setup_main_logger(log_dir=log_dir, experiment_name=run_base)
            logger = get_main_logger()
            logger.info(f"Running experiment mode: {news_mode}")
            logger.info(f"Results folder: {results_folder}")
            logger.info(f"Config loaded from: {args.config}")
            logger.info(
                f"Experiment: num_agents={num_agents}, news_mode={news_mode}, radius={grf_r_for_name}"
            )
            logger.info(
                f"Strategy: {config['action_strategy']}, Field: {config['field_type']}, Start: {config['start_position']}"
            )
        else:
            # No debug logging - silent mode (no console output)
            logger = get_main_logger()

            ENABLE_STEPWISE_PLOTTING = config["enable_plotting"]
            ENABLE_LOGGING = config["enable_logging"]
            mcts_params = config.get("mcts_params", {})

            field_type = config["field_type"]
            start_position = config["start_position"]
            action_strategy = config["action_strategy"]
            correlation_types = config["correlation_types"]
            n_steps = config["n_steps"]
            iters = config["iters"]

            # num_agents and ma_config already defined above for folder naming

            if isinstance(iters, int):
                iters = [0, iters]
            error_margins = [
                None if e == "None" else e for e in config["error_margins"]
            ]
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
                grf_r = config.get("cluster_radius", 5)
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
            # Note: No global np.random.seed() here - Camera and Field manage their own seeds
            # Per-iteration seeding happens inside the loop to ensure reproducible but different runs

            # Create initial camera (for single-agent or field initialization)
            camera1 = Camera(
                grid_info,
                60,
                camera_altitude=min_alt,
                f_overlap=overlap,
                s_overlap=overlap,
                seed=seed,
                a=1.0,
                b=0.015,
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
                seed=seed,  # Fixed seed for stable map across iterations
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
                        # Generate unique seed for this iteration
                        base_seed = config.get("seed", 42)
                        iteration_seed = base_seed + iter_idx

                        map_obj.reset(seed=iteration_seed)
                        ground_truth_map = map_obj.get_ground_truth()

                        if e_margin is not None:
                            # sampled sensor model
                            conf_dict = map_obj.init_s0_s1(
                                e=e_margin,
                                sensor=use_sensor_model,
                            )
                        else:
                            # theoretical sensor model
                            conf_dict = camera1.theoretical_conf_dict()

                        # Unified experiment call: use the same multi-agent framework
                        # for N==1 and N>1 to avoid duplicated logic.
                        # If single-agent experiment, build and pass initial camera/planner
                        init_camera = None
                        init_planner = None
                        init_occupancy_map = None
                        if num_agents == 1:
                            init_occupancy_map = OM(
                                grid_info.shape,
                                conf_dict=conf_dict,
                                correlation_type=corr_type,
                            )
                            init_planner = planning(
                                grid_info,
                                camera1,
                                action_strategy,
                                conf_dict=conf_dict,
                                optimal_alt=optimal_alt,
                                mcts_params=mcts_params,
                                seed=iteration_seed,
                            )
                            init_camera = camera1

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
                            min_alt=camera1.get_hrange()[0],
                            camera_hrange=camera1.get_hrange(),
                            overlap=overlap,
                            optimal_alt=optimal_alt,
                            seed=iteration_seed,
                            news_mode=news_mode,
                            init_camera=init_camera,
                            init_planner=init_planner,
                            init_occupancy_map=init_occupancy_map,
                            debug_logs=debug_logs,
                        )
                    if debug_logs:
                        logger.info(f"Experiment completed with {num_agents} agents")

            # Move completed experiment from temp/ to runs/<method>/ after this mode completes
            # (only for command-line runs)
            if (
                not skip_folder_generation
                and temp_run_base
                and results_folder.startswith(
                    os.path.join(PROJECT_PATH, "experiments", "temp")
                )
            ):
                try:
                    import shutil

                    # Get method name for organizing runs
                    method = config.get("action_strategy", "unknown")
                    runs_dir = os.path.join(PROJECT_PATH, "experiments", "runs", method)
                    os.makedirs(runs_dir, exist_ok=True)

                    # Build run directory name with timestamp and parameters
                    run_dirname = f"run_{temp_timestamp}_{temp_run_base}"

                    final_path = os.path.join(runs_dir, run_dirname)

                    # Move from temp to runs/<method>/
                    if os.path.exists(results_folder):
                        shutil.move(results_folder, final_path)
                        print(f"✓ Results saved to: {final_path}")
                        if debug_logs:
                            logger.info(f"✓ Results saved to: {final_path}")

                except Exception as e:
                    print(f"⚠ Failed to move results to runs/ directory: {e}")
                    print(f"⚠ Results remain at: {results_folder}")
                    if debug_logs:
                        logger.warning(
                            f"Failed to move results to runs/ directory: {e}"
                        )
                        logger.warning(f"Results remain at: {results_folder}")


def main():
    """Main entry point supporting both single and multi-strategy configs."""
    args = parse_args()
    configs = load_config(args.config)

    # Handle both single config and list of configs
    if not isinstance(configs, list):
        configs = [configs]

    logger = get_main_logger()
    logger.info(f"Loaded {len(configs)} configuration(s) from: {args.config}")

    # Run experiments for each strategy configuration
    for config_idx, config in enumerate(configs, 1):
        strategy_name = config.get("action_strategy", "unknown")
        logger.info(f"\n{'='*80}")
        logger.info(f"Running strategy {config_idx}/{len(configs)}: {strategy_name}")
        logger.info(f"{'='*80}\n")

        run_experiment_with_config(config, args)

        logger.info(f"\nCompleted strategy: {strategy_name}\n")


if __name__ == "__main__":
    main()

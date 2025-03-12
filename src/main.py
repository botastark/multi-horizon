import os
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

from helper import (
    FastLogger,
    compute_metrics,
    observed_m_ids,
    uav_position,
)
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import camera
from viewer import plot_metrics, plot_terrain, plot_terrain_2d


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
    ANNOTATION_PATH = os.path.join(PROJECT_PATH, "src", "annotation.txt")
    ORTHOMAP_PATH = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
    TILE_PIXEL_PATH = os.path.join(PROJECT_PATH, "data", "tomatotiles.txt")
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
    results_folder = os.path.join(
        PROJECT_PATH,
        f"results_{config['field_type'].lower()}_{config['start_position']}",
    )
    ENABLE_STEPWISE_PLOTTING = config["enable_plotting"]
    ENABLE_LOGGING = config["enable_logging"]

    field_type = config["field_type"]
    start_position = config["start_position"]
    action_strategy = config["action_strategy"]
    correlation_types = config["correlation_types"]
    n_steps = config["n_steps"]
    iters = config["iters"]
    error_margins = config["error_margins"]
    if action_strategy == "sweep":
        error_margins = [None]
        iters = 1

    # -----------------------------------------------------------------------------
    # Setup Grid and Field Parameters Based on Field Type
    # -----------------------------------------------------------------------------

    # desktop += f"results_{field_type.lower()}_{start_position}_trial"

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

    camera1 = camera(
        grid_info,
        60,
        rng=rng,
        camera_altitude=min_alt,
        f_overlap=overlap,
        s_overlap=overlap,
    )
    map = Field(
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
            for iter in tqdm(
                range(0, iters),
                desc=f"Iters (e={e_margin})",
                position=2,
                leave=False,
            ):

                log_folder = (
                    results_folder
                    + f"/txt/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}"
                )

                map.reset()
                ground_truth_map = map.get_ground_truth()
                # Initialize belief map with a uniform probability (0.5)
                belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
                assert ground_truth_map.shape == belief_map[:, :, 0].shape

                if e_margin is not None:
                    conf_dict = map.init_s0_s1(
                        e=e_margin,
                        sensor=use_sensor_model,
                    )
                else:
                    conf_dict = None

                occupancy_map = OM(
                    grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
                )

                planner = planning(
                    grid_info,
                    camera1,
                    action_strategy,
                    conf_dict=conf_dict,
                    optimal_alt=optimal_alt,
                )
                # Select initial UAV starting position

                if start_position == "border":
                    start_pos = random.choice(
                        [
                            (
                                -grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),  # Left border
                            (
                                grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),  # Right border
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                grid_info.y / 2,
                            ),  # Top border
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                -grid_info.y / 2,
                            ),  # Bottom border
                        ]
                    )
                elif start_position == "corner":
                    # start_pos = random.choice(
                    #    [
                    #        (-grid_info.x / 2, -grid_info.y / 2),
                    #        (-grid_info.x / 2, grid_info.y / 2),
                    #        (grid_info.x / 2, -grid_info.y / 2),
                    #        (grid_info.x / 2, grid_info.y / 2),
                    #    ]
                    # )
                    start_pos = (grid_info.x / 2, -grid_info.y / 2)
                # Initialize UAV position and list for tracking path and actions
                uav_pos = uav_position((start_pos, camera1.get_hrange()[0]))
                uav_positions, actions = [uav_pos], []
                # Update camera settings based on UAV initial state
                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)
                # Initialize observed cell ids set and metric lists
                observed_ids = set()
                entropy, mse, height, coverage = [], [], [], []
                if ENABLE_LOGGING:
                    # Initialize logger for this iteration
                    logger = FastLogger(
                        log_folder,
                        strategy=action_strategy,
                        pairwise=corr_type,
                        grid=grid_info,
                        init_x=uav_pos,
                        r=grf_r,
                        n_agent=iter,
                        e=e_margin,
                        conf_dict=conf_dict,
                    )
                # Create directory for saving step-by-step results
                os.makedirs(
                    results_folder
                    + f"/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter}/steps/",
                    exist_ok=True,
                )

                # -------------------------------------------------------------------------
                # Mapping and Planning Loop (per step)
                # -------------------------------------------------------------------------

                for step in tqdm(
                    range(0, n_steps),
                    desc=f"steps",
                    position=3,
                    leave=False,
                ):
                    # print(f"\n=== mapping {[step]} ===")
                    sigmas = None

                    if conf_dict is not None:
                        s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                        sigmas = [s0, s1]

                    fp_vertices_ij, submap = map.get_observations(
                        uav_pos,
                        sigmas,
                    )

                    observed_field_range = camera1.get_range(
                        index_form=False,
                    )
                    # Update occupancy map with new observation and propagate messages
                    occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
                    occupancy_map.propagate_messages_(
                        fp_vertices_ij, submap, uav_pos, max_iterations=1
                    )

                    # Update the belief map from the occupancy map's belief
                    belief_map[:, :, 1] = occupancy_map.get_belief().copy()
                    belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

                    # Update observed cell IDs and compute metrics
                    observed_ids.update(observed_m_ids(camera1, uav_pos))
                    entropy_val, mse_val, coverage_val = compute_metrics(
                        ground_truth_map, belief_map, observed_ids, grid_info
                    )
                    entropy.append(entropy_val)
                    mse.append(mse_val)
                    coverage.append(coverage_val)
                    height.append(uav_pos.altitude)
                    if ENABLE_LOGGING and logger is not None:
                        # Log current metrics and actions
                        logger.log_data(entropy[-1], mse[-1], height[-1], coverage[-1])
                        logger.log("actions: " + str(actions))
                    if ENABLE_STEPWISE_PLOTTING:
                        # Save metrics plot for current iteration
                        plot_metrics(
                            f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/iter_{iter}.png",
                            entropy,
                            mse,
                            coverage,
                            height,
                        )
                    # Planning: select the next action based on current belief
                    next_action, info_gain_action = planner.select_action(
                        belief_map, uav_positions
                    )

                    # Update UAV position based on the next action
                    uav_pos = uav_position(camera1.x_future(next_action))
                    actions.append(next_action)
                    uav_positions.append(uav_pos)
                    # Update camera with the new UAV state
                    camera1.set_altitude(uav_pos.altitude)
                    camera1.set_position(uav_pos.position)
                    if ENABLE_STEPWISE_PLOTTING:
                        # Plot and save the terrain visualization for this step
                        plot_terrain(
                            f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter}/steps/step_{step}.png",
                            belief_map,
                            grid_info,
                            uav_positions[0:-1],
                            ground_truth_map,
                            submap,
                            observed_field_range,
                            fp_vertices_ij,
                            camera1.get_hrange(),
                        )

                    # plot_terrain_2d(
                    #     f"{results_folder}/steps/step_{step}.png",
                    #     grid_info,
                    #     ground_truth_map,
                    # )


if __name__ == "__main__":
    main()

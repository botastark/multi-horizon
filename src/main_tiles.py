import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper import (
    FastLogger,
    compute_metrics,
    gaussian_random_field,
    init_s0_s1,
    observed_m_ids,
    uav_position,
)
from mapper import get_observations
from mapper_LBP import OccupancyMap as OML
from planner import planning
from uav_camera import camera
from viewer import plot_metrics, plot_terrain

import os

# from tile_class import TileOperations


desktop = "/home/bota/Desktop/active_sensing"
# desktop = "/Users/botaduisenbay/active_sensing"
cache_dir = desktop + "/cache/"

action_select_strategy = "ig"  # "ig", "random" "sweep" ig_with_mexgen
# correlation_types = ["biased", "equal", "adaptive"]
correlation_types = ["equal"]
grf_r = 4  # gaussian gield radius, in case of tiles use "tiles"
n_steps = 100
iters = 20
es = [0.3, 0.1, 0.05]
# es = [None]


class grid_info:
    x = 50  # 60
    y = 50  # 110 for real field
    length = 0.125  # 1
    shape = (int(x / length), int(y / length))


camera1 = camera(
    grid_info,
    60,
    #  x_range=(0, grid_info.x), y_range=(0, grid_info.y)
)
uav_pos = uav_position(((0.0, 0.0), camera1.get_hstep()))
rng = np.random.default_rng(123)
ground_truth_map = gaussian_random_field(grf_r, grid_info.shape)
# for iter in tqdm(range(iters), desc=f"Iterations", position=0):
for correlation_type in tqdm(correlation_types, desc="pairwise", position=0):
    for sampled_sigma_error_margin in tqdm(
        es, desc=f"Error Margins (pairwise = {correlation_type})", position=1
    ):
        for iter in tqdm(
            range(iters),
            desc=f"Iterations (e={sampled_sigma_error_margin})",
            position=2,
            leave=False,
        ):

            folder = (
                desktop
                + f"/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}"
            )
            if not os.path.exists(folder):
                os.makedirs(folder)

            camera1 = camera(
                grid_info,
                60,
                #  x_range=(0, grid_info.x), y_range=(0, grid_info.y)
            )
            if isinstance(grf_r, int):
                ground_truth_map = gaussian_random_field(grf_r, grid_info.shape)

                if sampled_sigma_error_margin is not None:
                    conf_dict = init_s0_s1(
                        camera1.get_hstep(), e=sampled_sigma_error_margin
                    )
                else:
                    conf_dict = None

            mapper = OML(grid_info.shape, conf_dict=conf_dict)

            planner_ = planning(
                grid_info, camera1, action_select_strategy, conf_dict=conf_dict
            )

            uav_pos = uav_position(((0, 0), camera1.get_hstep()))

            logger = FastLogger(
                folder,
                strategy=action_select_strategy,
                pairwise=correlation_type,
                grid=grid_info,
                init_x=uav_pos,
                r=grf_r,
                n_agent=iter,
                e=sampled_sigma_error_margin,
            )

            # if isinstance(grf_r, str) and grf_r == "tiles":
            #     tiles_dir = "/home/bota/Downloads/projtiles1/"
            #     gps_csv = "/home/bota/Desktop/active_sensing/data/gpstiles.csv"
            #     row_imgs_dir = "/media/bota/BOTA/wheat/APPEZZAMENTO_PICCOLO/"
            #     annotation_path = "/home/bota/Desktop/active_sensing/src/annotation.txt"
            #     tile_ops = TileOperations(tiles_dir, gps_csv, row_imgs_dir)
            #     # ground_truth_map = tile_ops.groundtruth_tiles(grid_info, cache_dir="cache")
            #     # ground_truth_map = tile_ops.gt2map(annotation_path)
            # el

            # else:
            #     print(f"{grf_r}")
            #     print(
            #         "choose correct grf_r: int - guassian radius, 'tiles' - use prediction on tiles"
            #     )

            belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
            uav_positions, past_observations, actions = [uav_pos], [], []

            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)

            obs_ms = set()
            entropy, mse, height, coverage = [], [], [], []

            for step in tqdm(
                range(0, n_steps),
                desc=f"steps",
                position=4,
                # leave=False,
            ):
                # collect observations
                x_, y_, submap = get_observations(
                    grid_info, ground_truth_map, uav_pos, rng, confidence_dict=conf_dict
                )
                # mapping

                mapper.update_belief_OG(
                    x_.T,
                    y_.T,
                    submap,
                    uav_pos,
                    # mexgen=action_select_strategy
                )
                mapper.propagate_messages_(
                    x_.T,
                    y_.T,
                    submap,
                    uav_pos,
                    max_iterations=1,
                    correlation_type=correlation_type,
                )
                # Extract the beliefs

                belief_map[:, :, 1] = mapper.get_belief().copy()
                belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

                # collect metrics, log and plot
                obs_ms.update(observed_m_ids(camera1, uav_pos))
                entropy_val, mse_val, coverage_val = compute_metrics(
                    ground_truth_map, belief_map, obs_ms, grid_info
                )

                entropy.append(entropy_val)
                mse.append(mse_val)
                coverage.append(coverage_val)
                height.append(uav_pos.altitude)

                logger.log_data(entropy[-1], mse[-1], height[-1], coverage[-1])
                logger.log("actions: " + str(actions))

                plot_metrics(desktop, entropy, mse, coverage, height)

                plot_terrain(
                    f"{folder}/step_{step}.png",
                    belief_map,
                    grid_info,
                    uav_positions,
                    ground_truth_map,
                    submap,
                    x_,
                    y_,
                )

                if step == n_steps:
                    break

                # PLAN
                next_action, _ = planner_.select_action(belief_map, uav_positions)

                # ACT
                uav_pos = uav_position(camera1.x_future(next_action))
                # print(f"next actions: {actions}")
                # print(f"next position: {uav_pos.position}-{uav_pos.altitude}")
                uav_positions.append(uav_pos)
                actions.append(next_action)

                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)

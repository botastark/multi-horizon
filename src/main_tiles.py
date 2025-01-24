from mapper import OccupancyMap, get_observations
from mapper_LBP import OccupancyMap as OML

# from mapper_LBP import get_observations as get_observations_l
import os
import numpy as np

from tile_class import TileOperations

version_to_use = "Luca"
from helper import (
    FastLogger,
    compute_metrics,
    gaussian_random_field,
    init_s0_s1,
    observed_m_ids,
    uav_position,
)

# from helper import gaussian_random_field, get_observations
from planner import planning
from uav_camera import camera
from tqdm import tqdm
from viewer import plot_terrain, plot_metrics

desktop = "/home/bota/Desktop/active_sensing"
# desktop = "/Users/botaduisenbay/active_sensing"
cache_dir = desktop + "/cache/"


correlation_type = "biased"  # "biased", "equal" "adaptive"
action_select_strategy = "ig"  # "ig", "random" "sweep" ig_with_mexgen
iters = 20  # number of iterations for mean, std plots
es = [0.3, 0.1, 0.05]
grf_r = 4  # gaussian gield radius, in case of tiles use "tiles"
n_steps = 100  # n steps for drone

for e in tqdm(es, desc="Error Margins", position=0):
    for iter in tqdm(range(iters), desc=f"Iterations (e={e})", position=1, leave=False):

        # for e in es:
        #     print(f"creating for e:{e}")
        #     for iter in range(iters):
        desktop = "/home/bota/Desktop/active_sensing"

        sampled_sigma_error_margin = e

        # desktop += f"/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_gt"
        folder = (
            desktop
            + f"/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}"
        )

        if not os.path.exists(folder):
            os.makedirs(folder)

        class grid_info:
            x = 50  # 60
            y = 50  # 110 for real field
            length = 0.125  # 1
            shape = (int(x / length), int(y / length))

        camera1 = camera(
            grid_info, 60, x_range=(0, grid_info.x), y_range=(0, grid_info.y)
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
            e=e,
        )

        if isinstance(grf_r, str) and grf_r == "tiles":
            tiles_dir = "/home/bota/Downloads/projtiles1/"
            gps_csv = "/home/bota/Desktop/active_sensing/data/gpstiles.csv"
            row_imgs_dir = "/media/bota/BOTA/wheat/APPEZZAMENTO_PICCOLO/"
            annotation_path = "/home/bota/Desktop/active_sensing/src/annotation.txt"
            tile_ops = TileOperations(tiles_dir, gps_csv, row_imgs_dir)
            # ground_truth_map = tile_ops.groundtruth_tiles(grid_info, cache_dir="cache")
            # ground_truth_map = tile_ops.gt2map(annotation_path)
            # print(ground_truth_map.shape)
        elif isinstance(grf_r, int):
            ground_truth_map = gaussian_random_field(grf_r, grid_info.shape)
            conf_dict = init_s0_s1(camera1.get_hstep(), e=sampled_sigma_error_margin)
            # print(conf_dict)

        else:
            print(f"{grf_r}")
            print(
                "choose correct grf_r: int - guassian radius, 'tiles' - use prediction on tiles"
            )

        if version_to_use == "Bota":
            mapper = OccupancyMap(grid_info.shape, conf_dict=conf_dict)
        else:
            mapper = OML(grid_info.shape, conf_dict=conf_dict)
        camera1.set_altitude(uav_pos.altitude)
        camera1.set_position(uav_pos.position)
        uav_positions, past_observations, actions = [uav_pos], [], []

        belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        planner = planning(
            belief_map,
            camera1,
            action_select_strategy,
            conf_dict=conf_dict,
        )

        obs_ms = set()
        entropy, mse, height, coverage = [], [], [], []

        for step in range(n_steps + 1):

            # print(f"step {step}")
            # collect observations
            # x_, y_, submap = get_observations(grid_info, ground_truth_map, uav_pos, seed = 0, mexgen=action_select_strategy)
            x_, y_, submap = get_observations(grid_info, ground_truth_map, uav_pos)
            zx = x_ * grid_info.length
            zy = y_ * grid_info.length
            # mapping
            if version_to_use == "Bota":
                mapper.update_observations(zx, zy, submap, uav_pos, belief_map)
                mapper.propagate_messages(
                    max_iterations=1, correlation_type=correlation_type
                )
                belief_map = mapper.marginalize()
            else:
                l_m_0 = mapper.update_belief_OG(
                    x_.T, y_.T, submap, uav_pos, mexgen=action_select_strategy
                )
                mapper.propagate_messages_(
                    x_.T,
                    y_.T,
                    submap,
                    uav_pos,
                    max_iterations=1,
                    correlation_type=correlation_type,
                )
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

            # plot_metrics(desktop, entropy, mse, coverage, height)

            # plot_terrain(
            #     f"{desktop}/step_{step}.png",
            #     belief_map,
            #     grid_info,
            #     uav_positions,
            #     ground_truth_map,
            #     submap,
            #     x_,
            #     y_,
            # )

            if step == n_steps:
                break

            # PLAN
            next_action = planner.select_action(belief_map, uav_positions)

            # ACT
            uav_pos = uav_position(camera1.x_future(next_action))
            uav_positions.append(uav_pos)
            actions.append(next_action)

            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)

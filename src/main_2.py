import os
import numpy as np

from helper import (
    FastLogger,
    compute_metrics,
    observed_m_ids,
    uav_position,
)
from orthomap import Field
import random

from mapper_LBP import OccupancyMap as OML
from planner import planning
from uav_camera import camera
from tqdm import tqdm
from viewer import plot_metrics, plot_terrain

desktop = "/home/bota/Desktop/active_sensing/results_orthomap/results_random"
belief_buffer = None

field_type = "Ortomap"
# field_type = "Gaussian"
start = "corner"  # random corner or random border
action_select_strategy = "ig"
correlation_types = ["adaptive"]
n_steps = 100
iters = 20
es = [0.3, 0.1, 0.05]
# es = [None]
if field_type == "Ortomap":
    grf_r = "orto"
    min_alt = 19.5

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

    class grid_info:
        x = 50
        y = 50
        length = 0.125
        shape = (int(y / length), int(x / length))
        center = True


# use_sensor_model = False


seed = 123
rng = np.random.default_rng(seed)


camera1 = camera(grid_info, 60, rng=rng, camera_altitude=min_alt)
map = Field(
    grid_info, field_type, sweep=action_select_strategy, h_range=camera1.get_hrange()
)

for correlation_type in tqdm(correlation_types, desc="pairwise", position=0):
    for sampled_sigma_error_margin in tqdm(
        es, desc=f"Error Margins (pairwise = {correlation_type})", position=1
    ):
        for iter in tqdm(
            range(0, iters),
            desc=f"Iters (e={sampled_sigma_error_margin})",
            position=2,
            leave=False,
        ):

            folder = (
                desktop
                + f"/txt/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}"
            )
            map.reset()
            ground_truth_map = map.get_ground_truth()
            belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
            assert ground_truth_map.shape == belief_map[:, :, 0].shape

            # min_alt = camera1.get_hstep()

            if sampled_sigma_error_margin is not None:
                conf_dict = map.init_s0_s1(
                    # camera1.get_hrange(),
                    e=sampled_sigma_error_margin,
                    sensor=use_sensor_model,
                )
                # print(conf_dict)
                # print(f"h_range: {camera1.get_hrange()}")
            else:
                conf_dict = None
            occupancy_map = OML(
                grid_info.shape, conf_dict=conf_dict, correlation_type=correlation_type
            )

            planner_mine = planning(
                grid_info,
                camera1,
                action_select_strategy,
                conf_dict=conf_dict,
                optimal_alt=min_alt,
            )
            if start == "border":
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
            elif start == "corner":
                start_pos = random.choice(
                    [
                        (-grid_info.x / 2, -grid_info.y / 2),
                        (-grid_info.x / 2, grid_info.y / 2),
                        (grid_info.x / 2, -grid_info.y / 2),
                        (grid_info.x / 2, grid_info.y / 2),
                    ]
                )
                # start_pos = (-grid_info.x / 2, -grid_info.y / 2)

            uav_pos = uav_position((start_pos, camera1.get_hrange()[0]))
            uav_positions, actions_bota = [uav_pos], []

            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)
            obs_ms = set()
            entropy, mse, height, coverage = [], [], [], []
            # if conf_dict is not None:
            # print(conf_dict)

            logger = FastLogger(
                folder,
                strategy=action_select_strategy,
                pairwise=correlation_type,
                grid=grid_info,
                init_x=uav_pos,
                r=grf_r,
                n_agent=iter,
                e=sampled_sigma_error_margin,
                conf_dict=conf_dict,
            )

            os.makedirs(
                desktop
                + f"/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}/{iter}/steps/",
                exist_ok=True,
            )

            # for step in range(0, n_steps):
            for step in tqdm(
                range(0, n_steps),
                desc=f"steps",
                position=3,
                leave=False,
            ):
                # print(f"\n=== mapping {[step]} ===")
                # current_pos = camera1.get_x()
                sigmas = None

                if conf_dict is not None:
                    s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                    sigmas = [s0, s1]

                fp_vertices_ij, submap = map.get_observations(
                    uav_pos,
                    sigmas,
                )

                obd_field = camera1.get_range(
                    index_form=False,
                )

                occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
                occupancy_map.propagate_messages_(
                    fp_vertices_ij, submap, uav_pos, max_iterations=1
                )

                # Extract the beliefs

                belief_map[:, :, 1] = occupancy_map.get_belief().copy()
                belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

                # Plan
                obs_ms.update(observed_m_ids(camera1, uav_pos))
                entropy_val, mse_val, coverage_val = compute_metrics(
                    ground_truth_map, belief_map, obs_ms, grid_info
                )
                entropy.append(entropy_val)
                mse.append(mse_val)
                coverage.append(coverage_val)
                height.append(uav_pos.altitude)
                logger.log_data(entropy[-1], mse[-1], height[-1], coverage[-1])
                logger.log("actions: " + str(actions_bota))

                plot_metrics(
                    f"{desktop}/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}/iter_{iter}.png",
                    entropy,
                    mse,
                    coverage,
                    height,
                )

                next_action, info_gain_action = planner_mine.select_action(
                    belief_map, uav_positions
                )

                # ACT

                uav_pos = uav_position(camera1.x_future(next_action))

                actions_bota.append(next_action)
                uav_positions.append(uav_pos)

                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)

                plot_terrain(
                    f"{desktop}/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}/{iter}/steps/step_{step}.png",
                    belief_map,
                    grid_info,
                    uav_positions[0:-1],
                    ground_truth_map,
                    submap,
                    obd_field,
                    fp_vertices_ij,
                )
                # exit()

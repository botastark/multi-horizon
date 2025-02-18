import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from helper import (
    FastLogger,
    compute_metrics,
    gaussian_random_field,
    init_s0_s1,
    observed_m_ids,
    uav_position,
)

# from mapper import get_observations
from mapper_LBP import OccupancyMap as OML
from planner import planning
from uav_camera import camera
from tqdm import tqdm
from viewer import plot_metrics, plot_terrain

desktop = "/home/bota/Desktop/active_sensing/cache"
belief_buffer = None
"""

from simulator import MappingEnv, Mapper, Agent, State, Camera, Proximity
from simulator import Planner, Viewer


# Parameters for testing
params = dict(
    a0=1.0,
    a1=1.0,
    b0=0.015,
    b1=0.015,
    inference_type="LBP_cts_vectorized",
    cluster_radius=4,
    news_inference_type="LBP_single",
    map_type="gaussian",
    planner_type="selfish",
    env_type="adhoc",
    altitude=0,
    n_agents=1,
    centralized=False,
    n_runs=1,
    n_steps=100,
    render=True,
    weights_type="equal",
    p_eq=0.5,
)
field_len = 50.0
sim_env = MappingEnv(field_len=field_len, fov=np.pi / 3, **params)
ground_truth_map_lucas = sim_env.generate_map()
Initialize the simulator's Mapper
sim_mapper = Mapper(
    n_cell=sim_env.n_cell,
    min_space_z=sim_env.min_space_z,
    max_space_z=sim_env.max_space_z,
    **params,
)
planner_luca = Planner(
    sim_env.action_to_direction,
    sim_env.altitude_to_size,
    sim_env.position_graph,
    sim_env.position_to_data,
    sim_env.regions_limits,
    sim_env.optimal_altitude,
    **params,
)

"""


class grid_info:
    x = 50  # 60
    y = 50  # 110 for real field
    length = 0.125  # 1
    shape = (int(y / length), int(x / length))
    center = True


grf_r = 4
# correlation_types = ["biased", "equal", "adaptive"]
correlation_types = ["equal"]
n_steps = 100
iters = 1
# es = [None, 0.3, 0.1, 0.05]
es = [0.3]
rng = np.random.default_rng(123)

# Initialize the mapper's OccupancyMap
action_select_strategy = "ig"

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
                + f"/txt_new/{correlation_type}_{action_select_strategy}_e{sampled_sigma_error_margin}_r{grf_r}"
            )
            ground_truth_map = gaussian_random_field(grf_r, grid_info.shape)

            belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
            assert ground_truth_map.shape == belief_map[:, :, 0].shape
            camera1 = camera(grid_info, 60, rng=rng)

            if sampled_sigma_error_margin is not None:
                conf_dict = init_s0_s1(
                    camera1.get_hstep(), e=sampled_sigma_error_margin
                )
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
            )

            uav_pos = uav_position(((0.0, 0.0), camera1.get_hstep()))
            uav_positions, actions_bota = [uav_pos], []

            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)
            obs_ms = set()
            entropy, mse, height, coverage = [], [], [], []

            logger = FastLogger(
                desktop,
                strategy=action_select_strategy,
                pairwise=correlation_type,
                grid=grid_info,
                init_x=uav_pos,
                r=grf_r,
                n_agent=iter,
                e=sampled_sigma_error_margin,
            )
            # for step in range(0, n_steps):
            for step in tqdm(
                range(0, n_steps),
                desc=f"steps",
                position=3,
                leave=False,
            ):
                # print(f"\n=== mapping {[step]} ===")
                current_pos = camera1.get_x()
                sigmas = None

                if conf_dict is not None:
                    s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                    sigmas = [s0, s1]

                fp_vertices_ij, submap = camera1.get_observations(
                    ground_truth_map,
                    sigmas,
                )
                obd_field = camera1.get_range(
                    index_form=False,
                )
                [[x_min, x_max], [y_min, y_max]] = obd_field

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
                    f"{desktop}/iter_{iter}.png", entropy, mse, coverage, height
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
                    f"{desktop}/step_{step}.png",
                    belief_map,
                    grid_info,
                    uav_positions[0:-1],
                    ground_truth_map,
                    submap,
                    obd_field,
                    fp_vertices_ij,
                )

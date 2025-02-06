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

desktop = "/home/bota/Desktop/active_sensing/cache/"
action_select_strategy = "ig"
correlation_type = "equal"
grf_r = 4
sampled_sigma_error_margin = None
n_steps = 100
iters = 1


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


for iter in tqdm(range(iters), desc=f"Iterations", position=0):
    ground_truth_map = gaussian_random_field(grf_r, grid_info.shape)
    camera1 = camera(
        grid_info,
        60,
        #  x_range=(0, grid_info.x), y_range=(0, grid_info.y)
    )
    mapper = OML(grid_info.shape)
    planner_ = planning(
        grid_info,
        camera1,
        action_select_strategy,
    )

    belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)

    uav_positions, actions = [uav_pos], []

    camera1.set_altitude(uav_pos.altitude)
    camera1.set_position(uav_pos.position)
    obs_ms = set()
    entropy, mse, height, coverage = [], [], [], []
    rng = np.random.default_rng()
    for step in tqdm(
        range(0, n_steps),
        desc=f"steps",
        position=1,
        # leave=False,
    ):
        # Generate observations
        #     s0, s1 = self.conf_dict[np.round(x_future.altitude, decimals=2)]

        x_, y_, submap = get_observations(
            grid_info,
            ground_truth_map,
            uav_pos,
            rng,
            #   center=True
        )

        # Update the OccupancyMap with observations

        mapper.update_belief_OG(x_.T, y_.T, submap, uav_pos)
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
        logger.log("actions: " + str(actions))

        plot_metrics(desktop, entropy, mse, coverage, height)

        next_action, info_gain_action = planner_.select_action(
            belief_map, uav_positions
        )

        # print(f"next actions: {actions}")
        # print(f"next action: {info_gain_action}")

        # ACT
        actions.append(next_action)
        uav_pos = uav_position(camera1.x_future(next_action))

        uav_positions.append(uav_pos)

        camera1.set_altitude(uav_pos.altitude)
        camera1.set_position(uav_pos.position)
        plot_terrain(
            f"{desktop}/step_{step}.png",
            belief_map,
            grid_info,
            uav_positions,
            ground_truth_map,
            submap,
            x_,
            y_,
        )

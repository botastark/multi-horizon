import numpy as np
from helper import (
    compute_coverage,
    observed_m_ids,
    plot_metrics,
    uav_position,
    compute_mse,
)
from terrain_creation import (
    fft_gaussian_random_field,
    terrain,
)
from uav_camera import camera
from planner import planning

desktop = "/home/bota/Desktop/step_"
action_select_strategy = "sweep"  # "ig", "random" "sweep"


class grid_info:
    x = 50
    y = 50
    length = 0.5
    shape = (int(x / length), int(y / length))


class camera_params:
    fov_angle = 60


camera = camera(grid_info, camera_params.fov_angle)

# Ground truth map with n gaussian peaks
true_map = terrain(grid_info)
true_map.set_map(fft_gaussian_random_field(true_map, 10))

true_map.plot_map(desktop + "gt.png", fit=False)
plan1 = planning(true_map, camera)

actions = []
uav_positions = []
entropies = []
mse = []
coverage = []

x = uav_position(((0, 0), 5.4))
obs_ms = set()
for step in range(300):
    print("step ", step)
    # Observe
    uav_positions.append(x)
    camera.set_altitude(x.altitude)
    camera.set_position(x.position)
    z = camera.sample_observation(true_map, x)
    # print(np.max(z.x))

    # Map
    plan1.mapping(x, z)
    entropies.append(plan1.get_entropy())
    current_state = plan1.get_current_state()
    mse.append(compute_mse(true_map.map, current_state.map))
    obs_ms.update(observed_m_ids(camera, x))
    coverage.append(compute_coverage(list(obs_ms), grid_info))

    # Plan
    next_action = plan1.select_action(strategy=action_select_strategy)

    # Act
    # x_{t+1} UAV position after taking action a
    x = uav_position(camera.x_future(next_action))
    actions.append(next_action)

    # Check plots
    if step % 25 == 0:
        current_x, current_z = plan1.get_last_observation()

        current_z.plot_prob(desktop + str(step) + "_prob_z.png")
        filename = desktop + str(step) + ".png"
        current_state.plot_prob(desktop + str(step) + "_prob.png")
        current_state.plot_terrain(filename, uav_positions, true_map, current_z)


plot_metrics(entropies, mse, coverage)

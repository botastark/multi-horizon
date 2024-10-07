from terrain_creation import terrain, generate_n_peaks
from uav_camera import camera
from planner import planning


class grid_info:
    x = 50
    y = 50
    length = 0.75
    shape = (int(x / length), int(y / length))


class camera_params:
    fov_angle = 60


curr_grid = grid_info

# Belief map that init with all 0.5
belief_map = terrain(curr_grid)

# Ground truth map with n gaussian peaks
n_peaks = 100
true_map = terrain(curr_grid)
z = generate_n_peaks(n_peaks, true_map)
true_map.set_map(z)
# true_map.plot_map(fit= False)


# Get observation at certain pos and altitude
class uav_:
    position = (0, 15)
    altitude = 10


uav = uav_

camera = camera(grid_info, camera_params.fov_angle)
camera.set_altitude(uav_.altitude)
camera.set_position(uav_.position)
# zo, xo, yo = camera.get_observation(true_map.get_map())

# obs_map = terrain(grid_info)
# obs_map.set_map(zo, x=xo, y=yo)
# obs_map.plot_map(fit = True)

# # Add Gaussian noise dep on altitude to observation z
# noisy_obs = prob_sensor_model(obs_map.get_map(), uav_altitude)
# noisy_obs_map = obs_map
# noisy_obs_map.set_map(noisy_obs, x=xo, y=yo)
# noisy_obs_map.plot_map(fit = True)
plan1 = planning()

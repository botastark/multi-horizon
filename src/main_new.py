from mapper import OccupancyMap, adapt_observations, get_observations
from helper import uav_position
from terrain_creation import gaussian_random_field, terrain
from planner_new import planning
from uav_camera import camera
import numpy as np

from viewer import plot_terrain

desktop = "/home/bota/Desktop/"
cache_dir = desktop + "active_sensing/cache/"


class grid_info:
    x = 20
    y = 20
    length = 0.125
    shape = (int(x / length), int(y / length))


ground_truth_map = gaussian_random_field(4, grid_info.shape[0])
mapper = OccupancyMap(grid_info.shape[0])
camera = camera(grid_info, 60)
uav_pos = uav_position(((0, 0), 5.4))

camera.set_altitude(uav_pos.altitude)
camera.set_position(uav_pos.position)
uav_positions, past_observations, actions = [uav_pos], [], []
n_steps = 10


belief_map = mapper.marginalize()
planner = planning(belief_map, camera)


# Ground truth map with n gaussian peaks
true_map = terrain(grid_info)
# true_map.set_map(gen_fast(true_map, 5))
true_map.set_map(ground_truth_map)
current_state = true_map.copy()
current_z = true_map.copy()

for step in range(n_steps):
    # collect observations
    zx, zy, submap = get_observations(grid_info, ground_truth_map, uav_pos)
    observations = adapt_observations(zx, zy, submap, grid_info)
    # mapping
    mapper.update_observations(observations, uav_pos, belief_map)
    mapper.set_last_observations(submap)
    mapper.propagate_messages(max_iterations=1, correlation_type="adaptive")
    belief_map = mapper.marginalize()

    # uav_pos = uav_position(((5, 5+step), 10.2))
    # uav_pos.position = (uav_pos.position[0], uav_pos.position[1]+1)
    # print("Marginal at (5, 5):", belief_map[5, 5])
    # print(
    #     f"Marginal at uav pos {uav_pos.position}:{belief_map[uav_pos.position[0], uav_pos.position[1]]}"
    # )

    # plan next action
    next_action = planner.select_action(belief_map, uav_positions)
    # act
    current_state.set_map(belief_map[:, :, 1])

    current_z.set_map(
        submap,
        x=zx,
        y=zy,
    )

    # current_z.plot_prob(desktop + str(step) + "_prob_z.png")
    filename = desktop + "step_" + str(step) + ".png"
    # current_state.plot_prob(desktop + "_prob_step" + str(step) + ".png")
    # current_state.plot_terrain(filename, uav_positions, true_map, current_z)
    plot_terrain(filename, belief_map, grid_info, uav_positions, true_map, current_z)

    uav_pos = uav_position(camera.x_future(next_action))
    uav_positions.append(uav_pos)
    actions.append(next_action)

    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)

print(actions)
"""
checks
print(marginals.shape)
print("Marginal at (0, 3):", marginals[0, 2])
print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 20):", marginals[10, 20, 1])
"""

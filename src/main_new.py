from mapper import OccupancyMap, get_observations
from helper import uav_position
from terrain_creation import gaussian_random_field
from planner_new import planning
from uav_camera import camera


class grid_info:
    x = 21
    y = 21
    length = 1
    shape = (int(x / length), int(y / length))


ground_truth_map = gaussian_random_field(4, grid_info.shape[0])
mapper = OccupancyMap(grid_info.shape[0])
camera = camera(grid_info, 60)
uav_pos = uav_position(((0, 0), 5.4))

camera.set_altitude(uav_pos.altitude)
camera.set_position(uav_pos.position)
uav_positions, past_observations, actions = [uav_pos], [], []
n_steps = 2


belief_map = mapper.marginalize()
planner = planning(belief_map, camera)

for step in range(n_steps):
    # collect observations
    submap, observations = get_observations(grid_info, ground_truth_map, uav_pos)
    # mapping
    mapper.update_observations(observations, uav_pos, belief_map)
    mapper.set_last_observations(submap)
    mapper.propagate_messages(max_iterations=2, correlation_type="adaptive")
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
    uav_pos = uav_position(camera.x_future(next_action))
    uav_positions.append(uav_pos)
    actions.append(next_action)
    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)


"""
checks
print(marginals.shape)
print("Marginal at (0, 3):", marginals[0, 2])
print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 20):", marginals[10, 20, 1])
"""

from mapper import OccupancyMap, get_observations
from helper import uav_position
from terrain_creation import gaussian_random_field

class grid_info:
    x = 21
    y = 21
    length = 1
    shape = (int(x / length), int(y / length))

ground_truth_map = gaussian_random_field(4, grid_info.shape[0])
mapper = OccupancyMap(grid_info.shape[0])

uav_positions = []
n_steps =  50


prev_obs = []
uav_pos = uav_position(((5, 5), 10.2))
belief_map = mapper.marginalize()

for step in range(n_steps):
    # collect observations
    observations = get_observations(grid_info, ground_truth_map, uav_pos)
    #mapping
    # for i,j,z in observations:
    #     if i==5 and j==5:
    #         print(f"(5,5) ->{z}")
    mapper.update_observations(observations, uav_pos, belief_map)
    mapper.propagate_messages(max_iterations=2, correlation="equal")
    belief_map = mapper.marginalize()
    # uav_pos = uav_position(((5, 5+step), 10.2))
    # uav_pos.position = (uav_pos.position[0], uav_pos.position[1]+1)
    print("Marginal at (5, 5):", belief_map[5, 5])
    print(f"Marginal at uav pos {uav_pos.position}:{belief_map[uav_pos.position[0], uav_pos.position[1]]}")

"""
checks
print(marginals.shape)
print("Marginal at (0, 3):", marginals[0, 2])
print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 20):", marginals[10, 20, 1])
"""


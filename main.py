from matplotlib import pyplot as plt
from helper import observed_m_ids, uav_position
from terrain_creation import (
    generate_correlated_gaussian_field,
    terrain,
)
from uav_camera import camera
from planner import planning


class grid_info:
    x = 50
    y = 50
    length = 0.5
    shape = (int(x / length), int(y / length))


class camera_params:
    fov_angle = 60


uav_init = uav_position(((0, 0), 5.4))

camera = camera(grid_info, camera_params.fov_angle)
camera.set_altitude(uav_init.altitude)
camera.set_position(uav_init.position)

# Ground truth map with n gaussian peaks
true_map = terrain(grid_info)
true_map_m = generate_correlated_gaussian_field(true_map, 20)
true_map.set_map(true_map_m)
desktop = "/home/bota/Desktop/step_"
true_map.plot_map(desktop + "gt.png", fit=False)
plan1 = planning(true_map, camera)


actions = []
current_x = uav_position(plan1.get_uav_current_pos())
uav_positions = [current_x]
entropies = []
for step in range(10):

    entropies.append(plan1.get_entropy())

    # Act
    next_action = plan1.select_action()
    plan1.take_action(next_action, true_map)
    actions.append(next_action)

    # collect observations
    current_x, current_z = plan1.get_last_observation()
    uav_positions.append(current_x)
    current_z.plot_prob(desktop + str(step) + "_prob_z.png")

    # plot current state
    current_state = plan1.get_current_state()
    filename = desktop + str(step) + ".png"
    current_state.plot_prob(desktop + str(step) + "_prob.png")

    current_state.plot_terrain(filename, uav_positions, true_map, current_z)


plt.figure(figsize=(8, 6))
steps = list(range(len(entropies)))
plt.plot(steps, entropies, marker="o", color="b", label="Entropy")

# Set axis labels and title
plt.xlabel("Step (integer)")
plt.ylabel("Entropy (double)")
plt.title("Entropy Over Steps")

# Add a legend
plt.legend()

# Display the plot
plt.grid(True)  # Optional: add grid lines for better visualization
plt.show()

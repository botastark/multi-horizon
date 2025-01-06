import numpy as np
from src.helper import uav_position
import matplotlib.pyplot as plt
from src.mapper import OccupancyMap
# from src.mapper_LBP import OccupancyMap
from simulator import MappingEnv, Mapper, Agent, State, Camera, Proximity

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
    weights_type="adaptive",
    p_eq=0.5,
)
belief_buffer = None
# Simulation setup
np.random.seed(42)  # Set a random seed for reproducibility
field_len = 50.0

sim_env = MappingEnv(field_len=field_len, fov=np.pi / 3, **params)

ground_truth_map = sim_env.generate_map()

# Initialize the simulator's Mapper
sim_mapper = Mapper(
    n_cell=sim_env.n_cell,
    min_space_z=sim_env.min_space_z,
    max_space_z=sim_env.max_space_z,
    **params
)
# Initialize the mapper's OccupancyMap
occupancy_map = OccupancyMap(n_cells=sim_env.n_cell)

grid_length = field_len / sim_env.n_cell
occupancy_beliefs = np.ones((sim_env.n_cell, sim_env.n_cell, 2))*0.5
actions = ["init","up", "up", "up", "up","down", "right", "front","right", "front","right", "front","right", "front", "right", "down","up","down", "right", "front", "right", "down","left", "left", "back","left","back", "left","left", "left","left", "left","down" ]
for step in range(1, len(actions)):
    print(f"\n=== mapping {actions[step-1]} ===")
    
    pos = sim_env.agents[0].state.position
    print(f"uav position {pos}")
    uav_pos = uav_position(((pos[0], pos[1]), pos[2]))

    # Generate observations
    observations = sim_env.get_observations(ground_truth_map)

    # adapt observations for occupancy map
    first_observation = observations[0]

    fp_ij = first_observation["fp_ij"]
    y = np.arange(fp_ij["ul"][1], fp_ij["ur"][1])   # Horizontal indices
    x = np.arange(fp_ij["ul"][0], fp_ij["bl"][0])  # Vertical indices
    x_, y_ = np.meshgrid(x, y, indexing='ij')
    submap = first_observation["z"]

    

    # Update the OccupancyMap with observations
    correlation_type = params["weights_type"]
    occupancy_map.update_observations(x_, y_, submap, uav_pos, occupancy_beliefs)
    occupancy_map.propagate_messages(fp_ij, max_iterations=1, correlation_type=correlation_type)
    my_msgs = occupancy_map.msgs_change()
    # l_m_0 = occupancy_map.update_belief_OG(x_.T, y_.T, submap, uav_pos)
    # occupancy_map.propagate_messages_( x_.T, y_.T, submap, uav_pos,  max_iterations=1, correlation_type=correlation_type)

    # Update the simulator's mapper beliefs
    if correlation_type == "equal":
        sim_mapper.set_pairwise_potential_h(sim_env.agents)
    elif correlation_type == "adaptive":
        sim_mapper.set_pairwise_potential_z(sim_env.agents, observations)


    sim_mapper.update_belief_OG( observations, sim_env.agents)
    sim_mapper.update_map_beliefs(sim_env.agents, observations)
    sim_mapper.update_news_and_fuse_map_beliefs(sim_env.agents, observations)
    his_msgs = sim_mapper.get_msgs()
    assert my_msgs.shape == his_msgs.shape
    # l_m_0_ = sim_mapper.get_map_beliefs()[:, :, 0]
    # diff = l_m_0 - l_m_0_
    # diff = np.array(diff).flatten()
    # print(f"diff {np.unique(diff)}")
    

    # Extract the beliefs
    occupancy_beliefs = occupancy_map.marginalize(fp_ij)
    # occupancy_beliefs = occupancy_map.get_belief()
    sim_beliefs = sim_mapper.get_map_beliefs()[:, :, 0].copy()

    # Compare results
    print("\n=== Comparing Beliefs ===")
    if occupancy_beliefs.ndim==3:
        belief_diff = np.abs(occupancy_beliefs[:,:,1] - sim_beliefs)
    else:
        belief_diff = np.abs(occupancy_beliefs[:,:] - sim_beliefs)

    print("Max difference between beliefs:", np.max(belief_diff))
    print("Mean difference between beliefs:", np.mean(belief_diff))

    sim_env.step([actions[step]])
    




    # Optionally visualize the differences
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.title("OccupancyMap Beliefs")
    if occupancy_beliefs.ndim==3:
        plt.imshow(occupancy_beliefs[:, :, 1], cmap="viridis")  # Probability of "occupied"
    else:
        plt.imshow(occupancy_beliefs[:, :], cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Simulator Mapper Beliefs")
    plt.imshow(sim_beliefs, cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Difference")
    plt.imshow(belief_diff, cmap="inferno")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("ground truth")
    plt.imshow(submap, cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    # break



    # msgs difference
    diff = my_msgs-his_msgs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.title("down")
    plt.imshow(diff[0,:, :], cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("up")
    plt.imshow(diff[2,:, :], cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("left")
    plt.imshow(diff[1,:, :], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("right")
    plt.imshow(diff[3,:, :], cmap="viridis")  # Probability of "occupied"
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    # break



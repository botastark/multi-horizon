import numpy as np
import matplotlib.pyplot as plt

from helper import (
    FastLogger,
    compute_metrics,
    gaussian_random_field,
    # get_observations,
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

# from simulator import MappingEnv, Mapper, Agent, State, Camera, Proximity
# from simulator import Planner, Viewer


# Parameters for testing
# params = dict(
#     a0=1.0,
#     a1=1.0,
#     b0=0.015,
#     b1=0.015,
#     inference_type="LBP_cts_vectorized",
#     cluster_radius=4,
#     news_inference_type="LBP_single",
#     map_type="gaussian",
#     planner_type="selfish",
#     env_type="adhoc",
#     altitude=0,
#     n_agents=1,
#     centralized=False,
#     n_runs=1,
#     n_steps=100,
#     render=True,
#     weights_type="equal",
#     p_eq=0.5,
# )
# field_len = 50.0
# sim_env = MappingEnv(field_len=field_len, fov=np.pi / 3, **params)
# ground_truth_map_lucas = sim_env.generate_map()
# Initialize the simulator's Mapper
# sim_mapper = Mapper(
#     n_cell=sim_env.n_cell,
#     min_space_z=sim_env.min_space_z,
#     max_space_z=sim_env.max_space_z,
#     **params,
# )
# planner_luca = Planner(
#     sim_env.action_to_direction,
#     sim_env.altitude_to_size,
#     sim_env.position_graph,
#     sim_env.position_to_data,
#     sim_env.regions_limits,
#     sim_env.optimal_altitude,
#     **params,
# )


class grid_info:
    x = 50  # 60
    y = 50  # 110 for real field
    length = 0.125  # 1
    shape = (int(x / length), int(y / length))
    center = True


grf_r = 4
# correlation_types = ["biased", "equal", "adaptive"]
correlation_types = ["equal"]
n_steps = 100
iters = 1
# es = [None, 0.3, 0.1, 0.05]
es = [None]
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

            # uav_pos = uav_position(((grid_info.x / 2, grid_info.y / 2), camera1.get_hstep()))
            uav_pos = uav_position(((0.0, 0.0), camera1.get_hstep()))
            # pos = sim_env.agents[0].state.position
            uav_positions, actions_bota = [uav_pos], []

            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)
            # count_disagree = 0
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
                position=4,
                # leave=False,
            ):
                # print(f"\n=== mapping {[step]} ===")

                # pos = sim_env.agents[0].state.position
                # uav_pos = uav_position(((pos[0], pos[1]), pos[2]))
                # observations = sim_env.get_observations(ground_truth_map)

                # adapt observations for occupancy map
                # first_observation = observations[0]

                # fp_ij = first_observation["fp_ij"]
                # y = np.arange(fp_ij["ul"][1], fp_ij["ur"][1])  # Horizontal indices
                # x = np.arange(fp_ij["ul"][0], fp_ij["bl"][0])  # Vertical indices
                # x_, y_ = np.meshgrid(x, y, indexing="ij")
                # submap = first_observation["z"]

                # s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                # sigmas = [s0, s1]
                x__, y__, submap_ = camera1.get_observations(
                    ground_truth_map,
                    # sigmas,
                )

                obd_field = camera1.get_range(
                    position=uav_pos.position,
                    altitude=uav_pos.altitude,
                    index_form=False,
                )

                # print(f"diff obs z: {np.max(submap_-submap)} {np.min(submap_-submap)}")
                # plt.figure(figsize=(12, 5))
                # plt.subplot(1, 2, 1)
                # plt.title("obs from get obs")
                # plt.imshow(submap_, cmap="viridis")  # Probability of "occupied"
                # plt.colorbar()

                # plt.subplot(1, 2, 2)
                # plt.title("obs from env")
                # plt.imshow(submap, cmap="viridis")  # Probability of "occupied"
                # plt.colorbar()
                # plt.tight_layout()
                # plt.show()

                # first_observation["z"] = submap_
                # observations[0] = first_observation

                # plt.figure(figsize=(12, 5))
                # plt.subplot(1, 3, 1)
                # plt.title("submap_ ")
                # plt.imshow(submap_, cmap="viridis")  # Probability of "occupied"
                # plt.colorbar()

                # plt.subplot(1, 3, 2)
                # plt.title("submap")
                # plt.imshow(submap, cmap="viridis")  # Probability of "occupied"
                # plt.colorbar()

                # plt.subplot(1, 3, 3)
                # plt.title("Difference")
                # plt.imshow(submap_ - submap, cmap="inferno")
                # plt.colorbar()

                # plt.tight_layout()
                # plt.show()

                occupancy_map.update_belief_OG(x__.T, y__.T, submap_, uav_pos)
                occupancy_map.propagate_messages_(
                    x__.T, y__.T, submap_, uav_pos, max_iterations=1
                )

                # # Update the simulator's mapper beliefs
                # if correlation_type == "equal":
                #     sim_mapper.set_pairwise_potential_h(sim_env.agents)
                # elif correlation_type == "adaptive":
                #     sim_mapper.set_pairwise_potential_z(sim_env.agents, observations)

                # sim_mapper.update_belief_OG(observations, sim_env.agents)
                # sim_mapper.update_news_and_fuse_map_beliefs(sim_env.agents, observations)
                # Extract the beliefs

                belief_map[:, :, 1] = occupancy_map.get_belief().copy()
                belief_map[:, :, 0] = 1 - belief_map[:, :, 1]
                # # sim_beliefs = sim_mapper.get_map_beliefs()[:, :, 0].copy()

                # # Compare results
                # print("\n=== Comparing Beliefs ===")
                # belief_diff = np.abs(belief_map[:, :, 1] - sim_beliefs)

                # print("Max difference between beliefs:", np.max(belief_diff))
                # print("Mean difference between beliefs:", np.mean(belief_diff))

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

                # planner_luca.reset_sweep()
                # planner_luca.compute_map_belief_entropies()

                # actions_luca, actions_data = planner_luca.get_actions(sim_env.agents, observations)

                next_action, info_gain_action = planner_mine.select_action(
                    belief_map, uav_positions
                )

                """
                # checking planner:

                if actions_luca[0] == next_action:
                    print(f"agreed: {next_action}")
                else:
                    print()
                    print("_______DISAGREEEE!!!!_________")
                    count_disagree += 1
                    print(f"luca's next actions {actions_luca}")
                    print(f"bota's next actions {next_action}")
                lucas_actions_info = actions_data[0]["admissible_action_to_IG"]
                actions_bota.append(next_action)
                eligible_actions = set()
                eligible_actions.update(list(actions_data[0]["admissible_action_to_IG"].keys()))
                eligible_actions.update(list(info_gain_action.keys()))
                for action in list(eligible_actions):
                    if action in lucas_actions_info and action in info_gain_action:
                        if round(lucas_actions_info[action][0], 1) != round(
                            info_gain_action[action], 1
                        ):
                            print(
                                f"ig for action {action} are different: l-{round(lucas_actions_info[action][0],2)}, b - {round(info_gain_action[action], 2)}"
                            )
                    elif not action in lucas_actions_info and action in info_gain_action:
                        print(f"diff: action {action} is not in lucas")
                    else:
                        print(f"diff: action {action} is not in botas")
                """

                # ACT

                # sim_env.step(actions_luca)
                uav_pos = uav_position(camera1.x_future(next_action))
                # pos = sim_env.agents[0].state.position

                actions_bota.append(next_action)

                # uav_pos = uav_position(((pos[0], pos[1]), pos[2]))

                uav_positions.append(uav_pos)

                # uav_pos = uav_position(camera1.x_future(next_action))
                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)

                plot_terrain(
                    f"{desktop}/step_{step}.png",
                    belief_map,
                    grid_info,
                    uav_positions,
                    ground_truth_map,
                    submap_,
                    obd_field,
                    centered=True,
                )

                # Optionally visualize the differences
                """
                # checking mappers

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 4, 1)
                plt.title("OccupancyMap Beliefs")
                plt.imshow(belief_map[:, :, 1], cmap="viridis")  # Probability of "occupied"
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
                """

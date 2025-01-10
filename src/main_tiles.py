from mapper import OccupancyMap, get_observations
from mapper_LBP import OccupancyMap as OML
# from mapper_LBP import get_observations as get_observations_l
import timeit
import os
import numpy as np

from tile_class import TileOperations

version_to_use = "Luca"
from helper import (
    FastLogger,
    compute_metrics,
    gaussian_random_field,
    observed_m_ids,
    uav_position,
)
# from helper import gaussian_random_field, get_observations
from planner import planning
from uav_camera import camera

from viewer import plot_terrain, plot_metrics
# from tiles import init_tiles, observed_submap

desktop = "/home/bota/Desktop/active_sensing"
# desktop = "/Users/botaduisenbay/active_sensing"

cache_dir = desktop + "/cache/"
correlation_type = "equal"  # "biased", "equal" "adaptive"
action_select_strategy = "ig"  # "ig", "random" "sweep" ig_with_mexgen
desktop += f"/{correlation_type}_{action_select_strategy}"


n_steps = 100
grf_r = "tiles"
if not os.path.exists(desktop):
    os.makedirs(desktop)

class grid_info:
    x = 60
    y = 110
    length = 1
    shape = (int(x / length), int(y / length))

camera = camera(grid_info, 60,x_range=(0, grid_info.x), y_range=(0, grid_info.y))
uav_pos = uav_position(((0, 0), camera.get_hstep()))

logger = FastLogger(
    desktop,
    strategy=action_select_strategy,
    pairwise=correlation_type,
    grid=grid_info,
    init_x=uav_pos,
    r=grf_r,
)

if isinstance(grf_r, str) and grf_r=="tiles":
    tiles_dir = '/home/bota/Downloads/projtiles1/'
    gps_csv = '/home/bota/Desktop/active_sensing/src/gpstiles.csv'
    row_imgs_dir = "/media/bota/BOTA/wheat/APPEZZAMENTO_PICCOLO/"
    tile_ops = TileOperations(tiles_dir, gps_csv, row_imgs_dir)
    ground_truth_map = tile_ops.groundtruth_tiles(grid_info, cache_dir="cache")
elif isinstance(grf_r, int):
    ground_truth_map = gaussian_random_field(grf_r, grid_info.shape[0])
else:
    print(f"{grf_r}")
    print("choose correct grf_r: int - guassian radius, 'tiles' - use prediction on tiles")



if version_to_use=="Bota":
    mapper = OccupancyMap(grid_info.shape)
else:
    mapper = OML(grid_info.shape)
camera.set_altitude(uav_pos.altitude)
camera.set_position(uav_pos.position)
uav_positions, past_observations, actions = [uav_pos], [], []


belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
planner = planning(belief_map, camera, action_select_strategy)

obs_ms = set()
entropy, mse, height, coverage = [], [], [], []

for step in range(n_steps + 1):
    
    print(f"step {step}")
    # collect observations
    # x_, y_, submap = get_observations(grid_info, ground_truth_map, uav_pos, seed = 0, mexgen=action_select_strategy)
    x_, y_, submap = get_observations(grid_info, ground_truth_map, uav_pos)
    zx = x_*grid_info.length
    zy = y_*grid_info.length
    # mapping
    if version_to_use=="Bota":
        mapper.update_observations(zx, zy, submap, uav_pos, belief_map)
        mapper.propagate_messages(max_iterations=1, correlation_type=correlation_type)
        belief_map = mapper.marginalize()
    else:
        l_m_0 = mapper.update_belief_OG(x_.T, y_.T, submap, uav_pos, mexgen = action_select_strategy)
        mapper.propagate_messages_( x_.T, y_.T, submap, uav_pos,  max_iterations=1, correlation_type=correlation_type)
        belief_map[:,:, 1] = mapper.get_belief().copy()
        belief_map[:,:, 0] = 1-belief_map[:,:, 1]

    # collect metrics, log and plot
    obs_ms.update(observed_m_ids(camera, uav_pos))
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

    if step == n_steps:
        break

    # PLAN
    next_action = planner.select_action(belief_map, uav_positions)

    # ACT
    uav_pos = uav_position(camera.x_future(next_action))
    uav_positions.append(uav_pos)
    actions.append(next_action)

    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)
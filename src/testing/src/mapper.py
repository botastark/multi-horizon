import numpy as np
import math


def convert_xy_ij(x, y, grid, centered):
    if centered:
        center_j, center_i = (dim // 2 for dim in grid.shape)
        j = x / grid.length + center_j
        i = -y / grid.length + center_i
    else:
        j = x / grid.length
        i = grid.y - y / grid.length
    return int(i), int(j)


def get_range(uav_pos, grid, index_form=False):
    """
    calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
    """
    position = uav_pos.position
    altitude = uav_pos.altitude
    # print(f"get range alt:{altitude}")
    grid_length = grid.length
    fov = 60
    fov_rad = np.deg2rad(fov) / 2

    x_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length
    y_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length

    # print(f"dist x:{x_dist} y:{y_dist}")
    x_range = [0, grid.x]
    y_range = [0, grid.y]

    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    # print(f"ranges x{x_range} y{y_range}")
    # print(f"pos: {uav_pos.position} {uav_pos.altitude}")

    x_min, x_max = np.clip([position[0] - x_dist, position[0] + x_dist], *x_range)
    y_min, y_max = np.clip([position[1] - y_dist, position[1] + y_dist], *y_range)
    if x_max - x_min == 0 or y_max - y_min == 0:
        return [[0, 0], [0, 0]]

    # print(f"visible ranges x:({x_min}:{x_max}) y:({y_min}:{y_max})")
    if not index_form:
        return [[x_min, x_max], [y_min, y_max]]
    i_max, j_min = convert_xy_ij(x_min, y_min, grid, grid.center)
    i_min, j_max = convert_xy_ij(x_max, y_max, grid, grid.center)
    # if not grid.center:
    #     i_min, i_max = i_max, i_min
    # print(f"visible ranges i:({i_min}:{i_max}) j:({j_min}:{j_max})")
    return [[i_min, i_max], [j_min, j_max]]


# def tranf_coord(pos_x, pos_y, dist_x, dist_y, grid_length):
#     x_min = int(max((-pos_y - dist_y) / grid_length + 200, 0))
#     x_max = int(min((-pos_y + dist_y) / grid_length + 200, 400))
#     y_min = int(max((pos_x - dist_x) / grid_length + 200, 0))
#     y_max = int(min((pos_x + dist_x) / grid_length + 200, 400))

#     return [
#         [x_min, x_max],
#         [y_min, y_max],
#     ]


# def get_range(uav_pos, grid, center=False, index_form=False):
#     """
#     calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
#     """
#     # position = position if position is not None else position
#     # altitude = altitude if altitude is not None else altitude
#     fov = np.deg2rad(60)
#     x_angle = fov / 2  # degree
#     y_angle = fov / 2  # degree
#     x_dist = uav_pos.altitude * math.tan(x_angle)
#     y_dist = uav_pos.altitude * math.tan(y_angle)

#     # adjust func: for smaller square ->int() and for larger-> round()
#     x_dist = round(x_dist / grid.length) * grid.length
#     y_dist = round(y_dist / grid.length) * grid.length
#     # Trim if out of scope (out of the map)
#     x_range = [0, grid.x]
#     y_range = [0, grid.y]

#     if center:
#         x_range = [-grid.x / 2, grid.x / 2]
#         y_range = [-grid.y / 2, grid.y / 2]
#         [
#             [x_min_i, x_max_i],
#             [y_min_j, y_max_j],
#         ] = tranf_coord(
#             uav_pos.position[0], uav_pos.position[1], x_dist, y_dist, grid.length
#         )
#         if index_form:
#             return [
#                 [x_min_i, x_max_i],
#                 [y_min_j, y_max_j],
#             ]

#     x_min = max(uav_pos.position[0] - x_dist, x_range[0])
#     x_max = min(uav_pos.position[0] + x_dist, x_range[1])

#     y_min = max(uav_pos.position[1] - y_dist, y_range[0])
#     y_max = min(uav_pos.position[1] + y_dist, y_range[1])
#     if index_form:  # return as indix range
#         return [
#             [round(x_min / grid.length), round(x_max / grid.length)],
#             [round(y_min / grid.length), round(y_max / grid.length)],
#         ]

#     return [[x_min, x_max], [y_min, y_max]]


def get_observations(grid_info, ground_truth_map, uav_pos, sigmas, rng):
    [[x_min_id, x_max_id], [y_min_id, y_max_id]] = get_range(
        uav_pos, grid_info, index_form=True
    )
    print(f"obs area ids:{x_min_id}:{x_max_id}, {y_min_id}:{y_max_id} ")
    print(f"gt map shape:{ground_truth_map.shape}")

    submap = ground_truth_map[x_min_id:x_max_id, y_min_id:y_max_id]
    print(f"gt submap shape:{submap.shape}")
    x = np.arange(x_min_id, x_max_id, 1)
    y = np.arange(y_min_id, y_max_id, 1)
    sigma0, sigma1 = sigmas[0], sigmas[1]

    # rng = np.random.default_rng()
    random_values = rng.random(submap.shape)
    success0 = random_values <= 1.0 - sigma0
    success1 = random_values <= 1.0 - sigma1
    z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
    z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
    z = np.where(submap == 0, z0, z1)

    x, y = np.meshgrid(x, y, indexing="ij")

    return x, y, z


# tester for non center:

"""


class grid_info:
    x = 50  # 60
    y = 50  # 110 for real field
    length = 0.125  # 1
    shape = (int(x / length), int(y / length))


class uav_pos:
    altitude = 5
    position = (0, 0)


center = False

print(f"center: {center} uav pos:{uav_pos.position}")
print(f"id: {get_range(uav_pos, grid_info, center=center, index_form=True)}")
print(f"pos: {get_range(uav_pos, grid_info, center=center, index_form=False)}")
uav_pos.position = (10, 0)
print(f"center: {center} uav pos:{uav_pos.position}")
print(f"id: {get_range(uav_pos, grid_info, center=center, index_form=True)}")
print(f"pos: {get_range(uav_pos, grid_info, center=center, index_form=False)}")
"""

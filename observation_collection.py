import math
import numpy as np


class camera2map:
    def __init__(self, grid, fov_angle, camera_altitude=0, camera_pos=(0.0, 0.0)):
        self.grid = grid
        self.camera_altitude = camera_altitude
        self.camera_pos = camera_pos
        self.fov = fov_angle

    def set_position(self, pos):
        self.camera_pos = pos

    def set_altitude(self, alt):
        self.camera_altitude = alt

    def get_position(self):
        return self.camera_pos

    def get_altitude(self):
        return self.camera_altitude

    def get_range(self, index_form=False):
        """
        calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
        """
        x_angle = self.fov / 2  # degree
        y_angle = self.fov / 2  # degree
        x_dist = self.camera_altitude * math.tan(x_angle / 180 * 3.14)
        y_dist = self.camera_altitude * math.tan(y_angle / 180 * 3.14)
        # adjust func: for smaller square ->int() and for larger-> round()
        x_dist = int(x_dist / self.grid.length) * self.grid.length
        y_dist = int(y_dist / self.grid.length) * self.grid.length
        # Trim if out of scope (out of the map)
        x_min = max(self.camera_pos[0] - x_dist, 0.0)
        x_max = min(self.camera_pos[0] + x_dist, self.grid.x)

        y_min = max(self.camera_pos[1] - y_dist, 0.0)
        y_max = min(self.camera_pos[1] + y_dist, self.grid.y)
        if index_form:  # return as indix range
            x_min_id, y_min_id = self.pos2grid((x_min, y_min))
            x_max_id, y_max_id = self.pos2grid((x_max, y_max))
            return [[x_min_id, x_max_id], [y_min_id, y_max_id]]

        return [[x_min, x_max], [y_min, y_max]]

    def get_observation(self, map):
        """
        returns submap of terrain (z) to be seen by camera, x,y are position(not indices) wrt terrain
        """
        [[x_min_id, x_max_id], [y_min_id, y_max_id]] = self.get_range(index_form=True)
        submap = map[x_min_id : x_max_id + 1, y_min_id : y_max_id + 1]

        x_min_, x_max_ = self.grid2pos((x_min_id, x_max_id + 1))
        y_min_, y_max_ = self.grid2pos((y_min_id, y_max_id + 1))
        x = np.arange(x_min_, x_max_, self.grid.length)
        y = np.arange(y_min_, y_max_, self.grid.length)

        x, y = np.meshgrid(x, y, indexing="ij")
        return submap, x, y

    def pos2grid(self, pos):
        # from position in meters into grid coordinates
        return min(max(int(pos[0] / self.grid.length), 0), self.grid.shape[0] - 1), min(
            max(int(pos[1] / self.grid.length), 0), self.grid.shape[1] - 1
        )

    def grid2pos(self, coords):
        return coords[0] * self.grid.length, coords[1] * self.grid.length


# def prob_sensor_model(obs_map, h):
#     # probability of observation z given cell state m and uav position x
#     a = 1
#     b = 0.015
#     sigma_s_squared = a * (1 - np.exp(-2* b * h))
#     sigma_s = np.sqrt(sigma_s_squared)

#     # Generate noisy measurements
#     noisy_obs_map = np.random.normal(obs_map, sigma_s)
#     # Clip the noisy measurements to ensure they are between 0 and 1
#     noisy_obs_map = np.clip(noisy_obs_map, 0, 1)
#     return noisy_obs_map


# # Add sampling with probability distr for z==m and z!=m acc to (2)
# def sample_observation(obs_map, h):
#     # Dimensions of the grid
#     rows, cols = obs_map.shape

#     # Initialize the observed map
#     observed_map = np.copy(obs_map)

#     # Loop through each cell in the grid
#     for i in range(rows):
#         for j in range(cols):
#             # Sample whether the observation is correct or wrong
#             if np.random.random() > P_correct:
#                 # If wrong, replace with a random value within the range of true_map
#                 observed_map[i, j] = np.random.uniform(0, 1)

#     return observed_map

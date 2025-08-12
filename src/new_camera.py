import math
import numpy as np
from helper import uav_position


class Camera:
    def __init__(
        self,
        grid,
        fov_angle,
        xy_step=None,
        h_range=None,
        camera_altitude=0,
        camera_pos=(0.0, 0.0),
        rng=np.random.default_rng(123),
        a=1,
        b=0.015,
    ):
        self.grid = grid
        self.position = camera_pos
        self.rng = rng
        self.fov = fov_angle
        self.a = a
        self.b = b

        if self.grid.center:
            self.x_range = [-self.grid.x / 2, self.grid.x / 2]
            self.y_range = [-self.grid.y / 2, self.grid.y / 2]
        else:
            self.x_range = [0, self.grid.x]
            self.y_range = [0, self.grid.y]

        # Set xy_step directly or compute default
        self.xy_step = xy_step

        # Set h_range as a sorted list of allowed altitudes (if given)
        if h_range is not None:
            self.h_range = sorted(h_range)
        else:
            # default h_range with 6 steps starting from camera_altitude
            if camera_altitude == 0:
                base_alt = self.xy_step / np.tan(np.deg2rad(self.fov * 0.5))
            else:
                base_alt = camera_altitude
            self.h_range = [
                base_alt + i * (self.xy_step / np.tan(np.deg2rad(self.fov * 0.5)))
                for i in range(6)
            ]

        # Initialize altitude to first element if not set
        self.altitude = camera_altitude if camera_altitude > 0 else self.h_range[0]

        # Available actions
        self.actions = {"up", "down", "front", "back", "left", "right", "hover"}

    def reset(self):
        self.position = (0.0, 0.0)
        self.altitude = self.h_range[0]

    def set_position(self, pos):
        self.position = pos

    def get_hstep(self):
        # Here h_step is difference between consecutive altitudes (assuming uniform steps)
        if len(self.h_range) < 2:
            return 0
        return self.h_range[1] - self.h_range[0]

    def get_hrange(self):
        return self.h_range

    def set_altitude(self, alt):
        # Set altitude only if it is in h_range
        if alt in self.h_range:
            self.altitude = alt
        else:
            raise ValueError(f"Altitude {alt} not in allowed h_range {self.h_range}")

    def get_x(self):
        return uav_position((self.position, self.altitude))

    def ij_to_xy(self, i, j):
        if self.grid.center:
            center_i, center_j = (dim // 2 for dim in self.grid.shape)
            x = (j - center_j) * self.grid.length
            y = -(i - center_i) * self.grid.length
        else:
            x = j * self.grid.length
            y = (self.grid.shape[0] - i) * self.grid.length
        return (x, y)

    def convert_xy_ij(self, x, y, centered):
        if centered:
            center_i, center_j = (dim // 2 for dim in self.grid.shape)
            j = x / self.grid.length + center_j
            i = -y / self.grid.length + center_i
        else:
            j = x / self.grid.length
            i = self.grid.shape[0] - y / self.grid.length
        return int(i), int(j)

    def get_range(self, position=None, altitude=None, index_form=False):
        position = position if position is not None else self.position
        altitude = altitude if altitude is not None else self.altitude
        grid_length = self.grid.length
        fov_rad = np.deg2rad(self.fov) / 2

        x_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length

        x_min, x_max = np.clip(
            [position[0] - x_dist, position[0] + x_dist], *self.x_range
        )
        y_min, y_max = np.clip(
            [position[1] - y_dist, position[1] + y_dist], *self.y_range
        )

        if x_max - x_min == 0 or y_max - y_min == 0:
            return [[0, 0], [0, 0]]

        if not index_form:
            return [[x_min, x_max], [y_min, y_max]]

        i_max, j_min = self.convert_xy_ij(x_min, y_min, self.grid.center)
        i_min, j_max = self.convert_xy_ij(x_max, y_max, self.grid.center)

        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, ground_truth_map, sigmas=None):
        [[i_min, i_max], [j_min, j_max]] = self.get_range(index_form=True)

        submap = ground_truth_map[i_min:i_max, j_min:j_max]

        if sigmas is None:
            sigma = self.a * (1 - np.exp(-self.b * self.altitude))
            sigmas = [sigma, sigma]

        sigma0, sigma1 = sigmas
        random_values = self.rng.random(submap.shape)
        success0 = random_values <= 1.0 - sigma0
        success1 = random_values <= 1.0 - sigma1
        z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
        z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
        z = np.where(submap == 0, z0, z1)

        fp_vertices_ij = {
            "ul": np.array([i_min, j_min]),
            "bl": np.array([i_max, j_min]),
            "ur": np.array([i_min, j_max]),
            "br": np.array([i_max, j_max]),
        }

        return fp_vertices_ij, z

    def x_future(self, action, x=None):
        if x is None:
            x = self.get_x()

        # For altitude, move to next/previous element in h_range
        current_alt = x.altitude
        try:
            current_idx = self.h_range.index(current_alt)
        except ValueError:
            # Current altitude not in h_range, fallback to nearest index
            current_idx = np.argmin([abs(h - current_alt) for h in self.h_range])

        # Calculate next altitude index for up/down
        if action == "up":
            if current_idx + 1 < len(self.h_range):
                new_alt = self.h_range[current_idx + 1]
                return (x.position, new_alt)
            else:
                return (x.position, current_alt)  # no change, at max altitude
        elif action == "down":
            if current_idx - 1 >= 0:
                new_alt = self.h_range[current_idx - 1]
                return (x.position, new_alt)
            else:
                return (x.position, current_alt)  # no change, at min altitude

        # For XY movements:
        if action == "front" and (x.position[1] + self.xy_step) <= self.y_range[1]:
            return (x.position[0], x.position[1] + self.xy_step), current_alt
        elif action == "back" and (x.position[1] - self.xy_step) >= self.y_range[0]:
            return (x.position[0], x.position[1] - self.xy_step), current_alt
        elif action == "right" and (x.position[0] + self.xy_step) <= self.x_range[1]:
            return (x.position[0] + self.xy_step, x.position[1]), current_alt
        elif action == "left" and (x.position[0] - self.xy_step) >= self.x_range[0]:
            return (x.position[0] - self.xy_step, x.position[1]), current_alt
        else:
            return x.position, current_alt

    def permitted_actions(self, x):
        permitted = ["hover"]

        try:
            current_idx = self.h_range.index(x.altitude)
        except ValueError:
            current_idx = np.argmin([abs(h - x.altitude) for h in self.h_range])

        for action in self.actions:
            if action == "up" and current_idx + 1 < len(self.h_range):
                permitted.append(action)
            elif action == "down" and current_idx - 1 >= 0:
                permitted.append(action)
            elif (
                action == "front" and (x.position[1] + self.xy_step) <= self.y_range[1]
            ):
                permitted.append(action)
            elif action == "back" and (x.position[1] - self.xy_step) >= self.y_range[0]:
                permitted.append(action)
            elif (
                action == "right" and (x.position[0] + self.xy_step) <= self.x_range[1]
            ):
                permitted.append(action)
            elif action == "left" and (x.position[0] - self.xy_step) >= self.x_range[0]:
                permitted.append(action)

        return permitted

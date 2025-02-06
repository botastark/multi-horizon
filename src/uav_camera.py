import math
import numpy as np

# from helper import id_converter, sample_event_matrix,
from helper import uav_position

# from terrain_creation import terrain


class camera:
    def __init__(
        self,
        grid,
        fov_angle,
        camera_altitude=0,
        camera_pos=(0.0, 0.0),
        x_range=(0, 50),
        y_range=(0, 50),
    ):

        self.grid = grid
        self.altitude = camera_altitude
        self.position = camera_pos
        self.fov = fov_angle
        self.x_range = x_range
        self.y_range = y_range

        # Dynamic xy_step and h_step calculation if not explicitly provided
        self.xy_step = (self.x_range[1] - self.x_range[0]) / 2 / 8
        self.h_step = self.xy_step / np.tan(np.deg2rad(self.fov * 0.5))
        self.h_range = (self.h_step, 6 * self.h_step)
        self.a = 1
        self.b = 0.015
        self.actions = {"up", "down", "front", "back", "left", "right", "hover"}

    def reset(self):
        self.position = (0.0, 0.0)
        self.altitude = self.h_step

    def set_position(self, pos):
        self.position = pos

    def get_hstep(self):
        return self.h_step

    def set_altitude(self, alt):
        self.altitude = alt

    def get_x(self):
        return uav_position((self.position, self.altitude))

    def get_range(self, position=None, altitude=None, index_form=False):
        """
        calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
        """
        position = position if position is not None else self.position
        altitude = altitude if altitude is not None else self.altitude

        x_angle = self.fov / 2  # degree
        y_angle = self.fov / 2  # degree
        x_dist = altitude * math.tan(x_angle / 180 * np.pi)
        y_dist = altitude * math.tan(y_angle / 180 * np.pi)

        # adjust func: for smaller square ->int() and for larger-> round()
        x_dist = round(x_dist / self.grid.length) * self.grid.length
        y_dist = round(y_dist / self.grid.length) * self.grid.length
        # Trim if out of scope (out of the map)
        x_min = max(position[0] - x_dist, 0.0)
        x_max = min(position[0] + x_dist, self.grid.x)

        y_min = max(position[1] - y_dist, 0.0)
        y_max = min(position[1] + y_dist, self.grid.y)
        if index_form:  # return as indix range
            return [
                [round(x_min / self.grid.length), round(x_max / self.grid.length)],
                [round(y_min / self.grid.length), round(y_max / self.grid.length)],
            ]

        return [[x_min, x_max], [y_min, y_max]]

    def pos2grid(self, pos):
        # from position in meters into grid coordinates
        return min(
            max(round(pos[0] / self.grid.length), 0), self.grid.shape[0] - 1
        ), min(max(round(pos[1] / self.grid.length), 0), self.grid.shape[1] - 1)

    def grid2pos(self, coords):
        return coords[0] * self.grid.length, coords[1] * self.grid.length

    def x_future(self, action):
        # possible_actions = {"up", "down", "front", "back", "left", "right", "hover"}

        if action == "up" and self.altitude + self.h_step <= self.h_range[1] + 1:
            return self.position, self.altitude + self.h_step
        elif action == "down" and self.altitude - self.h_step >= self.h_range[0] - 1:
            return (self.position, self.altitude - self.h_step)
        # front (+y)
        elif action == "front" and self.position[1] + self.xy_step <= self.y_range[1]:
            return (self.position[0], self.position[1] + self.xy_step), self.altitude
        # back (-y)
        elif action == "back" and self.position[1] - self.xy_step >= self.y_range[0]:
            return (self.position[0], self.position[1] - self.xy_step), self.altitude
        # right (+x)
        elif action == "right" and self.position[0] + self.xy_step <= self.x_range[1]:
            return (self.position[0] + self.xy_step, self.position[1]), self.altitude
        # left (-x)
        elif action == "left" and self.position[0] - self.xy_step >= self.x_range[0]:
            return (self.position[0] - self.xy_step, self.position[1]), self.altitude
        # hover
        else:
            return self.position, self.altitude

    def permitted_actions(self, x):
        # possible_actions = {"up", "down", "front", "back", "left", "right", "hover"}
        permitted_actions = ["hover"]
        for action in self.actions:
            if action == "up" and x.altitude + self.h_step <= self.h_range[1] + 1:
                permitted_actions.append(action)
            elif action == "down" and x.altitude - self.h_step >= self.h_range[0] - 1:
                permitted_actions.append(action)
            elif action == "front" and x.position[1] + self.xy_step <= self.y_range[1]:
                permitted_actions.append(action)
            elif action == "back" and x.position[1] - self.xy_step >= self.y_range[0]:
                permitted_actions.append(action)
            elif action == "right" and x.position[0] + self.xy_step <= self.x_range[1]:
                permitted_actions.append(action)
            elif action == "left" and x.position[0] - self.xy_step >= self.x_range[0]:
                permitted_actions.append(action)
        return permitted_actions

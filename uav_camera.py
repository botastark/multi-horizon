import math
import numpy as np
from helper import argmax_event_matrix, id_converter, sample_event_matrix, uav_position


class camera:
    def __init__(
        self,
        grid,
        fov_angle,
        camera_altitude=0,
        camera_pos=(0.0, 0.0),
        x_range=(0, 50),
        y_range=(0, 50),
        xy_step = 2.94,
        h_range=(5.4, 32.4),
        h_step=5.4,
    ):
        self.grid = grid
        self.altitude = camera_altitude
        self.position = camera_pos
        self.fov = fov_angle
        self.x_range = x_range
        self.y_range = y_range
        self.xy_step = xy_step
        self.h_range = h_range
        self.h_step = h_step
        self.a = 1
        self.b = 0.015
        self.actions = {"up", "down", "front", "back", "left", "right", "hover"}

    def set_position(self, pos):
        self.position = pos

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
        x_dist = altitude * math.tan(x_angle / 180 * 3.14)
        y_dist = altitude * math.tan(y_angle / 180 * 3.14)
        # adjust func: for smaller square ->int() and for larger-> round()
        x_dist = int(x_dist / self.grid.length) * self.grid.length
        y_dist = int(y_dist / self.grid.length) * self.grid.length
        # Trim if out of scope (out of the map)
        x_min = max(position[0] - x_dist, 0.0)
        x_max = min(position[0] + x_dist, self.grid.x)

        y_min = max(position[1] - y_dist, 0.0)
        y_max = min(position[1] + y_dist, self.grid.y)
        if index_form:  # return as indix range
            x_min_id, y_min_id = self.pos2grid((x_min, y_min))
            x_max_id, y_max_id = self.pos2grid((x_max, y_max))
            return [[x_min_id, x_max_id], [y_min_id, y_max_id]]

        return [[x_min, x_max], [y_min, y_max]]

    def get_observation(self, map, position=None, altitude=None):
        """
        returns submap of terrain (z) to be seen by camera, x,y are position(not indices) wrt terrain
        """
        [[x_min_id, x_max_id], [y_min_id, y_max_id]] = self.get_range(
            position=position, altitude=altitude, index_form=True
        )
        submap = map[x_min_id : x_max_id + 1, y_min_id : y_max_id + 1]

        x_min_, x_max_ = self.grid2pos((x_min_id, x_max_id + 1))
        y_min_, y_max_ = self.grid2pos((y_min_id, y_max_id + 1))
        x = np.arange(x_min_, x_max_, self.grid.length)
        y = np.arange(y_min_, y_max_, self.grid.length)

        x, y = np.meshgrid(x, y, indexing="ij")
        return submap, x, y

    def sensor_model(self, m_i, z_i, x):
        sigma = self.a * (1 - np.exp(-self.b * x.altitude))
        if z_i == m_i:
            return (
                1 - sigma
            )  # Get the probability of observing the true state value at this altitude
        else:
            return sigma

    def sample_observation(
        self,
        sampled_M,
        x=None,
        noise=False,
    ):
        if x == None:
            x = uav_position((self.position, self.altitude))

        # creating z as a terrain object, disregard z_
        z = sampled_M.copy()
        z_, z_x, z_y = self.get_observation(
            sampled_M.map, position=x.position, altitude=x.altitude
        )
        z.set_map(z_, x=z_x, y=z_y)
        for z_i_id in z:
            m_i_id = id_converter(z, z_i_id, sampled_M)
            m_i = sampled_M.map[m_i_id]
            z_prob_0 = self.sensor_model(m_i, 0, x)
            z.probability[:, z_i_id[0], z_i_id[1]] = [z_prob_0, 1 - z_prob_0]

        z.map = argmax_event_matrix(z.probability)
        # z.map = sample_event_matrix(z.probability)
        return z

    def pos2grid(self, pos):
        # from position in meters into grid coordinates
        return min(max(int(pos[0] / self.grid.length), 0), self.grid.shape[0] - 1), min(
            max(int(pos[1] / self.grid.length), 0), self.grid.shape[1] - 1
        )

    def grid2pos(self, coords):
        return coords[0] * self.grid.length, coords[1] * self.grid.length

    def _count_possible_states(self):
        count = 1  # hover
        if self.altitude + self.h_step <= self.h_range[1]:  # up
            count += 1
        if self.altitude - self.h_step >= self.h_range[0]:  # down
            count += 1
        if self.position[1] + self.xy_step <= self.y_range[1]:  # front (+y)
            count += 1
        if self.position[1] - self.xy_step <= self.y_range[0]:  # back (-y)
            count += 1
        if self.position[0] + self.xy_step <= self.x_range[1]:  # right (+x)
            count += 1
        if self.position[0] - self.xy_step <= self.x_range[0]:  # left (-x)
            count += 1
        return count

    def _prob_candidate_x(self):
        """
        P(x_{t+1}) = P(x_{t+1} | x_{t}) * P(x_{t})
        P(x_{t+1} | x_{t}) over all possible future positions from the current position
        P(x_{t}) = 1
        """
        prob_candidate_given_current = 1 / self._count_possible_states()
        return prob_candidate_given_current  # * self.prob_uav_position()

    def prob_future_observation(self, x_future, z_future):
        # if z_future is observable from x_future
        # get x, y bbox of terrain cell visibale from x_future
        # if z_future(i.e m_i coordinates are within that range, P(z_{t+1}|x_{t+1}) = 1, and otherwise, 0)
        [[x_min_id, x_max_id], [y_min_id, y_max_id]] = self.get_range(
            position=x_future.position, altitude=x_future.altitude, index_form=True
        )
        x_min_, x_max_ = self.grid2pos((x_min_id, x_max_id + 1))
        y_min_, y_max_ = self.grid2pos((y_min_id, y_max_id + 1))
        if x_min_ <= z_future.x <= x_max_:
            if y_min_ <= z_future.y <= y_max_:
                return self._prob_candidate_x() * z_future.probability
            else:
                raise TypeError("check prob_future_observation y out of range")
        else:
            raise TypeError("check prob_future_observation: x out of range")
        return 0

    def x_future(self, action):
        # possible_actions = {"up", "down", "front", "back", "left", "right", "hover"}

        if action == "up" and self.altitude + self.h_step <= self.h_range[1]:
            return self.position, self.altitude + self.h_step
        elif action == "down" and self.altitude - self.h_step >= self.h_range[0]:
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
            if action == "up" and x.altitude + self.h_step <= self.h_range[1]:
                permitted_actions.append(action)
            elif action == "down" and x.altitude - self.h_step >= self.h_range[0]:
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

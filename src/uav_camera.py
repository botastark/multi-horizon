import math
import numpy as np
from helper import uav_position


class Camera:
    """
    Camera simulation class for UAV-based terrain observation.

    This class models a camera attached to a UAV. It computes the camera's
    field-of-view (FoV) and ground footprint based on the UAV's position and altitude,
    and provides methods to simulate observations, compute footprint ranges, and
    determine permitted actions based on UAV movement constraints.

    Attributes:
        grid: A grid object representing the terrain. Expected to have attributes:
            - x: width of the grid (in meters)
            - y: height of the grid (in meters)
            - center (bool): whether the grid is centered at (0, 0)
            - length: the side length of a grid cell
            - shape: (rows, cols) of the grid
        fov (float): Camera field-of-view angle in degrees.
        altitude (float): Current altitude of the UAV.
        position (tuple): Current (x, y) position of the UAV.
        rng (np.random.Generator): Random number generator for observation noise.
        xy_step (float): Step size for horizontal movement (in meters).
        h_step (float): Step size for altitude adjustment (in meters).
        h_range (tuple): Permissible altitude range as (min_altitude, max_altitude).
        a (float): Parameter used to compute the noise sigma.
        b (float): Parameter used to compute the noise sigma.
        actions (set): Set of available actions.
    """

    def __init__(
        self,
        grid,
        fov_angle,
        h_step=None,
        f_overlap=0.8,
        s_overlap=0.7,
        camera_altitude=0,
        camera_pos=(0.0, 0.0),
        rng=np.random.default_rng(123),
        a=1,
        b=0.015,
    ):
        """
        Initialize the Camera.

        Parameters:
            grid: The terrain grid object.
            fov_angle (float): Field-of-view angle (in degrees).
            h_step (float, optional): Step size for altitude adjustment. If None, computed automatically.
            f_overlap (float): Desired forward overlap ratio (0 to 1).
            s_overlap (float): Desired side overlap ratio (0 to 1).
            camera_altitude (float): Initial altitude of the camera/UAV.
            camera_pos (tuple): Initial (x, y) position of the camera/UAV.
            rng (np.random.Generator): Random number generator for simulation.
            a (float): Parameter for computing observation noise sigma.
            b (float): Parameter for computing observation noise sigma.
        """

        self.grid = grid
        self.altitude = camera_altitude
        self.position = camera_pos
        self.rng = rng
        self.fov = fov_angle
        self.a = a
        self.b = b
        # Define grid boundaries based on whether grid is centered or not.
        if self.grid.center:
            self.x_range = [-self.grid.x / 2, self.grid.x / 2]
            self.y_range = [-self.grid.y / 2, self.grid.y / 2]
        else:
            self.x_range = [0, self.grid.x]
            self.y_range = [0, self.grid.y]
        # Compute horizontal step size (xy_step) based on overlaps and FoV.
        if f_overlap != None and s_overlap != None:
            # Compute horizontal step size (xy_step) based on overlaps and FoV.
            theta_w = np.deg2rad(self.fov)  # Horizonatal FoV (radians)
            theta_h = np.deg2rad(self.fov)  # Vertical FoV (radians)

            # Compute ground footprint dimensions
            W = 2 * self.altitude * np.tan(theta_w / 2)  # Ground width (meters)
            H = 2 * self.altitude * np.tan(theta_h / 2)  # Ground height (meters)

            # Compute step sizes based on overlap
            xy_step_f = H * (1 - f_overlap)  # Forward step
            xy_step_s = W * (1 - s_overlap)  # Side step
            self.xy_step = round(xy_step_f, 2)
        else:
            # Fallback: use half the minimum range divided by 8.
            min_range = min(
                self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]
            )
            self.xy_step = min_range / 2 / 8

        if h_step is None:
            self.h_step = self.xy_step / np.tan(np.deg2rad(self.fov * 0.5))
        else:
            self.h_step = h_step
        # Ensure altitude is set (if initially 0, use h_step).
        if self.altitude == 0 or self.altitude is None:
            self.altitude = self.h_step
            # Define permissible altitude range (from current altitude to five steps above).
        self.h_range = (self.altitude, self.altitude + 5 * self.h_step)

        # Define available actions.
        self.actions = {"up", "down", "front", "back", "left", "right", "hover"}
        # print(f"H range: {self.h_range}")
        # print(f"xy_step {self.xy_step}, h_step {self.h_step}")

    def reset(self):
        self.position = (0.0, 0.0)
        self.altitude = self.h_step

    def set_position(self, pos):
        """
        Update the UAV's (camera's) position.

        Parameters:
            pos (tuple): New (x, y) position.
        """
        self.position = pos

    def get_hstep(self):
        """
        Get the vertical (altitude) step size.

        Returns:
            float: h_step value.
        """
        return self.h_step

    def get_hrange(self):
        """
        Get the permissible altitude range.

        Returns:
            tuple: (min_altitude, max_altitude)
        """
        return self.h_range

    def set_altitude(self, alt):
        """
        Update the camera's altitude.

        Parameters:
            alt (float): New altitude.
        """
        self.altitude = alt

    def get_x(self):
        """
        Get the UAV's full state including position and altitude.

        Returns:
            Object: UAV state as defined by the helper.uav_position function.
        """
        return uav_position((self.position, self.altitude))

    def convert_xy_ij(self, x, y, centered):
        """
        Convert real-world (x, y) coordinates to grid indices (i, j).

        Parameters:
            x (float): X coordinate.
            y (float): Y coordinate.
            centered (bool): If True, the grid is considered centered; conversion is adjusted accordingly.

        Returns:
            tuple: (i, j) grid indices as integers.
        """
        if centered:
            center_i, center_j = (dim // 2 for dim in self.grid.shape)
            j = x / self.grid.length + center_j
            i = -y / self.grid.length + center_i
        else:
            j = x / self.grid.length
            i = self.grid.shape[0] - y / self.grid.length
        return int(i), int(j)

    def get_range(self, position=None, altitude=None, index_form=False):
        """
        Calculate the visible ground footprint range for the camera.

        This method computes the ground area (or grid indices) visible to the camera based on its
        current position, altitude, and field-of-view.

        Parameters:
            position (tuple, optional): (x, y) position. Defaults to current position.
            altitude (float, optional): Altitude value. Defaults to current altitude.
            index_form (bool): If True, return grid indices; otherwise return world coordinates.

        Returns:
            list: [[x_min, x_max], [y_min, y_max]] in world coordinates or
                  [[i_min, i_max], [j_min, j_max]] in grid index form.
        """
        # Use provided position/altitude or default to current values.
        position = position if position is not None else self.position
        altitude = altitude if altitude is not None else self.altitude
        grid_length = self.grid.length
        fov_rad = np.deg2rad(self.fov) / 2
        # Use provided position/altitude or default to current values.
        x_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(altitude * math.tan(fov_rad) / grid_length) * grid_length
        # Clip the computed ranges within grid boundaries.
        x_min, x_max = np.clip(
            [position[0] - x_dist, position[0] + x_dist], *self.x_range
        )
        y_min, y_max = np.clip(
            [position[1] - y_dist, position[1] + y_dist], *self.y_range
        )
        # Return a default empty range if footprint is zero-sized.
        if x_max - x_min == 0 or y_max - y_min == 0:
            return [[0, 0], [0, 0]]
        """
        print(f"dist x:{x_dist} y:{y_dist}")
        print(f"ranges x{self.x_range} y{self.y_range}")
        print(f"pos: {self.position} {self.altitude}")
        print(f"visible ranges x:({x_min}:{x_max}) y:({y_min}:{y_max})")
        """
        # Return in world coordinate form.
        if not index_form:
            return [[x_min, x_max], [y_min, y_max]]
        # Convert the world coordinate range to grid indices.s
        i_max, j_min = self.convert_xy_ij(x_min, y_min, self.grid.center)
        i_min, j_max = self.convert_xy_ij(x_max, y_max, self.grid.center)
        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, ground_truth_map, sigmas=None):
        """
        Simulate camera observations over the ground truth map.

        This method extracts the submap corresponding to the camera's visible area and simulates
        noisy observations based on the current altitude. Noise is modeled using an exponential decay
        function with parameters a and b.

        Parameters:
            ground_truth_map (np.ndarray): The full terrain label map.
            sigmas (list, optional): List containing sigma values for noise simulation for each class.
                                     If None, sigma is computed as: a * (1 - exp(-b * altitude)).

        Returns:
            tuple:
                - dict: A dictionary of footprint vertices (in grid indices) with keys 'ul', 'bl', 'ur', 'br'.
                - np.ndarray: The noisy observed submap.
        """

        # Get grid indices corresponding to the camera footprint.
        [[i_min, i_max], [j_min, j_max]] = self.get_range(index_form=True)

        submap = ground_truth_map[i_min:i_max, j_min:j_max]
        """
        print(f"obs area ids:{x_min_id}:{x_max_id}, {y_min_id}:{y_max_id} ")
        print(f"gt map shape:{ground_truth_map.shape}")
        print(f"gt submap shape:{submap.shape}")
        """
        # Compute noise sigma if not provided.
        if sigmas is None:
            sigma = self.a * (1 - np.exp(-self.b * self.altitude))
            sigmas = [sigma, sigma]

        sigma0, sigma1 = sigmas[0], sigmas[1]
        # Generate random noise values and simulate observation success based on sigma thresholds.
        # rng = np.random.default_rng()
        random_values = self.rng.random(submap.shape)
        success0 = random_values <= 1.0 - sigma0
        success1 = random_values <= 1.0 - sigma1
        z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
        z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
        z = np.where(submap == 0, z0, z1)

        # Define footprint vertices in grid indices.
        fp_vertices_ij = {
            "ul": np.array([i_min, j_min]),
            "bl": np.array([i_max, j_min]),
            "ur": np.array([i_min, j_max]),
            "br": np.array([i_max, j_max]),
        }

        return fp_vertices_ij, z

    def x_future(self, action, x=None):
        """
        Compute the UAV's future state given an action.

        This method calculates the next position and altitude based on the selected action.
        Actions include moving vertically (up/down) and horizontally (front/back/left/right).

        Parameters:
            action (str): One of the actions from {"up", "down", "front", "back", "left", "right", "hover"}.
            x (object, optional): Current UAV state as returned by get_x(). If None, uses current state.

        Returns:
            tuple: A tuple (new_position, new_altitude).
        """
        if x is None:
            x = self.get_x()
        # Compute future state based on action and ensure movement is within allowed ranges.
        if action == "up" and (x.altitude + self.h_step) <= self.h_range[1]:
            return (x.position, x.altitude + self.h_step)
        elif action == "down" and (x.altitude - self.h_step) >= self.h_range[0]:
            return (x.position, x.altitude - self.h_step)
        elif action == "front" and (x.position[1] + self.xy_step) <= self.y_range[1]:
            return (x.position[0], x.position[1] + self.xy_step), x.altitude
        elif action == "back" and (x.position[1] - self.xy_step) >= self.y_range[0]:
            return (x.position[0], x.position[1] - self.xy_step), x.altitude
        elif action == "right" and (x.position[0] + self.xy_step) <= self.x_range[1]:
            return (x.position[0] + self.xy_step, x.position[1]), x.altitude
        elif action == "left" and (x.position[0] - self.xy_step) >= self.x_range[0]:
            return (x.position[0] - self.xy_step, x.position[1]), x.altitude
        else:
            # If action is not permitted, remain in the current state.
            return x.position, x.altitude

    def permitted_actions(self, x):
        """
        Determine the set of actions permitted from the current UAV state.

        Checks each action against the grid boundaries and altitude range, returning only those
        actions that keep the UAV within valid limits.

        Parameters:
            x: Current UAV state (as returned by get_x()).

        Returns:
            list: A list of permitted action strings.
        """
        permitted_actions = ["hover"]

        for action in self.actions:
            future_x = self.x_future(action, x=x)

            # Altitude checks
            if action == "up" and future_x[1] <= self.h_range[1]:
                permitted_actions.append(action)
            elif action == "down" and future_x[1] >= self.h_range[0]:
                permitted_actions.append(action)

            # XY position checks (now using future_x)
            elif action == "front" and future_x[0][1] <= self.y_range[1]:  # Corrected
                permitted_actions.append(action)
            elif action == "back" and future_x[0][1] >= self.y_range[0]:  # Corrected
                permitted_actions.append(action)
            elif action == "right" and future_x[0][0] <= self.x_range[1]:
                permitted_actions.append(action)
            elif action == "left" and future_x[0][0] >= self.x_range[0]:
                permitted_actions.append(action)

        return permitted_actions

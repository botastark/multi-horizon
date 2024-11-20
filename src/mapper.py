import numpy as np
import math
from helper import adaptive_weights_matrix


class OccupancyMap:
    def __init__(self, n_cells):
        self.N = n_cells  # Grid size (100x100)
        self.states = [0, 1]  # Possible states
        # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N, self.N, 2), 0.5)  # 2 states: [0, 1]
        self.last_observations = np.array([])
        # Initialize messages (for each edge, uniform message)
        self.messages = {}
        for i in range(self.N):
            for j in range(self.N):
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for ni, nj in neighbors:
                    if 0 <= ni < self.N and 0 <= nj < self.N:  # Valid neighbor
                        self.messages[((i, j), (ni, nj))] = [0.5, 0.5]

    # Local evidence function (sensor model)
    def sensor_model(self, z_i, m_i, x):
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x.altitude))
        if z_i == m_i:
            return (
                1 - sigma
            )  # Get the probability of observing the true state value at this altitude
        else:
            return sigma

    def local_evidence(self, z, x):
        """
        Compute local evidence phi(X_i) for a given cell based on observation z and error sigma.

        Parameters:
            z: Observed state (0 or 1).
            x: UAV position, (x.positiom - (i,j), x.altitude - h)

        Returns:
            A list [P(free), P(occupied)] representing the local evidence for the cell.
        """
        if z == 0:
            return [
                self.sensor_model(0, 0, x),
                self.sensor_model(0, 1, x),
            ]  # [P(z=0|m=0), P(z=0|m=1)]
        elif z == 1:
            return [
                self.sensor_model(1, 0, x),
                self.sensor_model(1, 1, x),
            ]  # [P(z=1|m=0), P(z=1|m=1)]
        else:
            return [0.5, 0.5]  # Default uniform prior if no observation

    # Pairwise potential function
    def pairwise_potential(self, correlation_type=None):
        """
        Compute pairwise potential psi(X_i, X_j) for neighboring cells.
        Options:
            - Uniform: (0.5, 0.5)
            - Biased: Fixed (0.7, 0.3).
            - Adaptive: Based on a metric like Pearson correlation.
        """
        if correlation_type == "equal":
            # Default: Uniform potential
            return np.array([[0.5, 0.5], [0.5, 0.5]])
        elif correlation_type == "biased":
            # Fixed bias
            return np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            # Adaptive: Pearson correlation coefficient
            return np.array(adaptive_weights_matrix(self.last_observations))

    # Update observations
    def update_observations(self, observations, uav_pos, marginals):
        """
        Update local evidence phi based on observations.
        observations: List of tuples (i, j, {'free': p_free, 'occupied': p_occupied}).
        """
        self.last_observations = observations
        for i, j, obs in observations:
            # self.phi[i, j] = self.local_evidence(obs, uav_pos)

            # # Get local evidence for this observation
            local_evidence = self.local_evidence(obs, uav_pos)  # [P(free), P(occupied)]

            # # Fuse with prior belief
            # prior_belief = self.phi[i, j]

            # Use current marginals as the prior
            prior_belief = marginals[i, j]

            # Fuse the prior belief with the new local evidence
            fused_belief = prior_belief * local_evidence  # Element-wise multiplication

            # Normalize to ensure it's a valid probability distribution
            fused_belief /= np.sum(fused_belief)

            # Update phi with the fused belief
            self.phi[i, j] = fused_belief

    def set_last_observations(self, submap):
        self.last_observations = np.array(submap)

    def propagate_messages(self, max_iterations=5, correlation_type=None):
        """
        Perform loopy belief propagation to update messages.

        Parameters:
            max_iterations: Number of iterations to perform.
            correlation_type:  for pairwise potentials.

        Returns:
            Updated messages.
        """
        for _ in range(max_iterations):
            new_messages = {}
            for ((i, j), (ni, nj)), message in self.messages.items():
                # Compute incoming messages from other neighbors
                incoming_messages = [
                    self.messages[((ni_, nj_), (i, j))]
                    for ni_, nj_ in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    if (ni_, nj_) != (ni, nj)
                    and 0 <= ni_ < self.N
                    and 0 <= nj_ < self.N
                ]
                # Product of incoming messages
                prod_incoming = np.prod(incoming_messages, axis=0)

                # Pairwise potential
                psi = self.pairwise_potential(correlation_type)
                # print(psi)
                # Compute new message
                new_message = np.dot(self.phi[i, j] * prod_incoming, psi)
                new_message /= np.sum(new_message)  # Normalize
                new_messages[((i, j), (ni, nj))] = new_message
            self.messages = new_messages

    def marginalize(self):
        """
        Compute marginals for each cell.
        Returns:
            Marginals array of shape (N, N, 2).
        """
        marginals = np.zeros((self.N, self.N, 2))
        for i in range(self.N):
            for j in range(self.N):
                incoming_messages = [
                    self.messages[((ni, nj), (i, j))]
                    for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    if 0 <= ni < self.N and 0 <= nj < self.N
                ]
                prod_incoming = np.prod(incoming_messages, axis=0)
                marginals[i, j] = self.phi[i, j] * prod_incoming
                marginals[i, j] /= np.sum(marginals[i, j])  # Normalize
        return marginals


def get_range(uav_pos, grid, index_form=False):
    """
    calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
    """
    # position = position if position is not None else self.position
    # altitude = altitude if altitude is not None else self.altitude
    fov = 60
    x_angle = fov / 2  # degree
    y_angle = fov / 2  # degree
    x_dist = uav_pos.altitude * math.tan(x_angle / 180 * 3.14)
    y_dist = uav_pos.altitude * math.tan(y_angle / 180 * 3.14)

    # adjust func: for smaller square ->int() and for larger-> round()
    x_dist = round(x_dist / grid.length) * grid.length
    y_dist = round(y_dist / grid.length) * grid.length
    # Trim if out of scope (out of the map)
    x_min = max(uav_pos.position[0] - x_dist, 0.0)
    x_max = min(uav_pos.position[0] + x_dist, grid.x)

    y_min = max(uav_pos.position[1] - y_dist, 0.0)
    y_max = min(uav_pos.position[1] + y_dist, grid.y)
    if index_form:  # return as indix range
        return [
            [round(x_min / grid.length), round(x_max / grid.length)],
            [round(y_min / grid.length), round(y_max / grid.length)],
        ]

    return [[x_min, x_max], [y_min, y_max]]


def get_observations(grid_info, ground_truth_map, uav_pos):
    [[x_min_id, x_max_id], [y_min_id, y_max_id]] = get_range(
        uav_pos, grid_info, index_form=True
    )
    submap = ground_truth_map[x_min_id:x_max_id, y_min_id:y_max_id]
    # print(submap)
    # print(f"[ {x_min_id}, {x_max_id}], [{y_min_id}, {y_max_id}] ")

    observations = [
        (x_min_id + i, y_min_id + j, submap[i, j])  # Global indices with binary state
        for i in range(submap.shape[0])
        for j in range(submap.shape[1])
    ]
    return submap, observations

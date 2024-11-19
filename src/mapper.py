import numpy as np
from helper import uav_position

class OccupancyMap:
    def __init__(self):
        # Grid dimensions
        self.N = 100  # Grid size (100x100)
        self.states = [0, 1]  # Possible states
         # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N, self.N, 2), 0.5)  # 2 states: [0, 1]
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
            return [self.sensor_model(0, 0, x), self.sensor_model(0, 1, x)]  # [P(z=0|m=0), P(z=0|m=1)]
        elif z == 1:
            return [self.sensor_model(1, 0, x), self.sensor_model(1, 1, x)]   # [P(z=1|m=0), P(z=1|m=1)]
        else:
            return [0.5, 0.5]  # Default uniform prior if no observation

    # Pairwise potential function
    def pairwise_potential(self, i, j, ni, nj, correlation=None):
        """
        Compute pairwise potential psi(X_i, X_j) for neighboring cells.
        Options:
            - Uniform: (0.5, 0.5)
            - Biased: Fixed (0.7, 0.3).
            - Adaptive: Based on a metric like Pearson correlation.
        """
        if correlation == "equal":
            # Default: Uniform potential
            return np.array([[0.5, 0.5], [0.5, 0.5]])
        elif correlation == "biased":
            # Fixed bias
            return np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            # Adaptive: Example with Pearson correlation coefficient
            # Normalize correlation to [0.3, 0.7] range
            adaptive_value = 0.5 + 0.2 * correlation
            return np.array(
                [
                    [adaptive_value, 1 - adaptive_value],
                    [1 - adaptive_value, adaptive_value],
                ]
            )

    # Update observations
    def update_observations(self, observations, uav_pos):
        """
        Update local evidence phi based on observations.
        observations: List of tuples (i, j, {'free': p_free, 'occupied': p_occupied}).
        """
        for i, j, obs in observations:
            self.phi[i, j] = self.local_evidence(obs, uav_pos)

    def propagate_messages(self, max_iterations=5, correlation=None):
        """
        Perform loopy belief propagation to update messages.

        Parameters:
            max_iterations: Number of iterations to perform.
            correlation: Optional metric for adaptive pairwise potentials.

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
                psi = self.pairwise_potential(i, j, ni, nj, correlation)

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


grid = OccupancyMap()


class grid_info:
    x = 20
    y = 20
    length = 0.25
    shape = (int(x / length), int(y / length))


class camera_params:
    fov_angle = 60


x = uav_position(((0, 0), 5.2))


observations = [
    (10, 10, 1),
    (20, 20, 0),
]
grid.update_observations(observations, x)
grid.propagate_messages(max_iterations=5, correlation="equal")
marginals = grid.marginalize()
print("Marginal at (10, 10):", marginals[10, 10])

print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 20):", marginals[10, 20, 1])

observations = [
    (10, 10, 0),
    # (20, 20, 0),
]
print("after shange ")
grid.update_observations(observations, x)
grid.propagate_messages(max_iterations=5, correlation="equal")
marginals = grid.marginalize()
print("Marginal at (10, 10):", marginals[10, 10])

print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 20):", marginals[10, 20, 1])
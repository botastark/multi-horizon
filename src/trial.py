import numpy as np


class OccupancyMap:
    def __init__(self):
        # Grid dimensions
        self.N = 100  # Grid size (100x100)
        self.states = ["free", "occupied"]  # Possible states

        # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N, self.N, 2), 0.5)  # 2 states: [free, occupied]

        # Initialize messages (for each edge, uniform message)
        self.messages = {}
        for i in range(self.N):
            for j in range(self.N):
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for ni, nj in neighbors:
                    if 0 <= ni < self.N and 0 <= nj < self.N:  # Valid neighbor
                        self.messages[((i, j), (ni, nj))] = [0.5, 0.5]

    # Local evidence function (sensor model)
    def local_evidence(self, z, sigma=0.01):
        """
        Compute local evidence phi(X_i) for a given cell based on observation z and error sigma.

        Parameters:
            z: Observed state ("free" or "occupied").
            sigma: Probability of error in the observation.

        Returns:
            A list [P(free), P(occupied)] representing the local evidence for the cell.
        """
        if z == "free":
            return [1 - sigma, sigma]  # High probability for "free"
        elif z == "occupied":
            return [sigma, 1 - sigma]  # High probability for "occupied"
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
    def update_observations(self, observations):
        """
        Update local evidence phi based on observations.
        observations: List of tuples (i, j, {'free': p_free, 'occupied': p_occupied}).
        """
        for i, j, obs in observations:
            self.phi[i, j] = self.local_evidence(obs)

    def propagate_messages(self, max_iterations=10, correlation=None):
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


observations = [
    (10, 10, "occupied"),
    (20, 20, "free"),
]
grid.update_observations(observations)
grid.propagate_messages(max_iterations=10, correlation="equal")
marginals = grid.marginalize()
print("Marginal at (10, 10):", marginals[10, 10])

print("Probability of occupied at (10, 10):", marginals[10, 10, 1])
print("Probability of occupied at (10, 15):", marginals[10, 15, 1])

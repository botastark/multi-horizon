import numpy as np
import math
from src.helper import adaptive_weights_matrix
I, J = 0,1

class OccupancyMap:
    def __init__(self, n_cells):
        self.N = n_cells  # Grid size (100x100)
        # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N, self.N, 2), 0.5)  # 2 states: [0, 1]
        self.last_observations = {}
        self.local_evidence = None
        self.messages = {}

        # for i in range(self.N):
        #     for j in range(self.N):
        #         neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        #         for ni, nj in neighbors:
        #             if 0 <= ni < self.N and 0 <= nj < self.N:  # Valid neighbor
        #                 self.messages[((i, j), (ni, nj))] = np.array([0.5, 0.5])


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
            return np.array([[0.5, 0.5], [0.5, 0.5]])
        elif correlation_type == "biased":
            return np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            # Adaptive: Pearson correlation coefficient
            return np.array(adaptive_weights_matrix(self.last_observations["z"]))


    def update_observations(self, i, j, submap, uav_pos, marginals):
        """
        Update local evidence phi based on observations.

        Parameters:
            submap: 2D array of observed states (0 or 1).
            uav_pos: UAV position as a dictionary with altitude and other metadata.
            marginals: 3D array of current marginals [P(free), P(occupied)].
        """
        # Flatten submap and grid indices for vectorized processing
        z = submap.flatten()
        i_flat = i.flatten()
        j_flat = j.flatten()

        # # Compute local evidence for all observations
        a, b = 1, 0.015
        sigma = a * (
            1 - np.exp(-b * uav_pos.altitude)
        )

        # # Vectorized local evidence computation
        self.local_evidence = np.zeros((len(z), 2))  # [P(free), P(occupied)]
        self.local_evidence[:, 0] = np.where(z == 0, 1 - sigma, sigma)  # P(free)
        self.local_evidence[:, 1] = np.where(z == 0, sigma, 1 - sigma)  # P(occupied)


        # Extract prior beliefs from marginals
        prior_beliefs = marginals[i_flat, j_flat]  # Shape (len(z), 2)

        # Fuse prior beliefs with local evidence
        fused_beliefs = prior_beliefs * self.local_evidence  # Element-wise multiplication
        fused_beliefs /= fused_beliefs.sum(axis=1, keepdims=True)  # Normalize

        # Update phi for the observed grid cells
        self.phi[i_flat, j_flat] = np.round(fused_beliefs, decimals=10)
        self.last_observations = {"z": np.array(submap), "i":i_flat, "j":j_flat}

    def propagate_messages(self, fp_ij, max_iterations=5, correlation_type=None):
        j_min, j_max = fp_ij["ul"][1], fp_ij["ur"][1] - 1
        i_min, i_max = fp_ij["ul"][0], fp_ij["bl"][0] - 1

        # Pairwise potential
        psi = self.pairwise_potential(correlation_type)
        # init 
        for i in range(self.N):
            for j in range(self.N):
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                for ni, nj in neighbors:
                    if 0 <= ni < self.N and 0 <= nj < self.N:  # Valid neighbor
                        self.messages[((i, j), (ni, nj))] = [0.5, 0.5]

        for _ in range(max_iterations):
            new_messages = {}
            for ((i, j), (ni, nj)), message in self.messages.items():
                if (
                    i_min <= i <= i_max and j_min <= j <= j_max and
                    i_min <= ni <= i_max and j_min <= nj <= j_max
                ):
                    
                    # Nk (ni_, nj_) neighbors of C (i,j) except for Ni (ni, nj), messgae from Nk to C
                    incoming_msgs = np.array([
                        self.messages[(ni_, nj_), (i, j)]
                        for ni_, nj_ in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                        if (ni_, nj_) != (ni, nj) 
                        and 0 <= ni_ <= self.N-1 
                        and 0 <= nj_ <= self.N-1
                    ])
                    prod_incoming = np.prod(incoming_msgs, axis=0)

                    # Retrieve local evidence for (i, j)
                    indices = np.where((self.last_observations["i"] == i) & (self.last_observations["j"] == j))[0]
                    local_evidence_c = self.local_evidence[indices[0]]
                    local_evidence_c =self.phi[i,j]
                    # local_evidence_c = np.array([1,1])
                    # Compute outgoing message from c to Ni
                    # new_message = np.dot(psi, prod_incoming) 
                    # print(f"correct order: {new_message}")
                    m_j0=local_evidence_c[0]*psi[0,0]*prod_incoming[0]+local_evidence_c[1]*psi[1,0]*prod_incoming[1]
                    m_j1=local_evidence_c[0]*psi[0,1]*prod_incoming[0]+local_evidence_c[1]*psi[1,1]*prod_incoming[1]
                    # print(f"m j {m_j0},{m_j1}")
                    new_message = [m_j0, m_j1]
                    # new_message =local_evidence_c * np.dot( prod_incoming, psi)
                    # print(f"m prev {new_message}")
                    # print(f'weird order {new_message}')
                    new_message /= np.sum(new_message)  # Normalize
                    new_messages[((i, j), (ni, nj))] = np.round(new_message, decimals=10)

            # Update message storage
            self.messages.update(new_messages)
            

    def marginalize(self):
        """
        Compute marginals for each cell.
        Returns:
            Marginals array of shape (N, N, 2).
        """
        marginals = np.zeros((self.N, self.N, 2))
        for i in range(self.N):
            for j in range(self.N):
                incoming_messages = np.array([
                    self.messages[((ni, nj), (i, j))]
                    for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    if 0 <= ni < self.N and 0 <= nj < self.N
                ])
                prod_incoming = np.prod(incoming_messages, axis=0)
                marginals[i, j] = self.phi[i, j] * prod_incoming
                marginals[i, j] /= np.sum(marginals[i, j])  # Normalize
        return np.round(marginals, decimals=10)


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

    x = np.arange(
        x_min_id * grid_info.length, x_max_id * grid_info.length, grid_info.length
    )
    y = np.arange(
        y_min_id * grid_info.length, y_max_id * grid_info.length, grid_info.length
    )

    x, y = np.meshgrid(x, y, indexing="ij")

    return x, y, submap


def adapt_observations(x, y, submap, grid_info):
    x_min_id = int(x[0, 0] / grid_info.length)
    y_min_id = int(y[0, 0] / grid_info.length)

    observations = [
        (x_min_id + i, y_min_id + j, submap[i, j])  # Global indices with binary state
        for i in range(submap.shape[0])
        for j in range(submap.shape[1])
    ]
    return observations

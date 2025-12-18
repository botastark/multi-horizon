import numpy as np
from helper import adaptive_weights_matrix
from typing import Dict, Optional, Tuple, Any


class OccupancyMap:
    """
    OccupancyMap constructs a grid-based map where each cell represents the belief
    of being occupied (1) or free (0). It supports Loopy Belief Propagation for
    spatial consistency and observation fusion.

    Args:
        grid_size (tuple): Dimensions of the grid (rows, cols)
        conf_dict (dict): Optional, maps UAV altitudes to confidence values (sigma0, sigma1)
        correlation_type (str): One of ['equal', 'biased', 'adaptive']
    """

    def __init__(
        self, grid_size, conf_dict=None, correlation_type=None, eps: float = 0.0
    ):
        self.N = grid_size
        self.conf_dict = conf_dict
        self.correlation_type = correlation_type
        self.eps = eps

        self.phi = np.full((self.N[0], self.N[1], 2), 0.5)  # Local evidence
        self.map_beliefs = np.full((self.N[0], self.N[1]), 0.5)  # Marginal beliefs

        self.msgs = None
        self.msgs_buffer = None
        self.direction_to_slicing_data = None
        self._init_LBP_msgs()

        # Sensor model parameters (set during OG update)
        self.sigma0: Optional[float] = None
        self.sigma1: Optional[float] = None

        # Used for adaptive pairwise potentials
        self.last_observations = np.array([])

    def reset(self, conf_dict=None):
        """Reset the map beliefs and message structures."""
        self.conf_dict = conf_dict
        self.phi = np.full((self.N[0], self.N[1], 2), 0.5)
        self.last_observations = np.array([])
        self._init_LBP_msgs()
        self.map_beliefs = np.full((self.N[0], self.N[1]), 0.5)

    def _init_LBP_msgs(self):
        """Initialize messages and their buffers for belief propagation."""
        self.msgs = np.ones((4 + 1, self.N[0], self.N[1]), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5
        I, J = 0, 1
        # Define slicing rules for message passing in each direction

        # (channelS, row_slice, col_slice) to product & marginalize
        # (row_slice, col_slice) to read
        # (channel, row_slice, col_slice) to write
        self.direction_to_slicing_data = {
            "up": {
                "product_slice": lambda fp_ij: (
                    (1, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        1 if fp_ij["ul"][I] == 0 else 0, fp_ij["bl"][I] - fp_ij["ul"][I]
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    2,
                    slice(
                        max(0, fp_ij["ul"][I] - 1), min(self.N[0], fp_ij["bl"][I] - 1)
                    ),
                    slice(max(0, fp_ij["ul"][J]), min(self.N[1], fp_ij["br"][J])),
                ),
            },
            "right": {
                "product_slice": lambda fp_ij: (
                    (0, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        0,
                        (
                            fp_ij["ur"][J] - fp_ij["ul"][J] - 1
                            if fp_ij["ur"][J] == self.N[1]
                            else fp_ij["ur"][J] - fp_ij["ul"][J]
                        ),
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    3,
                    slice(max(0, fp_ij["ul"][I]), min(self.N[0], fp_ij["bl"][I])),
                    slice(
                        max(0, fp_ij["ul"][J] + 1), min(self.N[1], fp_ij["br"][J] + 1)
                    ),
                ),
            },
            "down": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        0,
                        (
                            fp_ij["bl"][I] - fp_ij["ul"][I] - 1
                            if fp_ij["bl"][I] == self.N[0]
                            else fp_ij["bl"][I] - fp_ij["ul"][I]
                        ),
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    0,
                    slice(
                        max(0, fp_ij["ul"][I] + 1), min(self.N[0], fp_ij["bl"][I] + 1)
                    ),
                    slice(max(0, fp_ij["ul"][J]), min(self.N[1], fp_ij["br"][J])),
                ),
            },
            "left": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 2, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        1 if fp_ij["ul"][J] == 0 else 0, fp_ij["ur"][J] - fp_ij["ul"][J]
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    1,
                    slice(max(0, fp_ij["ul"][I]), min(self.N[0], fp_ij["bl"][I])),
                    slice(
                        max(0, fp_ij["ul"][J] - 1), min(self.N[1], fp_ij["br"][J] - 1)
                    ),
                ),
            },
        }

    def pairwise_potential(self, correlation_type=None):
        """
        Compute pairwise potential psi(X_i, X_j) for neighboring cells.
        Options:
            - Uniform: (0.5, 0.5)
            - Biased: Fixed (0.7, 0.3).
            - Adaptive: Based on a metric like Pearson correlation.
        """
        if correlation_type is None and self.correlation_type is not None:
            correlation_type = self.correlation_type
        if correlation_type == "equal":
            return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        elif correlation_type == "biased":
            return np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
        else:
            return np.array(
                adaptive_weights_matrix(self.last_observations), dtype=float
            )

    def get_indices(self, i, j):
        """Compute bounding box corner indices from 1D row/col index arrays."""
        return {
            "ul": np.array([np.min(i), np.min(j)]),
            "bl": np.array([np.max(i) + 1, np.min(j)]),
            "ur": np.array([np.min(i), np.max(j) + 1]),
            "br": np.array([np.max(i) + 1, np.max(j) + 1]),
        }

    def get_belief(self):
        """Return current belief map (probability of occupancy)."""
        return self.map_beliefs

    def update_belief_OG(self, fp_vertices_ij, z, uav_pos):
        """
        Update the belief map using a probabilistic observation model.

        Args:
            fp_vertices_ij (dict): Bounding box corner indices of the current field of view.
            z (ndarray): Binary sensor observations (0 = free, 1 = occupied).
            uav_pos (object): UAV (xy_position, altitude).
        """
        I, J = 0, 1
        if self.conf_dict is None:
            raise ValueError(
                "conf_dict must be provided with altitude->(sigma0, sigma1) mapping."
            )
        alt = np.round(uav_pos.altitude, decimals=2)
        self.sigma0, self.sigma1 = self.conf_dict[alt]

        # Likelihood of each state (0 or 1)
        likelihood_m_zero = np.where(z == 0, 1 - self.sigma0, self.sigma0)
        likelihood_m_one = np.where(z == 0, self.sigma1, 1 - self.sigma1)

        assert np.all(np.greater_equal(likelihood_m_one, 0.0)) and np.all(
            np.less_equal(likelihood_m_one, 1.0)
        )
        assert np.all(np.greater_equal(likelihood_m_zero, 0.0)) and np.all(
            np.less_equal(likelihood_m_zero, 1.0)
        )
        # Extract prior beliefs from current map
        prior = self.map_beliefs[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ]

        # Compute unnormalized posteriors
        posterior_m_zero = likelihood_m_zero * (1.0 - prior)
        posterior_m_one = likelihood_m_one * prior

        # Normalize
        denominator = posterior_m_zero + posterior_m_one + self.eps
        assert np.all(np.greater_equal(denominator, 0.0))
        posterior_m_one_norm = posterior_m_one / denominator

        # Recheck the normalization
        assert np.all(np.greater_equal(posterior_m_one_norm, 0.0))
        assert np.all(np.less_equal(posterior_m_one_norm, 1.0))
        # Update belief
        self.map_beliefs[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ] = posterior_m_one_norm

    def propagate_messages(
        self,
        fp_vertices_ij,
        z,
        max_iterations=1,
        correlation_type=None,
        reset_msgs=False,
    ):
        """
        Run Loopy Belief Propagation (LBP) to propagate local evidence spatially.

        Args:
            fp_vertices_ij (dict): Bounding box corner indices.
            z (ndarray): Binary observations.
            max_iterations (int): Number of LBP iterations.
            correlation_type (str): Type of pairwise potential ('equal', 'biased', 'adaptive').
        """
        self.last_observations = z
        if correlation_type is None:
            correlation_type = self.correlation_type
        
        psi = self.pairwise_potential(correlation_type)

        # reset msgs and msgs_buffer

        if reset_msgs:
            self.msgs[:] = 0.5
            self.msgs_buffer[:] = 0.5
        self.msgs[4, :, :] = self.map_beliefs  # Inject current beliefs
        # Deterministic belief slice (matches MultiAgentMapper news LBP convention)
        # belief_slice = self.direction_to_slicing_data["left"]["product_slice"](
        #     fp_vertices_ij
        # )

        for it in range(max_iterations):
            # Start from current msgs so untouched regions remain stable
            # self.msgs_buffer[:4, :, :] = self.msgs[:4, :, :]
            for direction, data in self.direction_to_slicing_data.items():
                product_slice = data["product_slice"](fp_vertices_ij)
                read_slice = data["read_slice"](fp_vertices_ij)
                write_slice = data["write_slice"](fp_vertices_ij)

                # elementwise multiplication of msgs from neighbors
                mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                mul_1 = np.prod(self.msgs[product_slice], axis=0)

                # matrix-vector multiplication (factor-msg)
                # print(f"CURR PSI: {psi}")
                msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
                msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1

                # normalize the first coordinate of the msg
                norm_msg_1 = msg_1 / (msg_0 + msg_1 + self.eps)
                # buffering
                self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

            # AFTER finishing all directions in this iteration:
            # copy the first 4 channels only
            # the 5th one is the map belief
            self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]
            # copy the first 4 channels only
            # the 5th one is the map belief
            # self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

        # Update belief using final message product
        # Use the spatial slice from "left" (or any direction, they should be the same for the patch)
        # REF simulator uses all 5 channels for the final belief
        spatial_slice = self.direction_to_slicing_data["left"]["product_slice"](
            fp_vertices_ij
        )
        
        # Select all 5 channels: (0,1,2,3,4)
        # spatial_slice[1] is row slice, spatial_slice[2] is col slice
        msgs_slice = self.msgs[:, spatial_slice[1], spatial_slice[2]]

        bel_0 = np.prod(1 - msgs_slice, axis=0)
        bel_1 = np.prod(msgs_slice, axis=0)

        self.map_beliefs[spatial_slice[1], spatial_slice[2]] = bel_1 / (
            bel_0 + bel_1 + self.eps
        )

        assert np.all(
            np.greater_equal(self.map_beliefs[product_slice[1], product_slice[2]], 0.0)
        ) and np.all(
            np.less_equal(self.map_beliefs[product_slice[1], product_slice[2]], 1.0)
        )

    # def update_news_belief_LBP_and_fuse_single(self, zx, zy, z):
    #     """
    #     Update and propagate beliefs in `news_map_beliefs` using a single LBP iteration.

    #     Args:
    #         zx (ndarray): Row indices of the observed patch.
    #         zy (ndarray): Column indices of the observed patch.
    #         z (ndarray): Binary sensor values observartions.
    #     """
    #     fp_vertices_ij = self.get_indices(zx, zy)
    #     I, J = 0, 1
    #     sigma0, sigma1 = self.sigma0, self.sigma1

    #     # Compute observation likelihoods
    #     likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
    #     likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

    #     # Extract prior
    #     prior = self.news_map_beliefs[
    #         0,
    #         0,
    #         fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
    #         fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
    #     ]

    #     # Posterior update
    #     posterior_m_zero = likelihood_m_zero * (1.0 - prior)
    #     posterior_m_one = likelihood_m_one * prior
    #     assert np.all(np.greater_equal(posterior_m_one, 0.0))
    #     posterior_m_one_norm = posterior_m_one / (posterior_m_zero + posterior_m_one)
    #     assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
    #         np.less_equal(posterior_m_one_norm, 1.0)
    #     )

    #     # Write updated beliefs

    #     self.news_map_beliefs[
    #         0,
    #         0,
    #         fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
    #         fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
    #     ] = posterior_m_one_norm

    #     # Reset msgs and msgs_buffer
    #     self.msgs[:] = 0.5
    #     self.msgs_buffer[:] = 0.5
    #     # set msgs last channel with current map belief
    #     self.msgs[4, :, :] = self.news_map_beliefs[0, 0, :, :]

    #     # Run 1-step LBP update
    #     for direction, data in self.direction_to_slicing_data.items():
    #         product_slice = data["product_slice"](fp_vertices_ij)
    #         read_slice = data["read_slice"](fp_vertices_ij)
    #         write_slice = data["write_slice"](fp_vertices_ij)

    #         # elementwise multiplication of msgs
    #         mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
    #         mul_1 = np.prod(self.msgs[product_slice], axis=0)
    #         psi = self.pairwise_potential(self.correlation_type)

    #         # matrix-vector multiplication (factor-msg)
    #         msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
    #         msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1
    #         # normalize the first coordinate of the msg
    #         norm_msg_1 = msg_1 / (msg_0 + msg_1)

    #         # buffering
    #         self.msgs_buffer[write_slice] = norm_msg_1[read_slice]
    #     self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

    #     # Belief update
    #     bel_0 = np.prod(1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0)
    #     bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

    #     self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]] = bel_1 / (
    #         bel_0 + bel_1
    #     )

    #     assert np.all(
    #         np.greater_equal(
    #             self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]],
    #             0.0,
    #         )
    #     ) and np.all(
    #         np.less_equal(
    #             self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]],
    #             1.0,
    #         )
    #     )

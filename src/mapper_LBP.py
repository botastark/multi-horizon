from typing import Dict, List
import numpy as np
import math
from helper import adaptive_weights_matrix


class OccupancyMap:
    def __init__(self, grid_size, conf_dict=None, correlation_type=None):
        self.N = grid_size  # Grid size (100x100)
        self.states = [0, 1]  # Possible states
        self.conf_dict = conf_dict
        # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N[0], self.N[1], 2), 0.5)  # 2 states: [0, 1]
        self.last_observations = np.array([])
        self.msgs = None
        self.msgs_buffer = None
        self.direction_to_slicing_data = None
        self._init_LBP_msgs()
        self.map_beliefs = np.full((self.N[0], self.N[1]), 0.5)
        self.correlation_type = correlation_type

        self.sigma0 = None
        self.sigma1 = None

    def reset(self, conf_dict=None):
        self.conf_dict = conf_dict
        self.phi = np.full((self.N[0], self.N[1], 2), 0.5)  # 2 states: [0, 1]
        self.last_observations = np.array([])
        self._init_LBP_msgs()
        self.map_beliefs = np.full((self.N[0], self.N[1]), 0.5)

        #         global states, map_beliefs, map_belief_entropies, agg_map_belief, agg_map_belief_entropy, news_map_beliefs
        # map_belief_entropies = np.ones((self.n_cell, self.n_cell, self.n_agents), dtype=float)
        # agg_map_belief = np.ones((self.n_cell, self.n_cell), dtype=float) * 0.5
        # agg_map_belief_entropy = np.ones((self.n_cell, self.n_cell), dtype=float)
        self.news_map_beliefs = np.ones((1, 1, self.N[0], self.N[1]), dtype=float) * 0.5

    def _init_LBP_msgs(self):
        # n_cell = self.N
        # depth_to_direction = 0123_4 -> URDL_fake
        self.msgs = np.ones((4 + 1, self.N[0], self.N[1]), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5
        I, J = 0, 1

        # self.pairwise_potential = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

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

    def get_indices(self, x, y):
        grid_length = x[0, 1] - x[0, 0]  # First row, consecutive columns

        i = np.array((x / grid_length).astype(int))  # Convert x to grid indices
        j = np.array((y / grid_length).astype(int))  # Convert y to grid indices
        fp_vertices_ij = {
            "ul": np.array([np.min(i), np.min(j)]),
            "bl": np.array([np.max(i) + 1, np.min(j)]),
            "ur": np.array([np.min(i), np.max(j) + 1]),
            "br": np.array([np.max(i) + 1, np.max(j) + 1]),
        }
        return fp_vertices_ij

    def update_belief_OG(self, zx, zy, z, uav_pos, mexgen=None):
        fp_vertices_ij = self.get_indices(zx, zy)
        I, J = 0, 1
        if mexgen == None:
            a, b = 1, 0.015
            sigma = a * (
                1 - np.exp(-b * uav_pos.altitude)
            )  # Error parameter based on altitude

            if self.conf_dict is not None:
                s0, s1 = self.conf_dict[np.round(uav_pos.altitude, decimals=2)]

            else:
                s0, s1 = sigma, sigma
            self.sigma0 = s0
            self.sigma1 = s1

            likelihood_m_zero = np.where(z == 0, 1 - s0, s0)
            likelihood_m_one = np.where(z == 0, s1, 1 - s1)

        else:
            likelihood_m_one = self.sample_binary_observations(z, uav_pos.altitude)
            likelihood_m_zero = 1 - likelihood_m_one

        assert np.all(np.greater_equal(likelihood_m_one, 0.0)) and np.all(
            np.less_equal(likelihood_m_one, 1.0)
        )
        assert np.all(np.greater_equal(likelihood_m_zero, 0.0)) and np.all(
            np.less_equal(likelihood_m_zero, 1.0)
        )

        posterior_m_zero = likelihood_m_zero * (
            1.0
            - self.map_beliefs[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ]
        )
        posterior_m_one = (
            likelihood_m_one
            * self.map_beliefs[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ]
        )

        assert np.all(np.greater_equal(posterior_m_zero, 0.0))
        assert np.all(np.less_equal(posterior_m_zero, 1.0))

        assert np.all(np.greater_equal(posterior_m_one, 0.0))
        assert np.all(np.less_equal(posterior_m_one, 1.0))
        epsilon = 1e-20  # A small constant to prevent division by zero

        # Normalize posterior_m_one
        denominator = posterior_m_zero + posterior_m_one
        assert np.all(np.greater_equal(denominator, 0.0))  # Optional sanity check
        posterior_m_one_norm = posterior_m_one / (denominator + epsilon)

        # Recheck the normalization
        assert np.all(np.greater_equal(posterior_m_one_norm, 0.0))
        assert np.all(np.less_equal(posterior_m_one_norm, 1.0))

        self.map_beliefs[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ] = posterior_m_one_norm

    def propagate_messages_(
        self, zx, zy, z, uav_pos, max_iterations=5, correlation_type=None
    ):
        # Pairwise potential
        # self.update_belief_OG(zx, zy, z, uav_pos)
        self.last_observations = z

        psi = self.pairwise_potential(correlation_type)

        fp_vertices_ij = self.get_indices(zx, zy)
        # reset msgs and msgs_buffer
        self.msgs = np.ones_like(self.msgs) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5
        self.msgs[4, :, :] = self.map_beliefs[
            :, :
        ]  # set msgs last channel with current map belief
        for _ in range(max_iterations):
            for direction, data in self.direction_to_slicing_data.items():
                # print(direction)
                product_slice = data["product_slice"](fp_vertices_ij)
                read_slice = data["read_slice"](fp_vertices_ij)
                write_slice = data["write_slice"](fp_vertices_ij)

                # elementwise multiplication of msgs
                mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                mul_1 = np.prod(self.msgs[product_slice], axis=0)

                # matrix-vector multiplication (factor-msg)
                msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
                msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1

                # normalize the first coordinate of the msg
                norm_msg_1 = msg_1 / (msg_0 + msg_1)
                # buffering
                self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

            # copy the first 4 channels only
            # the 5th one is the map belief
            self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

        bel_0 = np.prod(1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0)
        bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

        # norm_bel_0 = bel_0 / (bel_0 + bel_1)
        self.map_beliefs[product_slice[1], product_slice[2]] = bel_1 / (bel_0 + bel_1)

        assert np.all(
            np.greater_equal(self.map_beliefs[product_slice[1], product_slice[2]], 0.0)
        ) and np.all(
            np.less_equal(self.map_beliefs[product_slice[1], product_slice[2]], 1.0)
        )
        """
            def update_news_belief_LBP_and_fuse_single(self, zx, zy, z):

                # global news_map_beliefs, map_beliefs
                fp_vertices_ij = self.get_indices(zx, zy)
                I, J = 0, 1
                sigma0, sigma1 = self.sigma0, self.sigma1

                likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
                likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

                posterior_m_zero = likelihood_m_zero * (
                    1.0
                    - self.news_map_beliefs[
                        0,
                        0,
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    ]
                )
                posterior_m_one = (
                    likelihood_m_one
                    * self.news_map_beliefs[
                        0,
                        0,
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    ]
                )

                assert np.all(np.greater_equal(posterior_m_one, 0.0))

                posterior_m_one_norm = posterior_m_one / (posterior_m_zero + posterior_m_one)

                assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
                    np.less_equal(posterior_m_one_norm, 1.0)
                )

                self.news_map_beliefs[
                    0,
                    0,
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                ] = posterior_m_one_norm

                # reset msgs and msgs_buffer
                self.msgs = np.ones_like(self.msgs) * 0.5
                self.msgs_buffer = np.ones_like(self.msgs) * 0.5
                self.msgs[4, :, :] = self.news_map_beliefs[
                    0, 0, :, :
                ]  # set msgs last channel with current map belief
                fp_vertices_ij = self.get_indices(zx, zy)

                # just 1 iteration
                for direction, data in self.direction_to_slicing_data.items():
                    product_slice = data["product_slice"](fp_vertices_ij)
                    read_slice = data["read_slice"](fp_vertices_ij)
                    write_slice = data["write_slice"](fp_vertices_ij)

                    # elementwise multiplication of msgs
                    mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                    mul_1 = np.prod(self.msgs[product_slice], axis=0)

                    psi = self.pairwise_potential(self.correlation_type)

                    # matrix-vector multiplication (factor-msg)
                    # matrix-vector multiplication (factor-msg)
                    msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
                    msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1

                    # normalize the first coordinate of the msg
                    norm_msg_1 = msg_1 / (msg_0 + msg_1)

                    # buffering
                    self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

                    # copy the first 4 channels only
                    # the 5th one is the map belief
                    self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

                    bel_0 = np.prod(
                        1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
                    )
                    bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

                    # norm_bel_0 = bel_0 / (bel_0 + bel_1)
                    self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]] = bel_1 / (
                        bel_0 + bel_1
                    )

                    assert np.all(
                        np.greater_equal(
                            self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]],
                            0.0,
                        )
                    ) and np.all(
                        np.less_equal(
                            self.news_map_beliefs[0, 0, product_slice[1], product_slice[2]],
                            1.0,
                        )
                    )

                    # neighbors_ids = []
                    # if len(observations) != 0:
                    #     neighbors_ids = observations[agent_id]["neighbors_ids"]

                    # for neighbor_id in neighbors_ids:
                    #     mul = (
                    #         news_map_beliefs[agent_id, agent_id, :, :]
                    #         * map_beliefs[:, :, neighbor_id]
                    #     )
                    #     map_beliefs[:, :, neighbor_id] = mul / (
                    #         mul
                    #         + (1.0 - news_map_beliefs[agent_id, agent_id, :, :])
                    #         * (1.0 - map_beliefs[:, :, neighbor_id])
                    #     )

                    #     assert np.all(
                    #         np.greater_equal(map_beliefs[:, :, neighbor_id], 0.0)
                    #     ) and np.all(np.less_equal(map_beliefs[:, :, neighbor_id], 1.0))

                    # if len(neighbors_ids) != 0:
                    #     news_map_beliefs[agent_id, agent_id, :, :] = 0.5
        """

    def get_belief(self):
        return self.map_beliefs

    def sample_binary_observations(self, belief_map, altitude, num_samples=5):
        """
        Samples binary observations from a belief map with noise based on altitude.

        Args:
            belief_map (np.ndarray): Belief map of shape (m, n, 2), where belief_map[..., 1] is P(m=1).
            altitude (float): UAV altitude affecting noise level.
            num_samples (int): Number of samples for averaging.
            noise_factor (float): Base noise factor scaled with altitude.

        Returns:
            np.ndarray: Averaged binary observation map of shape (m, n).
        """
        m, n = belief_map.shape
        sampled_observations = np.zeros((m, n, num_samples))
        a = 0.2
        b = 0.05
        var = a * (1 - np.exp(-b * altitude))
        noise_std = np.sqrt(var)
        # noise_std = noise_factor * altitude  # Noise increases with altitude

        for i in range(num_samples):
            # Sample from the probability map with added Gaussian noise
            noise = np.random.normal(loc=0.0, scale=noise_std, size=(m, n))
            noisy_prob = belief_map + noise  # Add noise to P(m=1)
            noisy_prob = np.clip(noisy_prob, 0, 1)  # Ensure probabilities are valid

            # Sample binary observation
            sampled_observations[..., i] = np.random.binomial(1, noisy_prob)

        # Return the averaged observation map
        return np.mean(sampled_observations, axis=-1)

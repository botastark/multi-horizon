import numpy as np
import math
from src.helper import adaptive_weights_matrix


class OccupancyMap:
    def __init__(self, n_cells):
        self.N = n_cells  # Grid size (100x100)
        self.states = [0, 1]  # Possible states
        # Initialize local evidence (uniform belief)
        self.phi = np.full((self.N, self.N, 2), 0.5)  # 2 states: [0, 1]
        self.last_observations = np.array([])
        self.msgs = None
        self.msgs_buffer = None
        self.direction_to_slicing_data  = None
        self._init_LBP_msgs()
        self.map_beliefs = np.full((self.N, self.N), 0.5)


    def _init_LBP_msgs(self ):
        n_cell = self.N
        # depth_to_direction = 0123_4 -> URDL_fake
        self.msgs = np.ones((4 + 1, n_cell, n_cell), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5
        I, J = 0,1

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
                    slice(max(0, fp_ij["ul"][I] - 1), min(n_cell, fp_ij["bl"][I] - 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
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
                            if fp_ij["ur"][J] == n_cell
                            else fp_ij["ur"][J] - fp_ij["ul"][J]
                        ),
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    3,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] + 1), min(n_cell, fp_ij["br"][J] + 1)),
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
                            if fp_ij["bl"][I] == n_cell
                            else fp_ij["bl"][I] - fp_ij["ul"][I]
                        ),
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    0,
                    slice(max(0, fp_ij["ul"][I] + 1), min(n_cell, fp_ij["bl"][I] + 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
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
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] - 1), min(n_cell, fp_ij["br"][J] - 1)),
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



    def set_last_observations(self, submap):
        self.last_observations = np.array(submap)

    def get_indices(self, x, y):
        grid_length = x[0, 1] - x[0, 0]  # First row, consecutive columns
        print(f"grid length {grid_length}")

        i = np.array((x / grid_length).astype(int))  # Convert x to grid indices
        j = np.array((y / grid_length).astype(int))  # Convert y to grid indices
        fp_vertices_ij = {
            "ul": np.array([np.min(i), np.min(j)]),
            "bl": np.array([np.max(i)+1, np.min(j)]),
            "ur": np.array([np.min(i), np.max(j)+1]),
            "br": np.array([np.max(i)+1, np.max(j)+1]),
        }

        return fp_vertices_ij



    def update_belief_OG(self, zx,zy,z, uav_pos):

        fp_vertices_ij = self.get_indices(zx,zy)
        a, b = 1, 0.015
        sigma = a * (
            1 - np.exp(-b * uav_pos.altitude)
        )  # Error parameter based on altitude
        print(f"sigma in botas' {sigma}")
        I, J=0,1

        likelihood_m_zero = np.where(z == 0, 1 - sigma, sigma)
        likelihood_m_one = np.where(z == 0, sigma, 1 - sigma)


        posterior_m_zero = likelihood_m_zero * (
            1.0
            - self.map_beliefs[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J]
            ]
        )
        posterior_m_one = (
            likelihood_m_one
            * self.map_beliefs[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J]
            ]
        )

        assert np.all(np.greater_equal(posterior_m_one, 0.0))

        # posterior_m_zero_norm = posterior_m_zero / (posterior_m_zero + posterior_m_one)
        posterior_m_one_norm = posterior_m_one / (
            posterior_m_zero + posterior_m_one
        )

        assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
            np.less_equal(posterior_m_one_norm, 1.0)
        )


        self.map_beliefs[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J]] = posterior_m_one_norm
        return likelihood_m_zero


    def propagate_messages_(self, zx, zy, z, uav_pos,  max_iterations=5, correlation_type=None):
        # Pairwise potential
        # self._update_belief_OG(zx,zy,z, uav_pos)
        self.last_observations = z
        
        psi = self.pairwise_potential(correlation_type)

        fp_vertices_ij = self.get_indices(zx,zy)
        # reset msgs and msgs_buffer
        self.msgs = np.ones_like(self.msgs) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5
        self.msgs[4, :, :] = self.map_beliefs[:, :]  # set msgs last channel with current map belief
        for _ in range(max_iterations):
            for direction, data in self.direction_to_slicing_data.items():
                product_slice = data["product_slice"](fp_vertices_ij)
                read_slice = data["read_slice"](fp_vertices_ij)
                write_slice = data["write_slice"](fp_vertices_ij)

                # elementwise multiplication of msgs
                mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                mul_1 = np.prod(self.msgs[product_slice], axis=0)

                # matrix-vector multiplication (factor-msg)
                msg_0 = (psi[0, 0] * mul_0
                    + psi[0, 1] * mul_1
                )
                msg_1 = (
                    psi[1, 0] * mul_0
                    + psi[1, 1] * mul_1
                )

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
        self.map_beliefs[product_slice[1], product_slice[2]] = bel_1 / (
            bel_0 + bel_1
        )

        assert np.all(
            np.greater_equal(
                self.map_beliefs[product_slice[1], product_slice[2]], 0.0
            )
        ) and np.all(
            np.less_equal(
                self.map_beliefs[product_slice[1], product_slice[2]], 1.0
            )
        )

    def get_belief(self):
        return self.map_beliefs
        


    # def _update_news_belief_LBP_and_fuse_single(self, observations):

    #     global news_map_beliefs, map_beliefs

    #     z, fp_vertices_ij =
    #     sigma0, sigma1 = 

    #     likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
    #     likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

    #     posterior_m_zero = likelihood_m_zero * (
    #         1.0
    #         - news_map_beliefs[
    #             fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
    #             fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
    #         ]
    #     )
    #     posterior_m_one = (
    #         likelihood_m_one
    #         * news_map_beliefs[
    #             fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
    #             fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
    #         ]
    #     )

    #     assert np.all(np.greater_equal(posterior_m_one, 0.0))

    #     posterior_m_one_norm = posterior_m_one / (
    #         posterior_m_zero + posterior_m_one
    #     )

    #     assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
    #         np.less_equal(posterior_m_one_norm, 1.0)
    #     )

    #     news_map_beliefs[
    #         fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
    #         fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
    #     ] = posterior_m_one_norm


    #     # reset msgs and msgs_buffer
    #     self.msgs = np.ones_like(self.msgs) * 0.5
    #     self.msgs_buffer = np.ones_like(self.msgs) * 0.5
    #     self.msgs[4, :, :] = news_map_beliefs[
    #         agent_id, agent_id, :, :
    #     ]  # set msgs last channel with current map belief

    #     fp_vertices_ij = observations[agent_id]["fp_ij"]

    #     # just 1 iteration
    #     for direction, data in self.direction_to_slicing_data.items():
    #         product_slice = data["product_slice"](fp_vertices_ij)
    #         read_slice = data["read_slice"](fp_vertices_ij)
    #         write_slice = data["write_slice"](fp_vertices_ij)

    #         # elementwise multiplication of msgs
    #         mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
    #         mul_1 = np.prod(self.msgs[product_slice], axis=0)

    #         # matrix-vector multiplication (factor-msg)
    #         msg_0 = (pairwise_potential[0, 0] * mul_0
    #             + pairwise_potential[0, 1] * mul_1
    #         )
    #         msg_1 = (
    #             pairwise_potential[1, 0] * mul_0
    #             + pairwise_potential[1, 1] * mul_1
    #         )

    #         # normalize the first coordinate of the msg
    #         norm_msg_1 = msg_1 / (msg_0 + msg_1)

    #         # buffering
    #         self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

    #     # copy the first 4 channels only
    #     # the 5th one is the map belief
    #     self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

    #     bel_0 = np.prod(
    #         1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
    #     )
    #     bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

    #     # norm_bel_0 = bel_0 / (bel_0 + bel_1)
    #     news_map_beliefs[product_slice[1], product_slice[2]] = (
    #         bel_1 / (bel_0 + bel_1)
    #     )

    #     assert np.all(
    #         np.greater_equal(
    #             news_map_beliefs[
    #                  product_slice[1], product_slice[2]
    #             ],
    #             0.0,
    #         )
    #     ) and np.all(
    #         np.less_equal(
    #             news_map_beliefs[
    #                 product_slice[1], product_slice[2]
    #             ],
    #             1.0,
    #         )
    #     )


    #     neighbors_ids = []
    #     if len(observations) != 0:
    #         neighbors_ids = observations[agent_id]["neighbors_ids"]

    #     for neighbor_id in neighbors_ids:
    #         mul = (
    #             news_map_beliefs[agent_id, agent_id, :, :]
    #             * map_beliefs[:, :, neighbor_id]
    #         )
    #         map_beliefs[:, :, neighbor_id] = mul / (
    #             mul
    #             + (1.0 - news_map_beliefs[agent_id, agent_id, :, :])
    #             * (1.0 - map_beliefs[:, :, neighbor_id])
    #         )

    #         assert np.all(
    #             np.greater_equal(map_beliefs[:, :, neighbor_id], 0.0)
    #         ) and np.all(np.less_equal(map_beliefs[:, :, neighbor_id], 1.0))

    #     if len(neighbors_ids) != 0:
    #         news_map_beliefs[agent_id, agent_id, :, :] = 0.5

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

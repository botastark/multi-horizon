# pairwise_factor_weights: equal, biased, adaptive
import random
import numpy as np

import math


def collect_sample_set(grid):
    rows, cols = grid.shape
    D = []

    # Calculate number of complete 3x3 grids
    num_blocks_row = rows // 3
    num_blocks_col = cols // 3

    # Iterate over the grid in steps of 3 to access central cells of each 3x3 block
    for block_i in range(num_blocks_row):
        for block_j in range(num_blocks_col):
            # Central cell in a 3x3 block
            central_i = block_i * 3 + 1
            central_j = block_j * 3 + 1
            c = grid[central_i, central_j]

            # Collect Von Neumann neighbors
            neighbors = [
                grid[central_i - 1, central_j],  # North
                grid[central_i + 1, central_j],  # South
                grid[central_i, central_j - 1],  # West
                grid[central_i, central_j + 1],  # East
            ]

            n = sum(neighbors)

            # Append the central cell value and the sum of neighbors
            D.append((c, n))
    return D


def pearson_correlation_coeff(d_sampled):
    c_values = [c for c, n in d_sampled]
    n_values = [n for c, n in d_sampled]
    avg_c = np.mean(c_values)
    avg_n = np.mean(n_values)
    p = 0
    numerator = 0
    sum_sq_central_diff = 0
    sum_sq_neighbors_diff = 0
    for c, n in d_sampled:
        c_diff = c - avg_c
        n_diff = n - avg_n
        numerator += n_diff * c_diff
        sum_sq_central_diff += c_diff**2
        sum_sq_neighbors_diff += n_diff**2
    denominator = np.sqrt(sum_sq_central_diff * sum_sq_neighbors_diff)
    p = numerator / denominator if denominator != 0 else 0
    return p


def adaptive_weights(m_i, m_j, obs_map):

    d_sampled = collect_sample_set(obs_map)
    p = pearson_correlation_coeff(d_sampled)
    exp = np.exp(-p)

    if m_i == m_j:
        return 1 / (1 + exp)
    else:
        return exp / (1 + exp)


def pairwise_factor(m_i, m_j, obs_map=[], type="equal"):
    if type == "equal":
        return 0.5
    elif type == "biased":
        if m_i == m_j:
            return 0.7
        else:
            return 0.3
    else:
        return adaptive_weights(m_i, m_j, obs_map)


def sensor_model(m, z, altitude):
    # probability of measurement z given cell state m ande uav position x
    a = 1
    b = 0.015
    sigma_s = a * (1 - np.exp(-b * altitude))

    if z == m:
        return (
            1 - sigma_s
        )  # Get the probability of observing the true state value at this altitude
    else:
        return sigma_s


def get_edges(m0, i, j):
    rows = len(m0)
    cols = len(m0[0]) if rows > 0 else 0
    possible_neighbors = [
        (i - 1, j),  # Top
        (i + 1, j),  # Bottom
        (i, j - 1),  # Left
        (i, j + 1),  # Right
    ]
    neighbors = [
        (ni, nj) for ni, nj in possible_neighbors if 0 <= ni < rows and 0 <= nj < cols
    ]
    return neighbors


def mapping(belief_map, obs_map, altitude):
    belief_z = belief_map.get_map()
    # obs_x_min, obs_x_max = obs_map.x_range
    # obs_y_min, obs_y_max = obs_map.y_range
    # for i in range(belief_z.shape[0]):
    #     for j in range(belief_z.shape[1]):
    #         m_pos = belief_map.grid2pos((i, j))
    #         m = belief_z[i][j]
    #         if obs_x_min <= m_pos[0] <= obs_x_max and obs_y_min <= m_pos[1] <= obs_y_max: # if m was observed
    #             z_coords = obs_map.pos2grid(m_pos)
    #             z = obs_map.map[z_coords[0], z_coords[1]]
    #             evidence_factor = sensor_model(m, z)
    #             # print("M pos {}-{} and coord {}-{}".format(m_pos[0], m_pos[1], i, j))
    #             # print("Z pos {}-{} and coord {}-{}".format(m_pos[0], m_pos[1], z_coords[0], z_coords[1]))

    obs_z = obs_map.get_map()
    # observed_belief = obs_map
    # mo = np.zeros_like(obs_z)
    prior = belief_map

    for i in range(obs_z.shape[0]):
        for j in range(obs_z.shape[1]):
            z_pos = obs_map.grid2pos((i, j))
            z_i = obs_z[i][j]

            m_coord = belief_map.pos2grid(z_pos)
            m = belief_map.map[m_coord[0], m_coord[1]]
            m_i = m
            # prior.map[i,j]
            evidence_factor = sensor_model(m_i, z_i, altitude)
            pairwise_factor = 1

            edges_ids = get_edges(belief_map.map, i, j)

            for edge_id in edges_ids:
                m_j = belief_map.map[edge_id]
                pairwise_factor *= adaptive_weights(m_i, m_j, obs_z)
            prior.map[i, j] = evidence_factor * pairwise_factor

            # evidence_factor = sensor_model(m, z, altitude)
            # print("Z pos {}-{} and coord {}-{}".format(z_pos[0], z_pos[1], i, j))
            # print("M pos {}-{} and coord {}-{}".format(z_pos[0], z_pos[1], m_coord[0], m_coord[1]))


def id_converter(map_s, coord_s, map_f):
    """
    given coordinates of map_s (i_s, j_s)
    return corresponding coordinates of map2 (i_f, j_f)
    """
    pos = map_s.grid2pos(coord_s)  # (x, y) for (i_s, j_s)
    print(pos)
    return map_f.pos2grid(pos)


def prob_uav_position(uav_pos, belief_map):
    if (
        not uav_pos.position[0] in belief_map.x_range
        or not uav_pos.position[1] in belief_map.y_range
    ):
        return 0
    else:
        return 1 / len(belief_map)[0] / len(belief_map)[1]


def count_possible_states(uav_pos, map):
    # possible_actions = {"up", "down", "front","back", "left","right", "hover"}
    count = 1  # hover
    max_h = 20
    min_h = 5
    step_h = 5
    step_xy = map.grid.length

    if uav_pos.altitude + step_h <= max_h:  # up
        count += 1
    if uav_pos.altitude - step_h >= min_h:  # down
        count += 1
    if uav_pos.position[1] + step_xy <= map.y_range[1]:  # front (+y)
        count += 1
    if uav_pos.position[1] - step_xy <= map.y_range[0]:  # back (-y)
        count += 1
    if uav_pos.position[0] + step_xy <= map.x_range[1]:  # right (+x)
        count += 1
    if uav_pos.position[0] - step_xy <= map.x_range[0]:  # left (-x)
        count += 1
    return count


def prob_candidate_uav_position(uav_pos, belief_map):
    """
    P(x_{t+1}) = P(x_{t+1} | x_{t}) * P(x_{t})
    P(x_{t+1} | x_{t}) = 1 over all possible future positions from the current position
    """
    prob_candidate_given_current = 1 / count_possible_states(uav_pos, belief_map)
    return prob_candidate_given_current * prob_uav_position(uav_pos, belief_map)


def normalize_2d_grid(grid):
    # Convert input to a NumPy array (in case it's a list)
    grid = np.array(grid, dtype=float)

    # Get the minimum and maximum values from the grid
    min_val = np.min(grid)
    max_val = np.max(grid)

    # Normalize the grid by scaling it to range [0, 1]
    normalized_grid = (grid - min_val) / (max_val - min_val)

    return normalized_grid


def calc_prob(belief_prob, belief_map, obs_map, uav, observed_k):
    belief_map_new = belief_map
    belief_map_new.set_map(belief_prob.get_map())

    for i, (i_b, j_b) in enumerate(observed_k):
        m_i = belief_map.map[i_b, j_b]
        i_o, j_o = id_converter(belief_map, (i_b, j_b), obs_map)
        z_i = obs_map.map[i_o, j_o]
        evidence_factor = sensor_model(m_i, z_i, uav.altitude)
        pairwise_factor = 1
        for j, (i_b_, j_b_) in enumerate(observed_k):
            adaptive_weights()
            m_j = belief_map.map[i_b_, j_b_]
            pairwise_factor *= adaptive_weights(m_i, m_j, obs_map.map)
        new_m_i = evidence_factor * pairwise_factor
        belief_map_new.map[i_b, j_b] = new_m_i
    belief_map_new.set_map(normalize_2d_grid(belief_map_new.map))
    return belief_map_new


def calc_future_obs_prob(belief_prob, belief_map, obs_map, uav, observed_k):
    for i, (i_b, j_b) in enumerate(observed_k):
        m_i = belief_map.map[i_b, j_b]
        i_o, j_o = id_converter(belief_map, (i_b, j_b), obs_map)
        z_i = obs_map.map[i_o, j_o]
        evidence_factor = sensor_model(m_i, z_i, uav.altitude)

        prior_m_i = belief_prob.map[i_b, j_b]  # prob of m_i
        prob_future_observation = evidence_factor * prob_candidate_uav_position(
            uav, belief_map
        )


def map_(belief_map, obs_map, uav, belief_prob=[]):

    if belief_prob == []:
        belief_prob = belief_map

    [observed_belief_i_min, observed_belief_j_min] = id_converter(
        obs_map, [0, 0], belief_map
    )
    [observed_belief_i_max, observed_belief_j_max] = id_converter(
        obs_map, [obs_map.map.shape[0] - 1, obs_map.map.shape[1] - 1], belief_map
    )
    # print("obs_pos x{} - y{}".format( x_o_pos, y_o_pos))
    # print("obs_bel coord x{}-{} - y{}-{}".format( observed_belief_i_min, observed_belief_i_max, observed_belief_j_min, observed_belief_j_max))
    # print("obs_bel pos x{}-{}".format( belief_map.x[observed_belief_i_min, observed_belief_i_max], belief_map.x[observed_belief_j_min, observed_belief_j_max]))
    # print("obs_bel pos y{}-{}".format( belief_map.y[observed_belief_i_min, observed_belief_i_max], belief_map.y[observed_belief_j_min, observed_belief_j_max]))
    observed_k = []

    for i_b, i_o in zip(
        range(observed_belief_i_min, observed_belief_i_max), range(obs_map.x_range[1])
    ):
        for j_b, j_o in zip(
            range(observed_belief_j_min, observed_belief_j_max),
            range(obs_map.y_range[1]),
        ):
            observed_k.append((i_b, j_b))

    future_belief = calc_prob(belief_prob, belief_map, obs_map, uav, observed_k)
    for i, (i_b, j_b) in enumerate(observed_k):
        m_i = belief_map.map[i_b, j_b]
        i_o, j_o = id_converter(belief_map, (i_b, j_b), obs_map)
        z_i = obs_map.map[i_o, j_o]
        evidence_factor = sensor_model(m_i, z_i, uav.altitude)
        prior_m_i = belief_prob.map[i_b, j_b]  # prob of m_i
        prob_future_observation = evidence_factor * prob_candidate_uav_position(
            uav, belief_map
        )
        posterior_m_i = future_belief.map[i_b, j_b]
        entropy = -prior_m_i * math.log2(prior_m_i) - (1 - prior_m_i) * math.log2(
            1 - prior_m_i
        )

        entropy_post = -posterior_m_i * math.log2(posterior_m_i) - (
            1 - posterior_m_i
        ) * math.log2(1 - posterior_m_i)
        expected_entropy = prob_future_observation * entropy_post

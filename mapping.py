# pairwise_factor_weights: equal, biased, adaptive
import random
import numpy as np



import numpy as np

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
    for (c, n) in d_sampled:
        c_diff = c - avg_c
        n_diff = n - avg_n
        numerator += n_diff *  c_diff
        sum_sq_central_diff += c_diff ** 2
        sum_sq_neighbors_diff += n_diff ** 2
    denominator = np.sqrt(sum_sq_central_diff * sum_sq_neighbors_diff)
    p = numerator / denominator if denominator != 0 else 0
    return p

def adaptive_weights(m_i, m_j, obs_map):
    
    d_sampled = collect_sample_set(obs_map)
    p = pearson_correlation_coeff(d_sampled)
    exp = np.exp(-p)
    
    if m_i == m_j:
        return 1/(1+exp)
    else:
        return exp/(1+exp)



def pairwise_factor(m_i, m_j, obs_map=[], type = "equal"):
    if type=="equal":
        return 0.5
    elif type == "biased":
        if m_i==m_j:
            return 0.7
        else:
            return 0.3
    else:
        return adaptive_weights(m_i, m_j, obs_map)

def sensor_model(m, z, altitude):
  # probability of observation z given cell state m ande uav position x
  a = 1
  b = 0.015
  sigma_s = a * (1 - np.exp(- b * altitude))
  
  if z==m: 
    return 1 - sigma_s # Get the probability of observing the true observation value at this altitude
  else:
     return sigma_s
def get_edges(m0, i, j):
    rows = len(m0)
    cols = len(m0[0]) if rows > 0 else 0
    possible_neighbors = [
        (i-1, j),  # Top
        (i+1, j),  # Bottom
        (i, j-1),  # Left
        (i, j+1)   # Right
    ]
    neighbors = [(ni, nj) for ni, nj in possible_neighbors if 0 <= ni < rows and 0 <= nj < cols]
    return neighbors

def mapping(belief_map, obs_map, altitude):
    # belief_z = belief_map.get_map()
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
            # mo[i][j] = m
            m_i = m
            # prior.map[i,j]
            evidence_factor = sensor_model(m_i, z_i, altitude)
            pairwise_factor = 1

            edges_ids = get_edges(belief_map.map, i, j)

            for edge_id in edges_ids:
                m_j = belief_map.map[edge_id]
                pairwise_factor *= adaptive_weights(m_i, m_j, obs_z)
            prior.map[i, j] =  evidence_factor*pairwise_factor


            # evidence_factor = sensor_model(m, z, altitude)
            # print("Z pos {}-{} and coord {}-{}".format(z_pos[0], z_pos[1], i, j))
            # print("M pos {}-{} and coord {}-{}".format(z_pos[0], z_pos[1], m_coord[0], m_coord[1]))



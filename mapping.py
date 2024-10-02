# pairwise_factor_weights: equal, biased, adaptive
import random
import numpy as np


def collect_sample_set(grid, h):
    rows, cols = grid.shape
    D = []
    for i in range(rows):
        for j in range(cols):
            c = grid[i, j]
            
            # Von Neumann neighborhood with edges 0
            neighbors = [
                grid[i - 1, j] if i > 0 else 0,  # North
                grid[i + 1, j] if i < rows - 1 else 0,  # South
                grid[i, j - 1] if j > 0 else 0,  # West
                grid[i, j + 1] if j < cols - 1 else 0   # East
            ]
            
            n = sum(1 for val in neighbors if val > 0)
            D.append((c, n))
    
    # Randomly sample h pairs from D
    D_sampled = random.sample(D, min(h, len(D)))
    return D_sampled


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
        numerator += n_diff *c_diff
        sum_sq_central_diff += c_diff ** 2
        sum_sq_neighbors_diff += n_diff ** 2
    denominator = np.sqrt(sum_sq_central_diff * sum_sq_neighbors_diff)
    p = numerator / denominator if denominator != 0 else 0
    return p

def adaptive_weights(m_i, m_j, obs_map):
    # total_number_of_cells_footprint/9 -> sample_set_size_h h
    h = obs_map.shape[0]*obs_map.shape[1]/9
    d_sampled = collect_sample_set(obs_map, h)
    p = pearson_correlation_coeff(d_sampled)
    exp = np.exp(-p)
    
    if m_i == m_j:
        return 1/(1+exp)
    else:
        return exp/(1+exp)



def get_w(m_i, m_j, obs_map=[], type = "equal"):
    if type=="equal":
        return 0.5
    elif type == "biased":
        if m_i==m_j:
            return 0.7
        else:
            return 0.3
    else:
        return adaptive_weights(m_i, m_j, obs_map)
        
        


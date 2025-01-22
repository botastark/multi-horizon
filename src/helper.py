# pairwise_factor_weights: equal, biased, adaptive
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def collect_sample_set(grid):
    # Create an array of central cells for each 3x3 block (using slices)
    rows, cols = grid.shape

    valid_rows = (rows // 3) * 3
    valid_cols = (cols // 3) * 3

    # remove remainer rows and cols % 3
    truncated_grid = grid[:valid_rows, :valid_cols]

    central_cells = truncated_grid[1::3, 1::3]

    # Create a matrix of neighbors for each central cell using slicing
    north = truncated_grid[0::3, 1::3]  # One row above central cells
    south = truncated_grid[2::3, 1::3]  # One row below central cells
    west = truncated_grid[1::3, 0::3]  # One column to the left
    east = truncated_grid[1::3, 2::3]  # One column to the right

    neighbors = np.stack([north, south, west, east], axis=-1)

    neighbor_sums = np.sum(neighbors, axis=-1)

    return np.column_stack((central_cells.flatten(), neighbor_sums.flatten()))


def pearson_correlation_coeff(d_sampled):
    c_values = d_sampled[:, 0]  # Central cell values
    n_values = d_sampled[:, 1]  # Neighbor sums

    avg_c = np.mean(c_values)
    avg_n = np.mean(n_values)

    # Vectorized calculations for the Pearson correlation
    c_diff = c_values - avg_c
    n_diff = n_values - avg_n

    numerator = np.sum(c_diff * n_diff)
    sum_sq_central_diff = np.sum(c_diff**2)
    sum_sq_neighbors_diff = np.sum(n_diff**2)

    denominator = np.sqrt(sum_sq_central_diff * sum_sq_neighbors_diff)

    return numerator / denominator if denominator != 0 else 0


def adaptive_weights_matrix(obs_map):
    d_sampled = collect_sample_set(obs_map)
    p = pearson_correlation_coeff(d_sampled)
    exp = np.exp(-p)
    psi = np.array(
        [
            [1 / (1 + exp), exp / (1 + exp)],  # For (m_i=0, m_j=0) and (m_i=0, m_j=1)
            [exp / (1 + exp), 1 / (1 + exp)],  # For (m_i=1, m_j=0) and (m_i=1, m_j=1)
        ]
    )
    return psi


def observed_m_ids(uav=None, uav_pos=None, aslist=True):
    if uav != None and uav_pos != None:
        [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = uav.get_range(
            position=uav_pos.position, altitude=uav_pos.altitude, index_form=True
        )
    else:
        raise TypeError("Pass either z or uav_position")
    if aslist:

        observed_m = []
        for i_b in range(obsd_m_i_min, obsd_m_i_max):
            for j_b in range(obsd_m_j_min, obsd_m_j_max):
                observed_m.append((i_b, j_b))
        return observed_m
    else:
        return [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]]


class uav_position:
    def __init__(self, input) -> None:

        self.position = input[0]
        self.altitude = input[1]

    def __eq__(self, other):
        if isinstance(other, uav_position):
            return self.position == other.position and self.altitude == other.altitude
        return False

    def __hash__(self):
        return hash((self.position, self.altitude))


def compute_mse(ground_truth_map, estimated_map):
    if ground_truth_map.shape != estimated_map.shape:
        raise ValueError("Input maps must have the same dimensions for MSE")
    mse = np.mean((ground_truth_map - estimated_map) ** 2)
    return mse


def compute_coverage(ms_set, grid):
    # ms = observed_m_ids(uav, pos)
    cell_area = grid.length * grid.length
    observed_area = len(ms_set) * cell_area
    total_area = grid.x * grid.y
    return observed_area / total_area


def compute_entropy(belief):
    assert np.all(np.greater_equal(belief, 0.0)), f"{belief[np.isnan(belief)]}"
    assert np.all(np.less_equal(belief, 1.0)), f"{belief[np.isnan(belief)]}"

    if belief.ndim == 3:
        v1 = belief[:, :, 0]
        v2 = belief[:, :, 1]
    else:
        v1 = belief
        v2 = 1.0 - belief
    if isinstance(belief, np.ndarray):
        v1 = np.where(v1 == 0.0, 1.0, v1)
        v2 = np.where(v2 == 0.0, 1.0, v2)
    else:
        if v1 == 0.0:
            v1 = 1.0
        if v2 == 0.0:
            v2 = 1.0

    l1 = np.log2(v1)
    l2 = np.log2(v2)
    assert np.all(np.less_equal(l1, 0.0))
    assert np.all(np.less_equal(l2, 0.0))

    entropy = np.sum(-(v1 * l1 + v2 * l2))

    assert np.all(np.greater_equal(entropy, 0.0))

    return entropy.astype(float)


def compute_metrics(ground_truth_map, belief, ms_set, grid):
    if belief.ndim == 3:
        estimated_map = (belief[..., 1] >= 0.5).astype(np.uint8)
    else:
        estimated_map = (belief >= 0.5).astype(np.uint8)
    mse = compute_mse(ground_truth_map, estimated_map)
    entropy = compute_entropy(belief)
    coverage = compute_coverage(ms_set, grid)

    return (entropy, mse, coverage)


class FastLogger:
    HEADER = "step\tentropy\tmse\theight\tcoverage\n"

    def __init__(
        self,
        dir,
        strategy="ig",
        pairwise="equal",
        n_agent=1,
        grid=None,
        r=None,
        init_x=None,
    ):

        self.strategy = strategy
        self.pairwise = pairwise
        self.n = n_agent
        self.grid = grid
        self.init_x = init_x
        self.step = 0
        self.r = r
        self.filename = (
            dir + "/" + self.strategy + "_" + self.pairwise + "_" + str(self.n) + ".txt"
        )

        with open(self.filename, "w") as f:
            f.write(f"Strategy: {self.strategy}\n")
            f.write(f"Pairwise: {self.pairwise}\n")
            f.write(f"N agents: {self.n}\n")
            f.write(
                f"Grid info: range: 0-{self.grid.x}-{self.grid.y}, cell_size:{self.grid.length}, map shape: {self.grid.shape}"
            )
            if isinstance(self.r, str):
                f.write(f"using {self.r} \n")
            else:
                f.write(f"Gaussian radius {self.r} \n")
            f.write(
                f"init UAV position: {self.init_x.position} - {self.init_x.altitude} \n"
            )
            # Print table header with aligned columns
            f.write(
                f"{'Step':<6} {'Entropy':<10} {'MSE':<8} {'Height':<8} {'Coverage':<10}\n"
            )
            f.write("-" * 48)  # Divider line
            f.write("\n")

    def log_data(self, entropy, mse, height, coverage):
        with open(self.filename, "a") as f:
            f.write(
                f"{self.step:<6} {round(entropy, 2):<10} {round(mse, 4):<8} {round(height, 1):<8} {round(coverage, 4):<10}\n"
            )

            f.flush()
        self.step += 1

    def log(self, text):
        with open(self.filename, "a") as f:
            f.write(text + "\n")
            f.flush()

    def collect_data(self, filename=None):
        filename = filename or self.filename
        info = {"strategy": None, "pairwise": None, "agents": None}
        entropy, mse, height, coverage = [], [], [], []

        try:
            with open(filename, "r") as f:
                lines = f.readlines()

            info["strategy"] = lines[0].strip()
            info["pairwise"] = lines[1].strip()
            info["agents"] = lines[2].strip()

            for line in lines[3:]:
                raw = line.split("\t")
                entropy.append(float(raw[1]))
                mse.append(float(raw[2]))
                height.append(float(raw[3]))
                coverage.append(float(raw[4]))

        except (IOError, IndexError, ValueError) as e:
            print(f"Error reading or parsing data: {e}")

        return info, (entropy, mse, height, coverage)


import os
import pickle


def gaussian_random_field(cluster_radius, n_cell, cache_dir="cache"):
    """
    Generate a 2D Gaussian random field and cache the results for reuse.
     https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
    Parameters:
    - cluster_radius: Correlation radius for the Gaussian field.
    - n_cell: Size of the field (n_cell_x x n_cell_y).

    - cache_dir: Directory to store cached fields (default: "cache").

    Returns:
    - 2D binary random field as a numpy array.
    """

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    n_cell_x, n_cell_y = n_cell

    # Generate cache filename
    cache_file = os.path.join(
        cache_dir, f"field_radius_{cluster_radius}_size_{n_cell_x}x{n_cell_y}.pkl"
    )

    # Try loading from cache
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print(f"Loading cached field from {cache_file}")
            return pickle.load(f)

    # Helper functions
    def _fft_indices(n):
        a = list(range(0, int(np.floor(n / 2)) + 1))
        b = reversed(range(1, int(np.floor(n / 2))))
        b = [-i for i in b]
        return a + b

    def _pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        val = np.sqrt(np.sqrt(kx**2 + ky**2) ** (-cluster_radius))
        return val

    # Generate amplitude for the given cluster_radius
    map_rng = np.random.default_rng(123)
    amplitude = np.zeros((n_cell_x, n_cell_y))
    fft_indices_x = _fft_indices(n_cell_x)
    fft_indices_y = _fft_indices(n_cell_y)

    for i, kx in enumerate(fft_indices_x):
        for j, ky in enumerate(fft_indices_y):
            amplitude[i, j] = _pk2(kx, ky)

    # Generate Gaussian random field
    noise = np.fft.fft2(map_rng.normal(size=(n_cell_x, n_cell_y)))
    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (
        np.max(random_field) - np.min(random_field)
    )

    # Make field binary
    normalized_random_field[normalized_random_field >= 0.5] = 1
    normalized_random_field[normalized_random_field < 0.5] = 0

    binary_field = normalized_random_field.astype(np.uint8)

    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(binary_field, f)
    print(f"Field generated and saved to {cache_file}")

    return binary_field


import math


def get_range(uav_pos, grid, index_form=False):
    """
    calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
    """
    # position = position if position is not None else self.position
    # altitude = altitude if altitude is not None else self.altitude
    fov = np.pi / 3
    x_angle = fov / 2  # rads
    y_angle = fov / 2  # rads
    x_dist = uav_pos.altitude * math.tan(x_angle)
    y_dist = uav_pos.altitude * math.tan(y_angle)

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


# def get_observations(grid_info, ground_truth_map, uav_pos):
#     [[x_min_id, x_max_id], [y_min_id, y_max_id]] = get_range(
#         uav_pos, grid_info, index_form=True
#     )
#     submap = ground_truth_map[x_min_id:x_max_id, y_min_id:y_max_id]

#     x = np.arange(
#         x_min_id * grid_info.length, x_max_id * grid_info.length, grid_info.length
#     )
#     y = np.arange(
#         y_min_id * grid_info.length, y_max_id * grid_info.length, grid_info.length
#     )

#     x, y = np.meshgrid(x, y, indexing="ij")

#     return x, y, submap


def get_observations(grid_info, ground_truth_map, uav_pos, seed=None, mexgen=None):
    [[x_min_id, x_max_id], [y_min_id, y_max_id]] = get_range(
        uav_pos, grid_info, index_form=True
    )
    m = ground_truth_map[x_min_id:x_max_id, y_min_id:y_max_id]
    # if tile_dict is not None:
    #     m = observed_submap((x_min_id,x_max_id), (y_min_id,y_max_id), tile_dict)

    if mexgen is not None:
        success1 = sample_binary_observations(m, uav_pos.altitude)
        success0 = 1 - success1
    else:
        if seed is None:
            seed = 1
        rng = np.random.default_rng(seed)
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * uav_pos.altitude))
        random_values = rng.random(m.shape)
        success0 = random_values <= 1.0 - sigma
        success1 = random_values <= 1.0 - sigma

    z0 = np.where(np.logical_and(success0, m == 0), 0, 1)
    z1 = np.where(np.logical_and(success1, m == 1), 1, 0)
    z = np.where(m == 0, z0, z1)

    x = np.arange(x_min_id, x_max_id)
    y = np.arange(y_min_id, y_max_id)

    x, y = np.meshgrid(x, y, indexing="ij")

    return x, y, z


def sample_binary_observations(belief_map, altitude, num_samples=5):
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

    for i in range(num_samples):
        # Sample from the probability map with added Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(m, n))
        noisy_prob = belief_map + noise  # Add noise to P(m=1)
        noisy_prob = np.clip(noisy_prob, 0, 1)  # Ensure probabilities are valid

        # Sample binary observation
        sampled_observations[..., i] = np.random.binomial(1, noisy_prob)

    # Return the averaged observation map
    return np.mean(sampled_observations, axis=-1)


"""
sampling updates start here
"""


def sigma(altitude):
    a = 1
    b = 0.015
    return a * (1 - np.exp(-b * altitude))


def sensor_model(true_matrix, altitude):
    sig = sigma(altitude)
    P_z_equals_m = 1 - sig
    P_z_not_equals_m = sig

    rows, cols = true_matrix.shape
    observation_matrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if true_matrix[i, j] == 1:
                observation_matrix[i, j] = np.random.choice(
                    [1, 0], p=[P_z_equals_m, P_z_not_equals_m]
                )
            else:
                observation_matrix[i, j] = np.random.choice(
                    [0, 1], p=[P_z_equals_m, P_z_not_equals_m]
                )

    return observation_matrix


def sampler(true_matrix, altitude, N):
    cumulative_observation = np.empty(true_matrix.shape)
    for i in range(N):
        observation_matrix = sensor_model(true_matrix, altitude)
        if i == 0:
            cumulative_observation = observation_matrix.copy()
            continue
        cumulative_observation = np.vstack([cumulative_observation, observation_matrix])
    return cumulative_observation


def calculate_statistics(true_matrix_, observation):
    n = int(observation.shape[0] / true_matrix_.shape[0])

    true_matrix = np.tile(true_matrix_, (n, 1))
    true_positive = np.sum((true_matrix == 1) & (observation == 1))
    false_negative = np.sum((true_matrix == 1) & (observation == 0))
    false_positive = np.sum((true_matrix == 0) & (observation == 1))
    true_negative = np.sum((true_matrix == 0) & (observation == 0))

    return {
        "True Positive": true_positive,
        "False Negative": false_negative,
        "False Positive": false_positive,
        "True Negative": true_negative,
    }


import json


def get_s0_s1(true_matrix, altitude, e=0.3):
    with open("/home/bota/Desktop/active_sensing/data/N_per_E.json", "r") as file:
        loaded_dict = json.load(file)
    altitude = round(altitude, 2)
    N = int(float(loaded_dict[str(altitude)][str(e)]))
    # # print(f"total n tiles: {true_matrix.shape[0]*true_matrix.shape[1]}")
    # N1 = np.sum(true_matrix == 1) * N
    # N0 = np.sum(true_matrix == 0) * N
    # print(f"with n= {N}, N0 {N0} and N1 {N1}")
    cum_obs = sampler(true_matrix, altitude, N)
    # stats = calculate_statistics(true_matrix, cum_obs)
    # print(f"stats: {stats}")
    n = int(cum_obs.shape[0] / true_matrix.shape[0])
    true_matrix_ = np.tile(true_matrix, (n, 1))
    # print(true_matrix)
    # print(cum_obs)

    c = confusion_matrix(true_matrix_.flatten(), cum_obs.flatten())
    c_norm = c / c.astype(np.float64).sum(axis=1, keepdims=True)
    c_norm = np.round(c_norm, decimals=2).transpose()
    c_norm = np.nan_to_num(c_norm) + 10e-10
    # print(c_norm)

    # matrix_dict = {
    #     # "S": round(sigma(altitude), 2),
    #     "FP": round(stats["False Positive"] / N0, 2),
    #     "FN": round(stats["False Negative"] / N1, 2),
    #     "TN": round(stats["True Negative"] / N0, 2),
    #     "TP": round(stats["True Positive"] / N1, 2),
    #     # "1-S": round(1 - sigma(altitude), 2),
    # }
    # # print(matrix_dict)

    # TP = matrix_dict["TP"]
    # FP = matrix_dict["FP"]
    # TN = matrix_dict["TN"]
    # FN = matrix_dict["FN"]

    # s0 = matrix_dict["FP"]
    # s1 = matrix_dict["FN"]
    s0 = c_norm[1][0]
    s1 = c_norm[0][1]
    # return np.array([[TN, FN], [FP, TP]]), (s0, s1)
    return c_norm, (s0, s1)

# pairwise_factor_weights: equal, biased, adaptive
from matplotlib import pyplot as plt
import numpy as np


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

    if belief.ndim==3:
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
    if belief.ndim==3:
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
                f"Grid info: range: 0-{self.grid.x}-{self.grid.y}, cell_size:{self.grid.length}, map shape: {self.grid.shape} \n"
            )
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

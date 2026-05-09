# pairwise_factor_weights: equal, biased, adaptive
import numpy as np
import os
from datetime import datetime


def H(var):
    """ "Compute binary entropy of a random variable (or belief map)."""

    assert not np.any(np.isnan(var)), f"NaN detected in var: {var}"
    var = np.clip(var, 0.0, 1.0)  # Clamps values to the range [0, 1]

    v1 = var
    v2 = 1.0 - var

    if isinstance(var, np.ndarray):
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

    entropy = -(v1 * l1 + v2 * l2)

    assert np.all(np.greater_equal(entropy, 0.0))

    return entropy


def expected_posterior(var, sigma0, sigma1):
    """
    Compute the expected posterior distribution of a binary random variable.
    """

    # probability of the evidence
    # p(z = 0) = p(z = 0|m = 0)p(m = 0) + p(z = 0|m = 1)p(m = 1)
    sigma0 = np.clip(sigma0, 0.0, 1.0)
    sigma1 = np.clip(sigma1, 0.0, 1.0)
    a = (1.0 - sigma0) * (1.0 - var) + (sigma1 * var)  # p(z=0)
    # p(z = 1) = 1 - p(z = 0)
    b = 1.0 - a + 1e-6  # p(z=1) with stability epsilon

    assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
    assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

    # posterior distribution probabilities
    # p(m = 1|z = 0) = (p(z = 0|m = 1)p(m = 1))/p(z = 0)
    p10 = (sigma1 * var) / a  # p(m=1|z=0)
    # p(m = 1|z = 1) = (p(z = 1|m = 1)p(m = 1))/p(z = 1)
    p11 = ((1.0 - sigma1) * var) / b  # p(m=1|z=1)

    assert np.all(np.greater_equal(np.round(p10, decimals=2), 0.0)) and np.all(
        np.less_equal(np.round(p10, decimals=2), 1.0)
    ), f"{p10}"
    assert np.all(np.greater_equal(p11, 0.0)) and np.all(
        np.less_equal(p11, 1.0)
    ), f"{sigma1}-{var[np.greater(p11, 1.0)]}-{b[np.greater(p11, 1.0)]}"
    return a, b, p10, p11


def cH(var, sigma0, sigma1):
    """
    Compute the conditional entropy for a binary random variable using sensor model likelihoods.
    """
    a, b, p10, p11 = expected_posterior(var, sigma0, sigma1)

    # conditional entropy: average of the entropy of the posterior distribution probabilities
    # H(m|z) = p(z = 0)H(p(m = 1|z = 0)) + p(z = 1)H(p(m = 1|z = 1))
    cH = a * H(p10) + b * H(p11)

    assert np.all(np.greater_equal(cH, 0.0))

    return cH


def collect_sample_set(grid):
    # Create an array of central cells for each 3x3 block (using slices)
    rows, cols = grid.shape
    win_size = 3

    if rows % win_size != 0 or cols % win_size != 0:
        pad_rows, pad_cols = 0, 0
        if rows % win_size != 0:
            current_shape = rows
            while current_shape % win_size != 0:
                current_shape += 1
            pad_rows = current_shape - rows

        if cols % win_size != 0:
            current_shape = cols
            while current_shape % win_size != 0:
                current_shape += 1
            pad_cols = current_shape - cols

        grid = np.pad(grid, ((0, pad_rows), (0, pad_cols)), mode="edge")
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

    def __repr__(self):
        return f"({self.position[0]:.2f}, {self.position[1]:.2f},{self.altitude:.2f})"


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
    # Use probabilistic MSE: compare ground truth (0/1) to P(m=1) directly
    if belief.ndim == 3:
        estimated_probs = belief[..., 1]
    else:
        estimated_probs = belief
    mse = compute_mse(ground_truth_map, estimated_probs)
    entropy = compute_entropy(belief)
    coverage = compute_coverage(ms_set, grid)

    return (entropy, mse, coverage)


import os
import datetime as _dt


class FastLogger:
    """
    Minimal, pretty text logger compatible with previous usage.

    New features:
      - Optionally prints extra header sections (e.g., MCTS params) **before** the step table
      - log_data(...) now accepts step, action, ig (info gain) and prints extra columns
      - log_multi_agent_data(...) for multi-agent logging with per-agent lists
      - Backwards compatible: old calls still work
    """

    def __init__(
        self,
        log_folder,
        strategy="",
        pairwise="",
        grid=None,
        init_x=None,
        r=None,
        n_agent=None,
        e=None,
        conf_dict=None,
        filename="run.log",
        header_extras=None,
        multi_agent=False,
        num_agents=1,
        iteration=None,
        news_mode=None,
        use_hierarchical_timing=False,
        run_id=None,
    ):
        os.makedirs(log_folder, exist_ok=True)
        self.path = os.path.join(log_folder, filename)
        self._f = open(self.path, "a", buffering=1)
        self.multi_agent = multi_agent
        self.num_agents = num_agents
        self._use_hierarchical_timing = use_hierarchical_timing

        # Header
        self._w(f"[{_dt.datetime.now().isoformat(timespec='seconds')}]\n")
        if run_id is not None:
            self._w(f"Run ID: {run_id}\n")
        self._w(f"Strategy: {strategy}\n")
        self._w(f"Pairwise: {pairwise}\n")

        # Information sharing mode (IG, IG_d, BS, BM)
        if news_mode is not None:
            self._w(f"News mode: {news_mode}\n")

        # Show actual agent count and iteration separately
        if num_agents is not None and num_agents > 0:
            self._w(f"Num agents: {num_agents}\n")
        if iteration is not None:
            self._w(f"Iteration: {iteration}\n")
        # Legacy support: if n_agent is provided but num_agents/iteration are not
        elif n_agent is not None and iteration is None:
            self._w(f"Iteration: {n_agent}\n")

        self._w(f"Error margin: {e}\n")
        self._w(f"Gaussian radius {r} \n")
        if grid is not None:
            shape = getattr(grid, "shape", None)
            center = getattr(grid, "center", None)
            x = getattr(grid, "x", None)
            y = getattr(grid, "y", None)
            length = getattr(grid, "length", None)
            self._w(
                f"Grid info: range: 0-{x}-{y}, cell_size:{length}, map shape: {shape}, center:{center}\n"
            )

        # UAV init positions - support list for multi-agent
        if init_x is not None:
            if isinstance(init_x, list):
                # Multi-agent: list of positions
                positions = []
                for uav in init_x:
                    try:
                        pos = getattr(uav, "position", getattr(uav, "pos", uav))
                        alt = getattr(uav, "altitude", getattr(uav, "alt", None))
                        positions.append(f"({pos[0]:.1f}, {pos[1]:.1f}, {alt:.1f})")
                    except Exception:
                        positions.append(str(uav))
                self._w(f"init UAV positions: [{', '.join(positions)}]\n")
            else:
                # Single agent
                try:
                    pos = getattr(init_x, "position", getattr(init_x, "pos", init_x))
                    alt = getattr(init_x, "altitude", getattr(init_x, "alt", None))
                    self._w(f"init UAV position: {tuple(pos)} - {alt} \n")
                except Exception:
                    self._w(f"init UAV position: {init_x}\n")
                    self._w(f"init UAV position: {init_x}\n")

        # Header extras (e.g. MCTS params)
        if header_extras:
            for title, text in header_extras:
                self._w(f"{title}: {text}\n")

        # Table header - different format for multi-agent
        if multi_agent:
            # All planners: no timing columns (see timestamps.csv files)
            self._w(
                f"{'Step':<6}{'Entropy':<12}{'MSE':<10}{'Coverage':<10}{'Heights':<30}{'Actions':<35}{'IGs':<45}\n"
            )
            self._w("-" * 180 + "\n")
        else:
            self._w("Step   Entropy      MSE        Height   Coverage   Action    IG\n")
            self._w(
                "----------------------------------------------------------------\n"
            )

    def _w(self, s: str):
        self._f.write(s)
        self._f.flush()  # Force immediate write to disk

    def log(self, s: str):
        self._w(str(s) + "\n")

    def log_data(
        self,
        entropy,
        mse,
        height,
        coverage,
        step=None,
        action=None,
        ig=None,
    ):
        """Pretty one-line row; old signature still works."""
        try:
            step_s = f"{step:<5d}" if step is not None else "-    "
        except Exception:
            step_s = f"{str(step):<5}"
        ent_s = f"{float(entropy):<11.2f}"
        mse_s = f"{float(mse):<10.3f}"
        h_s = f"{float(height):<8.1f}"
        cov_s = f"{float(coverage):<9.4f}"
        act_s = f"{str(action):<8}" if action is not None else "-       "
        ig_s = f"{float(ig):<.4f}" if ig is not None else "-"
        self._w(f"{step_s} {ent_s} {mse_s} {h_s} {cov_s} {act_s} {ig_s}\n")

    def log_multi_agent_data(
        self,
        entropy,
        mse,
        coverage,
        heights,
        actions,
        igs,
        step=None,
        planning_times=None,
        hlp_times=None,
        llp_times=None,
        hlp_replans=None,
    ):
        """
        Log multi-agent data with common metrics and per-agent lists.

        Args:
            entropy: Common fused entropy value
            mse: Common fused MSE value
            coverage: Common combined coverage value
            heights: List of heights per agent [h0, h1, ...]
            actions: List of actions per agent [a0, a1, ...]
            igs: List of info gains per agent [ig0, ig1, ...]
            step: Step number
            planning_times: List of planning times in ms per agent [t0, t1, ...]
            hlp_times: List of HLP times in ms per agent (MH-Dec-MCTS only)
            llp_times: List of LLP times in ms per agent (MH-Dec-MCTS only)
            hlp_replans: List of HLP replan flags per agent (1=replanned, 0=cached)
        """
        try:
            step_s = f"{step:<5d}" if step is not None else "-    "
        except Exception:
            step_s = f"{str(step):<5}"
        ent_s = f"{float(entropy):<12.2f}"
        mse_s = f"{float(mse):<8.3f}"
        cov_s = f"{float(coverage):<8.4f}"

        # Format per-agent lists with consistent width
        h_list = "[" + ", ".join(f"{h:.1f}" for h in heights) + "]"
        act_list = "[" + ", ".join(str(a) if a else "-" for a in actions) + "]"
        ig_list = "[" + ", ".join(f"{ig:.4f}" if ig else "-" for ig in igs) + "]"

        # Write log entry (no timing columns - see timestamps.csv files)
        self._w(
            f"{step_s:<6}{ent_s:<12}{mse_s:<10}{cov_s:<10}{h_list:<30}{act_list:<35}{ig_list:<45}\n"
        )

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


import os
import pickle


def gaussian_random_field(cluster_radius, n_cell, seed=None):
    """
    Generate a 2D Gaussian random field and cache the results for reuse.
     https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
    Parameters:
    - cluster_radius: Correlation radius for the Gaussian field.
    - n_cell: Size of the field (n_cell_x x n_cell_y).
    - seed: Random seed for reproducibility (default: None).

    - cache_dir: Directory to store cached fields (default: "cache").

    Returns:
    - 2D binary random field as a numpy array.
    """

    # Ensure cache directory exists
    n_cell_x, n_cell_y = n_cell

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
    map_rng = np.random.default_rng(seed)
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

    return binary_field


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


def create_run_folder(base_path):
    # Get today's date in YYYYMMDD format
    date_str = datetime.now().strftime("%Y%m%d")

    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)

    # Get all folders in base_path starting with today's date
    existing = [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and f.startswith(date_str + "_")
    ]

    # Extract run numbers
    run_numbers = []
    for name in existing:
        try:
            run_number = int(name.split("_")[1])
            run_numbers.append(run_number)
        except (IndexError, ValueError):
            continue

    next_run = max(run_numbers) + 1 if run_numbers else 1

    run_folder_name = f"{date_str}_{next_run}"
    run_folder_path = os.path.join(base_path, run_folder_name)
    os.makedirs(run_folder_path)

    return run_folder_path


# --- helper: compact tag for parameter-based result folders ---
def make_param_tag(mcts_params):
    import hashlib, json as _json

    if not mcts_params:
        return "default"
    keys = [
        "planning_depth",
        "num_iterations",
        "ucb1_c",
        "discount_factor",
        "timeout",
        "parallel",
    ]
    parts = []
    for k in keys:
        if k in mcts_params:
            v = mcts_params[k]
            if isinstance(v, float):
                v = round(v, 3)
            # short key like pd, ni, uc, df, to, pa
            short = "".join([w[0] for w in k.split("_")[:2]])
            parts.append(f"{short}{v}")
    if parts:
        return "mcts_" + "_".join(parts)
    h = hashlib.md5(_json.dumps(mcts_params, sort_keys=True).encode()).hexdigest()[:8]
    return f"mcts_{h}"

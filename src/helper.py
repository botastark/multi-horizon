import time
# pairwise_factor_weights: equal, biased, adaptive
import numpy as np
import os


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





    return -(v1 * np.log2(v1) + v2 * np.log2(v2))


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
    b = 1.0 - a  # p(z=1) with stability epsilon


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


    return cH


def get_sensor_params(altitude: float, conf_dict=None):
    """
    Return (s0, s1) sensor noise parameters for a given altitude.

    Lookup order:
    1. Exact key in conf_dict (rounded to 2 decimals)
    2. Nearest key in conf_dict
    3. Analytic formula: sigma = a * (1 - exp(-b * altitude))

    This is the single authoritative implementation used by all planners.
    """
    if conf_dict is not None and conf_dict != {}:
        key = np.round(altitude, decimals=2)
        if key in conf_dict:
            return conf_dict[key]
        try:
            keys = np.array(list(conf_dict.keys()), dtype=float)
            idx = np.argmin(np.abs(keys - altitude))
            return conf_dict[keys[idx]]
        except Exception:
            pass
    a, b = 1.0, 0.015
    sigma = a * (1.0 - np.exp(-b * altitude))
    return sigma, sigma


def footprint_dict_to_bounds(footprint):
    """Convert a camera footprint dict to (imin, imax, jmin, jmax) bounds."""
    return (
        int(footprint["ul"][0]),
        int(footprint["bl"][0]),
        int(footprint["ul"][1]),
        int(footprint["ur"][1]),
    )


def footprint_iou(footprint1, footprint2):
    """Compute IoU for footprints represented as (imin, imax, jmin, jmax)."""
    imin1, imax1, jmin1, jmax1 = footprint1
    imin2, imax2, jmin2, jmax2 = footprint2

    inter_imin = max(imin1, imin2)
    inter_imax = min(imax1, imax2)
    inter_jmin = max(jmin1, jmin2)
    inter_jmax = min(jmax1, jmax2)

    if inter_imax <= inter_imin or inter_jmax <= inter_jmin:
        return 0.0

    intersection_area = (inter_imax - inter_imin) * (inter_jmax - inter_jmin)
    area1 = (imax1 - imin1) * (jmax1 - jmin1)
    area2 = (imax2 - imin2) * (jmax2 - jmin2)
    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def select_argmax_action(rng, action_scores):
    """Select the max-score action, using rng for exact-score ties."""
    sorted_actions = sorted(
        action_scores.items(),
        key=lambda item: item[1][0] if isinstance(item[1], (list, tuple)) else item[1],
        reverse=True,
    )
    best_action, best_score = sorted_actions.pop(0)
    best_value = best_score[0] if isinstance(best_score, (list, tuple)) else best_score
    best_actions = [best_action]
    for action, score in sorted_actions:
        value = score[0] if isinstance(score, (list, tuple)) else score
        if value == best_value:
            best_actions.append(action)
    return rng.choice(best_actions)


def adaptive_weights_matrix(obs_map):
    win_size = 3
    observation = obs_map

    if observation.shape[0] % win_size != 0 or observation.shape[1] % win_size != 0:
        pad_rows, pad_cols = 0, 0

        if observation.shape[0] % win_size != 0:
            current_shape = observation.shape[0]
            while current_shape % win_size != 0:
                current_shape += 1
            pad_rows = current_shape - observation.shape[0]

        if observation.shape[1] % win_size != 0:
            current_shape = observation.shape[1]
            while current_shape % win_size != 0:
                current_shape += 1
            pad_cols = current_shape - observation.shape[1]

        observation = np.pad(observation, ((0, pad_rows), (0, pad_cols)), mode="edge")

    _nblocks_r = observation.shape[0] // win_size
    _nblocks_c = observation.shape[1] // win_size
    v = (
        observation.reshape(_nblocks_r, win_size, _nblocks_c, win_size)
        .swapaxes(1, 2)
        .reshape(_nblocks_r * _nblocks_c, win_size, win_size)
    )

    m_center = v[:, 1, 1]
    m_neighbors = np.zeros((v.shape[0], 4), dtype=int)
    m_neighbors[:, 0] = v[:, 0, 1]
    m_neighbors[:, 1] = v[:, 1, 2]
    m_neighbors[:, 2] = v[:, 2, 1]
    m_neighbors[:, 3] = v[:, 1, 0]

    counts_one = np.count_nonzero(m_neighbors, axis=1)
    stacked = np.hstack((m_center.reshape(-1, 1), counts_one.reshape(-1, 1)))

    center_values = stacked[:, 0].astype(float)
    neighbor_counts = stacked[:, 1].astype(float)
    centered_values = center_values - np.mean(center_values)
    centered_counts = neighbor_counts - np.mean(neighbor_counts)

    stability_eps = 1e-12
    numerator = np.sum(centered_values * centered_counts)
    denominator = np.sqrt(np.sum(centered_values**2) * np.sum(centered_counts**2))
    pearson = numerator / max(denominator, stability_eps)
    pearson = np.clip(pearson, -1.0, 1.0)

    sigmoid = 1 / (1 + np.exp(-pearson))

    return np.array(
        [[sigmoid, 1 - sigmoid], [1 - sigmoid, sigmoid]],
        dtype=float,
    )


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


import datetime as _dt


class FastLogger:
    """
    Minimal, pretty text logger compatible with previous usage.

    New features:
      - Optionally prints extra header sections (e.g., MCTS params) **before** the step table
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
        run_id=None,
    ):
        os.makedirs(log_folder, exist_ok=True)
        self.path = os.path.join(log_folder, filename)
        self._f = open(self.path, "a", buffering=1)
        self.multi_agent = multi_agent
        self.num_agents = num_agents

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


class StalenessMixin:
    """Age helpers for intent dataclasses that have a ``timestamp`` field."""

    STALE_MAX_AGE = 2.0
    STALE_DECAY = 0.9
    STALE_THRESHOLD_SEC = 2.0

    def is_stale(self, max_age=None):
        """Return True if the intent is older than max_age seconds."""
        limit = self.STALE_MAX_AGE if max_age is None else max_age
        return time.time() - self.timestamp > limit

    def staleness_discount(self, decay_factor=None, threshold_sec=None):
        """D-UCT discount for the intent's age: 1.0 when fresh, below 1.0 when stale."""
        decay = self.STALE_DECAY if decay_factor is None else decay_factor
        threshold = self.STALE_THRESHOLD_SEC if threshold_sec is None else threshold_sec
        age = time.time() - self.timestamp
        return decay ** max(0, age / threshold)


ACTION_TIE_PRIORITY = {
    "left": 0,
    "right": 1,
    "front": 2,
    "back": 3,
    "up": 4,
    "down": 5,
    "hover": 6,
}


def select_canonical_action(action_scores, atol=1e-9):
    """Select the max-score action, breaking numeric ties by fixed priority without using an RNG."""
    best_value = max(score[0] for score in action_scores.values())
    best_actions = [
        action
        for action, score in action_scores.items()
        if np.isclose(score[0], best_value, rtol=0.0, atol=atol)
    ]
    return min(best_actions, key=lambda action: ACTION_TIE_PRIORITY.get(action, 99))

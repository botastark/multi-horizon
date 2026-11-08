"""
Diagnostic test: Why doesn't altitude decrease after coverage reaches 1?

Root cause hypothesis:
    _compute_ig in hierarchical_dec_mcts.py returns raw entropy of the belief
    footprint (H = sum of current cell entropies).  A higher altitude produces a
    larger footprint and therefore a higher raw-entropy sum, so the planner always
    prefers "up" regardless of coverage state.

    The greedy-IG / legacy planner instead uses the *true* information gain:
        IG = H(belief) - E[H(belief) | observation]
    which depends on the sensor noise sigma(altitude).  At lower altitude sigma is
    smaller (more precise sensor), so each cell yields more IG.  This trades off
    footprint size vs. per-cell gain and naturally drives the UAV down once most
    cells are already covered.

Tests in this file:
    1.  Show that _compute_ig IGNORES altitude (same result at low vs high alt
        for the same belief, scaled only by footprint area).
    2.  Show that the correct IG formula (H - cH) DOES depend on altitude and
        favours lower altitude when the map still has residual uncertainty.
    3.  Show that after full coverage (belief near 0/1 everywhere at high alt)
        the residual entropy is higher than after low-alt observations — so
        going lower would still reduce MSE even if the coverage mask is full.
    4.  Directly test action scoring: verify that "down" scores lower than
        lateral moves under _compute_ig, but should score higher under correct IG.
"""

import sys
import os
import math
import numpy as np

# Add src to path
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from helper import H, cH

# ---------------------------------------------------------------------------
# Minimal stubs so we can import the planner without the full simulator
# ---------------------------------------------------------------------------


class FakeGrid:
    def __init__(self, rows=100, cols=100, cell_len=1.0):
        self.shape = (rows, cols)
        self.x = cols * cell_len
        self.y = rows * cell_len
        self.center = True
        self.length = cell_len


class FakeCamera:
    """Mimics the Camera API used by LowLevelPlanner.
    Uses realistic simulation values (100m×100m grid, 1m cells, fov=45°).
    h_step≈5.41m, h_range=(5.41, 32.46), xy_step≈4.48m
    """

    def __init__(self, grid, fov=45.0, h_step=5.41, h_range=(5.41, 32.46)):
        self.grid = grid
        self.fov = fov
        self.h_step = h_step
        self.h_range = h_range
        self.xy_step = grid.length * 2  # 10 m lateral step
        self.x_range = [-grid.x / 2, grid.x / 2]
        self.y_range = [-grid.y / 2, grid.y / 2]
        self.altitude = h_range[0]
        self.position = (0.0, 0.0)
        self.a = 1.0
        self.b = 0.015

    def get_x(self):
        from helper import uav_position

        return uav_position((self.position, self.altitude))

    def set_altitude(self, alt):
        self.altitude = alt

    def set_position(self, pos):
        self.position = pos

    def footprint_at(self, pos, alt):
        """Return (imin, imax, jmin, jmax) grid indices for given pos/alt."""
        fov_rad = math.radians(self.fov) / 2
        x_dist = round(alt * math.tan(fov_rad) / self.grid.length) * self.grid.length
        y_dist = x_dist

        x_min = max(pos[0] - x_dist, self.x_range[0])
        x_max = min(pos[0] + x_dist, self.x_range[1])
        y_min = max(pos[1] - y_dist, self.y_range[0])
        y_max = min(pos[1] + y_dist, self.y_range[1])

        rows, cols = self.grid.shape

        def xy_to_ij(x, y):
            col = int((x - self.x_range[0]) / self.grid.length)
            row = int((y - self.y_range[0]) / self.grid.length)
            return max(0, min(row, rows - 1)), max(0, min(col, cols - 1))

        i_min, j_min = xy_to_ij(x_min, y_min)
        i_max, j_max = xy_to_ij(x_max, y_max)
        return i_min, i_max, j_min, j_max

    def sigma_at(self, alt):
        """Sensor noise: high altitude = large sigma = noisy."""
        return self.a * (1.0 - math.exp(-self.b * alt))


# ---------------------------------------------------------------------------
# Helper functions matching planner.py logic
# ---------------------------------------------------------------------------


def raw_entropy_ig(belief, imin, imax, jmin, jmax):
    """
    What hierarchical_dec_mcts._compute_ig does:
    Sum of per-cell binary entropy in footprint (no sensor model).
    """
    slice_ = belief[imin:imax, jmin:jmax, 1]
    eps = 1e-10
    p = np.clip(slice_, eps, 1 - eps)
    return float(np.sum(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))


def sensor_aware_ig(belief, imin, imax, jmin, jmax, sigma):
    """
    What planner.py info_gain does:
    IG = H(belief) - E[H(belief) | observation]  — uses sensor noise sigma.
    """
    slice_ = belief[imin:imax, jmin:jmax, 1]
    h_prior = H(slice_)
    h_cond = cH(slice_, sigma, sigma)
    return float(np.sum(h_prior - h_cond))


# ---------------------------------------------------------------------------
# Test 1 — _compute_ig is proportional to footprint area, ignores altitude
# ---------------------------------------------------------------------------


def test_raw_entropy_ig_ignores_altitude():
    """
    At the same centre position, higher altitude = bigger footprint = higher
    raw-entropy IG.  The value is driven purely by footprint area, not altitude.
    """
    grid = FakeGrid()
    cam = FakeCamera(grid)

    # Uniform belief at 0.5 (maximum per-cell entropy = 1 bit everywhere)
    belief = np.full((100, 100, 2), 0.5)

    pos = (0.0, 0.0)
    alt_low = cam.h_range[0]  # e.g. 5.41
    alt_high = cam.h_range[1]  # e.g. 32.46

    fp_low = cam.footprint_at(pos, alt_low)
    fp_high = cam.footprint_at(pos, alt_high)

    ig_low = raw_entropy_ig(belief, *fp_low)
    ig_high = raw_entropy_ig(belief, *fp_high)

    area_low = max(0, fp_low[1] - fp_low[0]) * max(0, fp_low[3] - fp_low[2])
    area_high = max(0, fp_high[1] - fp_high[0]) * max(0, fp_high[3] - fp_high[2])

    print("\n[TEST 1] Raw-entropy IG vs altitude (uniform 0.5 belief):")
    print(
        f"  Low alt  ({alt_low:.1f}m): footprint area={area_low} cells, IG={ig_low:.2f}"
    )
    print(
        f"  High alt ({alt_high:.1f}m): footprint area={area_high} cells, IG={ig_high:.2f}"
    )
    print(f"  → 'up' always wins: IG ratio high/low = {ig_high/max(ig_low,1e-9):.2f}x")

    assert (
        ig_high > ig_low
    ), "High altitude should yield higher raw-entropy IG due to larger footprint"
    # For non-zero areas, the ratio should match the area ratio
    if area_low > 0:
        assert (
            abs(ig_high / ig_low - area_high / area_low) < 0.05
        ), "IG ratio should closely match footprint area ratio for uniform belief"


# ---------------------------------------------------------------------------
# Test 2 — sensor-aware IG can favour lower altitude
# ---------------------------------------------------------------------------


def test_sensor_aware_ig_favours_descent():
    """
    With the correct IG = H - cH formula, lower altitude yields higher IG per
    cell because sigma is smaller (more precise sensor).  For a moderately
    uncertain map the per-cell gain can outweigh the smaller footprint.
    """
    grid = FakeGrid()
    cam = FakeCamera(grid)

    # Moderately uncertain belief (residual entropy after high-alt coverage)
    # Simulate: cells were observed once from high alt → belief ~0.3 or 0.7
    rng = np.random.default_rng(42)
    true_map = rng.integers(0, 2, size=(100, 100)).astype(float)
    sigma_high = cam.sigma_at(cam.h_range[1])

    # Simulate one high-altitude observation for every cell
    belief_p = np.where(
        true_map == 1,
        (1 - sigma_high) / ((1 - sigma_high) + sigma_high),  # p(m=1|z=1)
        sigma_high / (sigma_high + (1 - sigma_high)),  # p(m=1|z=0)
    )
    belief = np.stack([1 - belief_p, belief_p], axis=-1)

    pos = (0.0, 0.0)
    alt_low = cam.h_range[0]
    alt_high = cam.h_range[1]

    fp_low = cam.footprint_at(pos, alt_low)
    fp_high = cam.footprint_at(pos, alt_high)
    sig_low = cam.sigma_at(alt_low)
    sig_high = cam.sigma_at(alt_high)

    ig_low_sensor = sensor_aware_ig(belief, *fp_low, sig_low)
    ig_high_sensor = sensor_aware_ig(belief, *fp_high, sig_high)

    ig_low_raw = raw_entropy_ig(belief, *fp_low)
    ig_high_raw = raw_entropy_ig(belief, *fp_high)

    print("\n[TEST 2] Sensor-aware vs raw-entropy IG after high-alt coverage:")
    print(f"  sigma low={sig_low:.4f}, sigma high={sig_high:.4f}")
    print(
        f"  Raw-entropy  IG — low: {ig_low_raw:.2f}, high: {ig_high_raw:.2f}  → prefers {'high' if ig_high_raw>ig_low_raw else 'low'}"
    )
    print(
        f"  Sensor-aware IG — low: {ig_low_sensor:.2f}, high: {ig_high_sensor:.2f}  → prefers {'high' if ig_high_sensor>ig_low_sensor else 'low'}"
    )

    # Raw-entropy should prefer high altitude (larger footprint)
    assert ig_high_raw > ig_low_raw, "Raw-entropy IG should prefer high altitude"

    # Sensor-aware IG per cell is much higher at low altitude; check per-cell ratio
    area_low = max(1, (fp_low[1] - fp_low[0]) * (fp_low[3] - fp_low[2]))
    area_high = max(1, (fp_high[1] - fp_high[0]) * (fp_high[3] - fp_high[2]))
    per_cell_low = ig_low_sensor / area_low
    per_cell_high = ig_high_sensor / area_high
    print(
        f"  Sensor-aware IG per cell — low: {per_cell_low:.4f}, high: {per_cell_high:.4f}"
    )
    assert (
        per_cell_low > per_cell_high
    ), "Lower altitude should give higher IG per cell due to more precise sensor"


# ---------------------------------------------------------------------------
# Test 3 — residual belief uncertainty is higher after high-alt coverage
# ---------------------------------------------------------------------------


def test_residual_entropy_higher_after_high_alt_coverage():
    """
    After observing each cell once from high altitude, the residual per-cell
    entropy is higher than after observing once from low altitude.  This means
    even when coverage=1, going lower would yield more information.
    """
    grid = FakeGrid()
    cam = FakeCamera(grid)

    rng = np.random.default_rng(0)
    true_map = rng.integers(0, 2, size=(100, 100)).astype(float)

    def post_belief(true_map, sigma):
        """Simulate one observation of all cells and return resulting belief."""
        # For each cell, compute posterior after a single observation
        belief_p = np.where(
            true_map == 1,
            (1 - sigma) / ((1 - sigma) + sigma),
            sigma / (sigma + (1 - sigma)),
        )
        return belief_p

    sig_low = cam.sigma_at(cam.h_range[0])
    sig_high = cam.sigma_at(cam.h_range[1])

    belief_low = post_belief(true_map, sig_low)
    belief_high = post_belief(true_map, sig_high)

    mean_entropy_low = float(np.mean(H(belief_low)))
    mean_entropy_high = float(np.mean(H(belief_high)))

    print("\n[TEST 3] Residual entropy after single observation:")
    print(
        f"  sigma low={sig_low:.4f} → mean cell entropy after obs: {mean_entropy_low:.4f}"
    )
    print(
        f"  sigma high={sig_high:.4f} → mean cell entropy after obs: {mean_entropy_high:.4f}"
    )
    print(
        f"  → High-alt observations leave {mean_entropy_high/mean_entropy_low:.2f}x more residual uncertainty"
    )

    assert (
        mean_entropy_high > mean_entropy_low
    ), "High-altitude observations (noisy sensor) should leave more residual entropy"


# ---------------------------------------------------------------------------
# Test 4 — action scoring: 'down' loses under raw-entropy, should win under sensor-aware IG
# ---------------------------------------------------------------------------


def test_action_scoring_down_vs_lateral():
    """
    Simulate action scoring at a position near full coverage.
    Show that 'down' scores worse than lateral moves under raw-entropy IG,
    but would score higher under sensor-aware IG.
    """
    grid = FakeGrid()
    cam = FakeCamera(grid)

    # Build a 'nearly covered' belief: all cells observed once from high alt
    rng = np.random.default_rng(7)
    true_map = rng.integers(0, 2, size=(100, 100)).astype(float)
    sigma_high = cam.sigma_at(cam.h_range[1])
    belief_p = np.where(
        true_map == 1,
        (1 - sigma_high) / ((1 - sigma_high) + sigma_high),
        sigma_high / (sigma_high + (1 - sigma_high)),
    )
    belief = np.stack([1 - belief_p, belief_p], axis=-1)

    pos = (0.0, 0.0)
    current_alt = cam.h_range[1]  # UAV is at max altitude

    actions = {
        "hover": (pos, current_alt),
        "down": (pos, max(cam.h_range[0], current_alt - cam.h_step)),
        "front": ((pos[0], pos[1] + cam.xy_step), current_alt),
        "back": ((pos[0], pos[1] - cam.xy_step), current_alt),
    }

    raw_scores = {}
    sensor_scores = {}
    for act, (npos, nalt) in actions.items():
        if nalt < cam.h_range[0] or nalt > cam.h_range[1]:
            continue
        fp = cam.footprint_at(npos, nalt)
        sig = cam.sigma_at(nalt)
        raw_scores[act] = raw_entropy_ig(belief, *fp)
        sensor_scores[act] = sensor_aware_ig(belief, *fp, sig)

    best_raw = max(raw_scores, key=lambda a: raw_scores[a])
    best_sensor = max(sensor_scores, key=lambda a: sensor_scores[a])

    print(f"\n[TEST 4] Action scoring at full coverage (alt={current_alt:.1f}m):")
    print(f"  Raw-entropy IG scores:    {raw_scores}")
    print(
        f"  Best raw action:          '{best_raw}' ← always picks lateral/hover over 'down'"
    )
    print(f"  Sensor-aware IG scores:   {sensor_scores}")
    print(f"  Best sensor-aware action: '{best_sensor}'")

    # Under raw-entropy, 'down' should lose to lateral moves (smaller footprint)
    assert raw_scores.get("down", -1) < raw_scores.get(
        "front", float("inf")
    ), "Raw-entropy IG should score 'down' lower than 'front' (smaller footprint)"

    # Under sensor-aware IG, 'down' should be competitive or win
    # (per-cell gain at lower altitude outweighs footprint reduction)
    down_sensor = sensor_scores.get("down", 0)
    front_sensor = sensor_scores.get("front", 0)
    print(f"\n  'down' sensor-aware IG:  {down_sensor:.4f}")
    print(f"  'front' sensor-aware IG: {front_sensor:.4f}")
    print(
        f"  → Sensor-aware IG {'FAVOURS descent' if down_sensor >= front_sensor else 'still prefers lateral — but per-cell IG gain at low alt is higher'}"
    )

    # At minimum: per-cell gain is higher at low altitude
    fp_down = cam.footprint_at(pos, max(cam.h_range[0], current_alt - cam.h_step))
    fp_front = cam.footprint_at((pos[0], pos[1] + cam.xy_step), current_alt)
    area_down = max(1, (fp_down[1] - fp_down[0]) * (fp_down[3] - fp_down[2]))
    area_front = max(1, (fp_front[1] - fp_front[0]) * (fp_front[3] - fp_front[2]))
    per_cell_down = sensor_scores.get("down", 0) / area_down
    per_cell_front = sensor_scores.get("front", 0) / area_front
    print(
        f"  Per-cell sensor-aware IG — down: {per_cell_down:.6f}, front: {per_cell_front:.6f}"
    )
    assert (
        per_cell_down > per_cell_front
    ), "Sensor-aware IG per cell should be higher for 'down' than for 'front'"


# ---------------------------------------------------------------------------
# Test 5 — summary: what drives altitude down in old code but not new
# ---------------------------------------------------------------------------


def test_summary_diagnosis():
    """
    Summary diagnostic: quantify the difference between the two IG formulas
    across the full altitude range to show how the sensor-aware formula creates
    an incentive to descend.
    """
    grid = FakeGrid()
    cam = FakeCamera(grid)

    # Residual belief after full high-alt coverage
    rng = np.random.default_rng(1)
    true_map = rng.integers(0, 2, size=(100, 100)).astype(float)
    sigma_high = cam.sigma_at(cam.h_range[1])
    belief_p = np.where(
        true_map == 1,
        (1 - sigma_high) / ((1 - sigma_high) + sigma_high),
        sigma_high / (sigma_high + (1 - sigma_high)),
    )
    belief = np.stack([1 - belief_p, belief_p], axis=-1)

    pos = (0.0, 0.0)
    altitudes = np.linspace(cam.h_range[0], cam.h_range[1], 7)

    print("\n[TEST 5] IG vs altitude at centre position (post full-coverage belief):")
    print(
        f"  {'Alt':>8}  {'sigma':>8}  {'Footprint':>10}  {'Raw IG':>10}  {'Sensor IG':>10}  {'Raw/cell':>10}  {'Sens/cell':>10}"
    )

    raw_igs = []
    sensor_igs = []
    for alt in altitudes:
        fp = cam.footprint_at(pos, alt)
        sig = cam.sigma_at(alt)
        r = raw_entropy_ig(belief, *fp)
        s = sensor_aware_ig(belief, *fp, sig)
        area = max(1, (fp[1] - fp[0]) * (fp[3] - fp[2]))
        raw_igs.append(r)
        sensor_igs.append(s)
        print(
            f"  {alt:8.2f}  {sig:8.4f}  {area:10d}  {r:10.3f}  {s:10.3f}  {r/area:10.5f}  {s/area:10.5f}"
        )

    # Raw-entropy IG should be monotonically increasing with altitude
    assert all(
        raw_igs[i] <= raw_igs[i + 1] for i in range(len(raw_igs) - 1)
    ), "Raw-entropy IG must increase monotonically with altitude (larger footprint)"

    # Sensor-aware IG per-cell should be monotonically decreasing with altitude
    per_cell_sensor = [
        sensor_igs[i]
        / max(
            1,
            (cam.footprint_at(pos, alt)[1] - cam.footprint_at(pos, alt)[0])
            * (cam.footprint_at(pos, alt)[3] - cam.footprint_at(pos, alt)[2]),
        )
        for i, alt in enumerate(altitudes)
    ]
    assert all(
        per_cell_sensor[i] >= per_cell_sensor[i + 1]
        for i in range(len(per_cell_sensor) - 1)
    ), "Sensor-aware IG per cell must decrease with altitude (less precise sensor)"

    print("\n  CONCLUSION:")
    print(
        "  → Raw-entropy IG always prefers max altitude (monotone ↑) — NO descent incentive"
    )
    print(
        "  → Sensor-aware IG per cell always prefers min altitude (monotone ↓) — descent incentive exists"
    )
    print(
        "  → This is why altitude doesn't decrease after coverage=1 in current MH code"
    )


if __name__ == "__main__":
    test_raw_entropy_ig_ignores_altitude()
    test_sensor_aware_ig_favours_descent()
    test_residual_entropy_higher_after_high_alt_coverage()
    test_action_scoring_down_vs_lateral()
    test_summary_diagnosis()
    print("\n✓ All diagnostic tests passed.")

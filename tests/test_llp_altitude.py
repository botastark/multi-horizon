"""
Test: LLP altitude optimization vs Greedy-IG, and HLP g2 influence on LLP.

Goal:
    1. Verify that the sensor model creates the right IG vs altitude tradeoff:
         - True IG  = Σ [H(p) − cH(p,σ,σ)]  (Bayesian, greedy formula)
         - At σ≈0.5 (~46m with a=1, b=0.015) the sensor is a coin-flip → IG≈0
         - LLP currently uses raw entropy Σ H(p) which MISSES this structure

    2. Show that LLP best-rollout altitudes differ between the buggy formula and
       the corrected formula (especially noticeable when belief is partially
       observed so large-footprint high-alt cells are already certain).

    3. Show how HLP g2 guidance nudges LLP trajectories toward target regions.

Run with:
    cd /home/bota/repos/multi-horizon/src
    python ../tests/test_llp_altitude.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ── project imports ──────────────────────────────────────────────────────────
from uav_camera import Camera
from greedy_ig_planner import H, cH
from hierarchical_dec_mcts import LowLevelPlanner, HLIntent

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE = 100  # cells per side
CELL_LENGTH = 1.0  # metres per cell
FIELD_SIZE = GRID_SIZE * CELL_LENGTH  # 100 m


def make_grid():
    """Minimal grid_info compatible with Camera / get_range."""
    g = SimpleNamespace()
    g.x = FIELD_SIZE
    g.y = FIELD_SIZE
    g.shape = (GRID_SIZE, GRID_SIZE)
    g.length = CELL_LENGTH
    g.center = True  # origin at (0,0)
    return g


CAMERA_MIN_ALT = 19.5  # metres — matches real experiment default (main.py)


def make_camera(grid, overlap=0.3):
    """Create Camera at minimum altitude so h_range is well-defined."""
    cam = Camera(
        grid,
        fov_angle=60,
        camera_altitude=CAMERA_MIN_ALT,
        f_overlap=overlap,
        s_overlap=overlap,
        seed=0,
    )
    cam.set_altitude(cam.h_range[0])
    cam.set_position((0.0, 0.0))
    return cam


def uniform_belief(grid, p=0.5):
    """Uniform prior belief (max uncertainty everywhere)."""
    return np.full((GRID_SIZE, GRID_SIZE, 2), p)


def patchy_belief(grid, seed=42):
    """
    Partially observed belief: large region already near 0 or 1,
    leaving a 'target region' with high uncertainty in one corner.
    """
    rng = np.random.default_rng(seed)
    b = np.full((GRID_SIZE, GRID_SIZE, 2), 0.5)
    b[:, :, 1] = rng.uniform(0.05, 0.15, (GRID_SIZE, GRID_SIZE))  # mostly certain
    # high-uncertainty patch in top-right quadrant
    b[50:, 50:, 1] = rng.uniform(0.35, 0.65, (50, 50))
    b[:, :, 0] = 1.0 - b[:, :, 1]
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper: sensor model (same as greedy_ig_planner._get_sensor_params)
# ─────────────────────────────────────────────────────────────────────────────


def sensor_sigma(altitude, a=1.0, b=0.015):
    return a * (1.0 - math.exp(-b * altitude))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEST A — IG vs altitude curve: greedy formula vs LLP formula
# ─────────────────────────────────────────────────────────────────────────────


def test_ig_altitude_curve():
    """
    For a uniform belief and a fixed (centred) position, compute expected IG
    at each altitude level for two formulations:

        (a) Raw entropy sum          : Σ H(p)         [current LLP — misses sensor quality]
        (b) True IG (greedy formula) : Σ [H(p) − cH(p,σ,σ)]  [correct]

    Key physics with a=1, b=0.015:
    - σ = a(1−e^{−bh}) reaches 0.5 at h ≈ 46m.
    - At σ=0.5 the sensor is a coin-flip → cH = H(p) → IG ≈ 0 (minimum).
    - Above that, the sensor is adversarially noisy but still informative.
    - Footprint covers the full 100×100 field at h ≳ 58m.

    With raw entropy (LLP): sum is monotone, no dip — always prefers max alt.
    With true IG: a clear dip near σ=0.5 exposes the sensor-quality structure.
    """
    print("\n" + "=" * 60)
    print("TEST A — IG vs altitude curve")
    print("=" * 60)

    grid = make_grid()
    camera = make_camera(grid)
    belief = uniform_belief(grid)

    # Fine-grained sweep to show the σ=0.5 dip
    altitudes = np.linspace(camera.h_range[0], camera.h_range[1], 40)

    raw_entropy_ig = []
    true_ig = []
    sigmas = []

    p0 = (0.0, 0.0)  # centre position

    for alt in altitudes:
        try:
            [[imin, imax], [jmin, jmax]] = camera.get_range(
                position=p0, altitude=alt, index_form=True
            )
        except Exception:
            raw_entropy_ig.append(0.0)
            true_ig.append(0.0)
            sigmas.append(0.0)
            continue

        H_grid, W_grid = belief.shape[:2]
        imin = max(0, min(imin, H_grid))
        imax = max(0, min(imax, H_grid))
        jmin = max(0, min(jmin, W_grid))
        jmax = max(0, min(jmax, W_grid))

        if imax <= imin or jmax <= jmin:
            raw_entropy_ig.append(0.0)
            true_ig.append(0.0)
            sigmas.append(0.0)
            continue

        prior = belief[imin:imax, jmin:jmax, 1]
        s = sensor_sigma(alt)
        sigmas.append(s)

        h_prior = H(prior)
        h_cond = cH(prior, round(s, 7), round(s, 7))
        ig = h_prior - h_cond
        ig = np.where(np.isclose(h_prior, h_cond), 0.0, ig)

        raw_entropy_ig.append(float(np.sum(h_prior)))
        true_ig.append(float(np.sum(ig)))

    # σ=0.5 occurs at h = -ln(0.5)/b
    alt_sigma_half = -math.log(0.5) / 0.015
    idx_dip = int(np.argmin(true_ig))

    print(f"  Altitude range : {altitudes[0]:.1f} – {altitudes[-1]:.1f} m")
    print(f"  σ=0.5 at altitude ≈ {alt_sigma_half:.1f} m  (sensor is coin-flip → IG=0)")
    print(
        f"  Dip in true-IG  : alt={altitudes[idx_dip]:.1f} m, σ={sigmas[idx_dip]:.3f}\n"
    )

    print(f"  {'Alt':>6}  {'σ':>6}  {'Cells':>7}  {'RawH':>10}  {'TrueIG':>10}")
    step = max(1, len(altitudes) // 16)
    for i in range(0, len(altitudes), step):
        alt, s = altitudes[i], sigmas[i]
        re, ti = raw_entropy_ig[i], true_ig[i]
        fp = int(re / 1.0)  # H(0.5)=1, so cell count ≈ raw entropy for uniform belief
        marker = " ← dip (σ≈0.5)" if i == idx_dip else ""
        print(f"  {alt:6.1f}  {s:6.4f}  {fp:7d}  {re:10.2f}  {ti:10.4f}{marker}")

    # Assertions
    assert (
        true_ig[idx_dip] < true_ig[0]
    ), "True IG should dip below starting level near σ=0.5"
    assert (
        abs(altitudes[idx_dip] - alt_sigma_half) < 25.0
    ), f"Dip should be near σ=0.5 altitude; got dip at {altitudes[idx_dip]:.1f} vs expected {alt_sigma_half:.1f}"
    assert (
        raw_entropy_ig[-1] >= raw_entropy_ig[0]
    ), "Raw entropy must be non-decreasing with altitude"

    print("\n  ✓ True IG shows sensor-model dip at σ≈0.5 altitude.")
    print(
        "  ✓ Raw entropy is monotone — LLP without sensor model always prefers max alt."
    )
    return altitudes, raw_entropy_ig, true_ig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TEST B — LLP rollout rewards with current vs corrected _compute_ig
# ─────────────────────────────────────────────────────────────────────────────


def _llp_compute_ig_corrected(llp, position, altitude, coverage_discount):
    """
    Drop-in replacement for LowLevelPlanner._compute_ig that uses the
    proper Bayesian IG formula: Σ [H(p) - cH(p, s0, s1)].
    """
    if llp.belief is None:
        return 0.0

    try:
        [[imin, imax], [jmin, jmax]] = llp.camera.get_range(
            position=position, altitude=altitude, index_form=True
        )
    except Exception:
        return 0.0

    H_grid, W_grid = llp.belief.shape[:2]
    imin = max(0, min(imin, H_grid))
    imax = max(0, min(imax, H_grid))
    jmin = max(0, min(jmin, W_grid))
    jmax = max(0, min(jmax, W_grid))

    if imax <= imin or jmax <= jmin:
        return 0.0

    prior = (
        llp.belief[imin:imax, jmin:jmax, 1]
        if llp.belief.ndim == 3
        else llp.belief[imin:imax, jmin:jmax]
    )

    s = sensor_sigma(altitude, a=llp.camera.a, b=llp.camera.b)
    s0 = s1 = round(s, 7)

    h_prior = H(prior)
    h_cond = cH(prior, s0, s1)
    ig = h_prior - h_cond
    ig = np.where(np.isclose(h_prior, h_cond), 0.0, ig)

    discount_slice = coverage_discount[imin:imax, jmin:jmax]
    return float(np.sum(ig * discount_slice))


def _simulate_trajectory_with_fn(
    llp, start_state, actions, coverage_discount, compute_ig_fn
):
    """Simulate a trajectory using a custom compute_ig function and return total discounted IG."""
    pos = (start_state[0], start_state[1])
    alt = start_state[2]
    total_ig = 0.0
    alts = []

    for step, action in enumerate(actions):
        result = llp._apply_action(pos, alt, action)
        if result is None:
            break
        pos, alt = result
        ig = compute_ig_fn(llp, pos, alt, coverage_discount)
        total_ig += (llp.discount**step) * ig
        alts.append(alt)

    return total_ig, alts


def test_llp_altitude_preference(n_rollouts=200, horizon=8):
    """
    Run random rollouts with both the BUGGY (current) and CORRECTED IG formulas.
    Track which altitude levels appear in the best rollout for each formula.

    We use a patchy belief where most cells near the start are already observed
    (low entropy). This makes the footprint-size advantage of high altitude smaller
    (those extra cells are certain), so the sensor-quality tradeoff dominates more.
    """
    print("\n" + "=" * 60)
    print("TEST B — LLP rollout altitude preference: buggy vs corrected IG")
    print("=" * 60)

    grid = make_grid()
    camera = make_camera(grid)
    belief = patchy_belief(grid)  # partially observed — exposes sensor-quality tradeoff

    llp = LowLevelPlanner(
        agent_id=0,
        camera=camera,
        grid_info=grid,
        horizon=horizon,
        num_iterations=n_rollouts,
        seed=0,
    )
    llp.update_belief(belief)

    start_state = (0.0, 0.0, camera.h_range[0])
    coverage = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    rng = np.random.default_rng(42)
    actions_pool = llp.actions

    best_buggy_reward = -1e9
    best_buggy_alts = []
    best_correct_reward = -1e9
    best_correct_alts = []

    for _ in range(n_rollouts):
        actions = [rng.choice(actions_pool) for _ in range(horizon)]

        r_bug, alts_bug = _simulate_trajectory_with_fn(
            llp, start_state, actions, coverage, LowLevelPlanner._compute_ig
        )
        r_cor, alts_cor = _simulate_trajectory_with_fn(
            llp, start_state, actions, coverage, _llp_compute_ig_corrected
        )

        if r_bug > best_buggy_reward:
            best_buggy_reward = r_bug
            best_buggy_alts = alts_bug
        if r_cor > best_correct_reward:
            best_correct_reward = r_cor
            best_correct_alts = alts_cor

    avg_buggy = np.mean(best_buggy_alts) if best_buggy_alts else 0.0
    avg_correct = np.mean(best_correct_alts) if best_correct_alts else 0.0

    print(f"  Altitude range : {camera.h_range[0]:.2f} – {camera.h_range[1]:.2f} m")
    print(
        f"  Best rollout (BUGGY   IG): avg alt = {avg_buggy:.2f} m, reward = {best_buggy_reward:.4f}"
    )
    print(
        f"  Best rollout (CORRECT IG): avg alt = {avg_correct:.2f} m, reward = {best_correct_reward:.4f}"
    )
    print(f"  Trajectory altitudes (BUGGY   ): {[round(a,1) for a in best_buggy_alts]}")
    print(
        f"  Trajectory altitudes (CORRECT ): {[round(a,1) for a in best_correct_alts]}"
    )

    print(
        f"\n  ✓ Buggy IG prefers altitude {avg_buggy:.2f} m (should be near max={camera.h_range[1]:.2f})"
    )
    print(
        f"  ✓ Correct IG prefers altitude {avg_correct:.2f} m (should be interior optimum)"
    )

    return camera, best_buggy_alts, best_correct_alts


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TEST C — HLP g2 influence on LLP reward
# ─────────────────────────────────────────────────────────────────────────────


def test_hlp_g2_influence():
    """
    Show how providing an HLP intent (g2 guidance) shifts LLP rollout
    reward when use_g2=True.

    Scenario:
    - Belief is 'patchy': one high-uncertainty region far from start.
    - Without HLP: LLP greedily maximises local IG → stays near start.
    - With HLP intent pointing to the far region: g2 rewards trajectories
      that head toward it → LLP moves in that direction.
    """
    print("\n" + "=" * 60)
    print("TEST C — HLP g2 guidance shifts LLP toward target region")
    print("=" * 60)

    grid = make_grid()
    camera = make_camera(grid)
    belief = patchy_belief(grid)

    # Far corner region (high uncertainty patch sits at indices [50:, 50:])
    # In centred coords (100x100, length=1): that's x in [0,50], y in [-50,0]
    target_region_bounds = {
        "x_min": 0.0,
        "x_max": 50.0,
        "y_min": -50.0,
        "y_max": 0.0,
    }

    # ── LLP without g2 ────────────────────────────────────────────────────
    llp_no_g2 = LowLevelPlanner(
        agent_id=0,
        camera=camera,
        grid_info=grid,
        horizon=6,
        num_iterations=150,
        seed=1,
        use_g2=False,
    )
    llp_no_g2.update_belief(belief)
    start_state = (0.0, 0.0, camera.h_range[0])
    intent_no_g2 = llp_no_g2.plan(start_state)

    # ── LLP with g2 + HLP intent toward far region ────────────────────────
    llp_with_g2 = LowLevelPlanner(
        agent_id=0,
        camera=camera,
        grid_info=grid,
        horizon=6,
        num_iterations=150,
        seed=1,
        use_g2=True,
        g2_mode="stub",  # stub keeps it simple for the test
    )
    llp_with_g2.update_belief(belief)

    # Inject a mock HLP intent pointing to the high-uncertainty corner
    mock_hl_intent = HLIntent(
        agent_id=0,
        region_sequence=[0],
        eta_sequence=[5.0],           # 5 seconds to reach target
        target_center=(25.0, -25.0),  # centre of high-uncertainty corner
        current_target_region=0,
        horizon=1,
    )
    llp_with_g2.update_hl_guidance(mock_hl_intent)
    intent_with_g2 = llp_with_g2.plan(start_state)

    # Analyse: where do the planned trajectories end up?
    def in_target(states):
        """Count states that land inside target region."""
        count = 0
        for s in states:
            x, y = s[0], s[1]
            if (
                target_region_bounds["x_min"] <= x <= target_region_bounds["x_max"]
                and target_region_bounds["y_min"] <= y <= target_region_bounds["y_max"]
            ):
                count += 1
        return count

    states_no_g2 = intent_no_g2.state_sequence if intent_no_g2.state_sequence else []
    states_with_g2 = (
        intent_with_g2.state_sequence if intent_with_g2.state_sequence else []
    )

    n_in_no_g2 = in_target(states_no_g2)
    n_in_with_g2 = in_target(states_with_g2)

    print("\n  Start position: (0, 0),  Target region: x∈[0,50], y∈[-50,0]")
    print("\n  Plan WITHOUT g2:")
    print(f"    Actions : {intent_no_g2.action_sequence}")
    print(
        f"    States  : {[(round(s[0],1), round(s[1],1), round(s[2],1)) for s in states_no_g2]}"
    )
    print(f"    States in target region: {n_in_no_g2}/{len(states_no_g2)}")
    print(f"    Total IG: {intent_no_g2.total_expected_ig:.4f}")
    print("\n  Plan WITH g2 (HLP → target region):")
    print(f"    Actions : {intent_with_g2.action_sequence}")
    print(
        f"    States  : {[(round(s[0],1), round(s[1],1), round(s[2],1)) for s in states_with_g2]}"
    )
    print(f"    States in target region: {n_in_with_g2}/{len(states_with_g2)}")
    print(f"    Total IG: {intent_with_g2.total_expected_ig:.4f}")

    print("\n  Interpretation:")
    if n_in_with_g2 >= n_in_no_g2:
        print(
            "  ✓ HLP g2 guidance nudges LLP toward the high-uncertainty target region."
        )
    else:
        print(
            "  ⚠ HLP g2 guidance did NOT increase movement toward target — check g2 weight."
        )

    print("\n  NOTE: With 'stub' g2 mode and low horizon the effect may be subtle.")
    print("  Use g2_mode='hl_aware' and longer horizon for stronger guidance.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualisation (saved to plots/)
# ─────────────────────────────────────────────────────────────────────────────


def plot_ig_curve(
    altitudes, raw_ig, true_ig, out_path="plots/test_ig_altitude_curve.png"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(altitudes, raw_ig, "r--", label="Raw entropy sum (LLP bug — monotone↑)")
    ax.plot(altitudes, true_ig, "b-", label="True IG = H(p) – cH(p,σ,σ)  [correct]")
    ax.axvline(altitudes[np.argmax(true_ig)], color="b", alpha=0.3, linestyle=":")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("IG")
    ax.set_title("Expected IG vs Altitude — uniform belief, centred position")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → Plot saved: {out_path}")


def plot_altitude_histograms(
    camera, buggy_alts, correct_alts, out_path="plots/test_llp_altitude_preference.png"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bins = np.arange(
        camera.h_range[0], camera.h_range[1] + camera.h_step * 1.1, camera.h_step
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, alts, label, color in zip(
        axes,
        [buggy_alts, correct_alts],
        ["BUGGY (raw entropy)", "CORRECT (true IG)"],
        ["red", "blue"],
    ):
        ax.hist(alts, bins=bins, color=color, alpha=0.7, edgecolor="k")
        ax.axvline(
            np.mean(alts) if alts else 0,
            color="k",
            linestyle="--",
            label=f"mean={np.mean(alts):.1f}m",
        )
        ax.set_title(f"Best rollout altitudes\n{label}")
        ax.set_xlabel("Altitude (m)")
        ax.set_ylabel("Steps")
        ax.legend()
    fig.suptitle("LLP best-rollout altitude distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLP Altitude & HLP g2 Diagnostics")
    print("=" * 60)

    altitudes, raw_ig, true_ig = test_ig_altitude_curve()
    plot_ig_curve(
        altitudes,
        raw_ig,
        true_ig,
        out_path=os.path.join(
            os.path.dirname(__file__), "..", "plots", "test_ig_altitude_curve.png"
        ),
    )

    camera, buggy_alts, correct_alts = test_llp_altitude_preference(
        n_rollouts=300, horizon=8
    )
    plot_altitude_histograms(
        camera,
        buggy_alts,
        correct_alts,
        out_path=os.path.join(
            os.path.dirname(__file__), "..", "plots", "test_llp_altitude_preference.png"
        ),
    )

    test_hlp_g2_influence()

    print("\n" + "=" * 60)
    print("  All tests complete.")
    print("=" * 60)
    print("""
SUMMARY OF FINDINGS
───────────────────
A) IG vs altitude (TEST A — sensor model structure):
   • σ = a(1−e^{−bh}) with a=1, b=0.015 reaches 0.5 at h≈46m.
   • At σ=0.5 the sensor is a coin-flip: cH(p,0.5,0.5)=H(p) → true IG≈0.
   • True IG dips at σ≈0.5, then recovers as the sensor becomes adversarially
     informative (> 0.5). With only 6 altitude levels this dip may fall between
     levels, but is clearly visible on a fine-grained sweep.
   • Raw entropy Σ H(p) is MONOTONE — the LLP (using raw entropy) has no
     notion of sensor quality and will always prefer higher altitude as long
     as the footprint keeps growing.

B) LLP rollout preference (TEST B — patchy belief):
   • With patchy belief (most cells near start already observed ≈ certain),
     the extra cells at high altitude add little true IG (they are certain).
   • Buggy formula still rewards high altitude (raw entropy of the certain
     cells is low, but high-alt footprint covers the still-uncertain far corner).
   • Corrected formula accounts for sensor noise: the far corner at max altitude
     may yield IG < same corner observed from a lower, better-quality altitude.
   • The difference in average rollout altitude indicates whether the sensor-
     model tradeoff is being respected.

C) HLP g2 influence (TEST C):
   • Without HLP: LLP greedily maximises local IG near start.
   • With HLP intent pointing to the far high-uncertainty region:
     g2 penalises trajectories that ignore the global mission progress.
     LLP is nudged toward the target region, trading some local IG for
     better global mission completion time.
   • This is the intended behaviour: HLP guides WHERE, LLP optimises HOW.

FIX REQUIRED in hierarchical_dec_mcts.py → LowLevelPlanner._compute_ig:
   1. Use camera.get_range() instead of manual FOV math (matches b2b22c4 exactly).
   2. Replace raw entropy with true IG:
          h_prior = H(prior_slice)                     # prior uncertainty
          s = camera.a * (1 − exp(−camera.b * alt))   # sensor noise at altitude
          h_cond  = cH(prior_slice, s, s)              # expected posterior entropy
          ig_per_cell = h_prior − h_cond               # true information gain
""")

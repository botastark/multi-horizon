"""
Smoke test: run greedy IG, Dec-MCTS and both MH-Dec-MCTS variants on the Gaussian map,
camera and sensor model of the main pipeline (main.py, Gaussian branch), for N_STEPS steps
and N_ITERS runs. Run i uses seed SEED + i for its ground-truth map, observation noise,
camera tie-breaking and planner RNGs (as in main.py, iteration_seed = base_seed + iter_idx).
Metrics are averaged over runs.

Checks per step: action is valid for the camera, belief stays finite in [0, 1], and the
belief is updated. A single noisy observation can raise footprint entropy (only the expected
change is a decrease), so the realized per-step IG is reported rather than asserted.

Outputs go to experiments/temp and are written after each planner finishes all its runs:
planner_<name>.txt (step log), planner_path_<name>_run0.png (run 0 path), planner_steps.csv,
planner_metrics.png (mean +- std over runs). planner_viewer_plot.png shows all planners at run 0.

Usage: python tests/test_planners_smoke.py [--iters N]
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from uav_camera import Camera
from helper import H, get_sensor_params, gaussian_random_field, compute_metrics, observed_m_ids, uav_position
from greedy_ig_planner import create_greedy_ig_planner
from dec_mcts import DecMCTSPlanner
from hierarchical_dec_mcts import create_hierarchical_planner, IntentBus

N_STEPS = 100
N_ITERS = 20
DEC_HORIZON = 5
SEED = 42
CLUSTER_RADIUS = 5
SENSOR_A, SENSOR_B = 1.0, 0.015
GRID_SHAPE = (400, 400)
GRID = SimpleNamespace(x=50, y=50, shape=GRID_SHAPE, length=0.125, center=True)
ALLOWED_ACTIONS = {"up", "down", "front", "back", "right", "left", "hover"}
START_XY = (-25.0, -25.0)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "temp")


def make_ground_truth(seed):
    return gaussian_random_field(CLUSTER_RADIUS, GRID_SHAPE, seed)


def make_camera(seed):
    cam = Camera(GRID, 60, camera_altitude=None, f_overlap=None, s_overlap=None,
                 seed=seed, a=SENSOR_A, b=SENSOR_B)
    cam.set_position(START_XY)
    cam.set_altitude(cam.h_range[0])
    conf_dict = cam.theoretical_conf_dict()
    return cam, conf_dict


def observe_and_update(belief, cam, conf_dict, pos, alt, gt, obs_rng):
    """Simulate a noisy observation at pos/alt and Bayes-update the occupancy belief."""
    [[imin, imax], [jmin, jmax]] = cam.get_range(position=pos, altitude=alt, index_form=True)
    m = gt[imin:imax, jmin:jmax]
    s0, s1 = get_sensor_params(alt, conf_dict)
    flip = obs_rng.random(m.shape)
    z = np.where(m == 1, (flip >= s1).astype(int), (flip < s0).astype(int))

    prior = belief[imin:imax, jmin:jmax, 1]
    lik1 = np.where(z == 1, 1.0 - s1, s1)
    lik0 = np.where(z == 1, s0, 1.0 - s0)
    post = lik1 * prior / (lik1 * prior + lik0 * (1.0 - prior))

    entropy_before = float(np.sum(H(prior)))
    belief[imin:imax, jmin:jmax, 1] = post
    belief[imin:imax, jmin:jmax, 0] = 1.0 - post
    entropy_after = float(np.sum(H(post)))
    return entropy_before, entropy_after


def one_step_ig(cam, conf_dict, belief, pos, alt):
    """One-step IG of every admissible action from pos/alt on belief (same formula for all planners)."""
    scorer = create_greedy_ig_planner(
        agent_id=0, camera=cam, grid_info=GRID, conf_dict=conf_dict,
        config={"enable_discounting": False, "overlap_penalty_weight": 0.0}, seed=0,
    )
    scorer.update_belief(belief)
    scored = scorer.score_admissible_actions(pos, alt)
    return {action: data["ig"] for action, data in scored.items()}


def plan_greedy(cam, conf_dict, belief, pos, alt, seed):
    planner = create_greedy_ig_planner(
        agent_id=0, camera=cam, grid_info=GRID, conf_dict=conf_dict,
        config={"enable_discounting": False, "overlap_penalty_weight": 0.0}, seed=seed,
    )
    planner.update_belief(belief)
    intent = planner.plan(current_position=pos, current_altitude=alt)
    assert np.isfinite(intent.expected_ig), "greedy expected_ig not finite"
    return intent.action, dict(planner.get_action_scores())


def plan_dec_mcts(cam, conf_dict, belief, pos, alt, seed):
    planner = DecMCTSPlanner(
        agent_id=0, camera=cam, grid_info=GRID, conf_dict=conf_dict,
        config={"horizon": DEC_HORIZON, "iterations": 20, "timeout": 30.0}, seed=seed,
    )
    planner.update_state(pos, alt, belief)
    intent = planner.plan()
    assert 1 <= len(intent.action_sequence) <= DEC_HORIZON, "dec-mcts plan empty or longer than horizon"
    assert len(intent.ig_sequence) == len(intent.action_sequence), "dec-mcts ig/action length mismatch"
    return intent.action_sequence[0], dict(planner._mcts_action_values)


def plan_mh_dec_mcts(cam, conf_dict, belief, pos, alt, seed, use_mcts_llp):
    planner = create_hierarchical_planner(
        agent_id=0, num_agents=1, camera=cam, grid_info=GRID,
        intent_bus=IntentBus(1),
        config={"llp_horizon": 3, "llp_iterations": 20, "hlp_horizon": 2,
                "hlp_iterations": 20, "tile_size": [50, 50],
                "use_mcts_llp": use_mcts_llp},
        conf_dict=conf_dict, seed=seed,
    )
    planner.update_state(pos, alt, belief)
    action, _info = planner.plan()
    own_values = getattr(planner.llp, "_mcts_action_values", None)
    return action, dict(own_values) if own_values else None


PLANNERS = {
    "greedy_ig": plan_greedy,
    "dec_mcts": plan_dec_mcts,
    "mh_dec_mcts_full": lambda cam, c, b, p, a, s: plan_mh_dec_mcts(cam, c, b, p, a, s, use_mcts_llp=True),
    "mh_dec_mcts_efficient": lambda cam, c, b, p, a, s: plan_mh_dec_mcts(cam, c, b, p, a, s, use_mcts_llp=False),
}


def run_planner(name, plan_fn, gt, seed):
    cam, conf_dict = make_camera(seed)
    belief = np.full((*GRID_SHAPE, 2), 0.5)
    obs_rng = np.random.default_rng(seed)
    pos, alt = START_XY, float(cam.altitude)

    records = []
    observed = set()
    path = [(pos[0], pos[1], alt)]
    for step in range(N_STEPS):
        ig_table = one_step_ig(cam, conf_dict, belief.copy(), pos, alt)
        action, own_scores = plan_fn(cam, conf_dict, belief.copy(), pos, alt, seed)
        assert action in ALLOWED_ACTIONS, f"{name} step {step}: unknown action {action!r}"
        assert action in cam.permitted_actions(cam.get_x()), f"{name} step {step}: {action} not permitted"

        nxt = cam.x_future(action)
        if nxt is not None:
            new_pos, new_alt = nxt
            pos = (float(new_pos[0]), float(new_pos[1]))
            alt = float(new_alt)
        cam.set_position(pos)
        cam.set_altitude(alt)
        path.append((pos[0], pos[1], alt))

        h_before, h_after = observe_and_update(belief, cam, conf_dict, pos, alt, gt, obs_rng)
        assert np.isfinite(h_after), f"{name} step {step}: entropy not finite"
        assert np.all(np.isfinite(belief)), f"{name} step {step}: belief not finite"
        assert belief.min() >= 0.0 and belief.max() <= 1.0, f"{name} step {step}: belief out of [0,1]"
        observed.update(observed_m_ids(cam, uav_position((pos, alt))))
        entropy, mse, coverage = compute_metrics(gt, belief, observed, GRID)
        records.append({"action": action, "alt": round(alt, 2), "dH": round(h_before - h_after, 3),
                        "ig": ig_table, "own": own_scores,
                        "fp": cam.get_range(position=pos, altitude=alt),
                        "entropy": entropy, "mse": mse, "coverage": coverage})

    assert not np.all(belief[..., 1] == 0.5), f"{name}: belief never updated"
    return records, path, belief[..., 1].copy()


def format_ig(table):
    return ", ".join(f"{a}={v:.1f}" for a, v in sorted(table.items()))


def plot_viewer_style(paths, footprints, beliefs, gt, out_path, title):
    """One row per planner: 3D terrain with path, top view with final footprint, final belief map."""
    stride = 4
    x_vals = np.arange(-GRID.x / 2, GRID.x / 2, GRID.length)
    y_vals = np.arange(-GRID.y / 2, GRID.y / 2, GRID.length)
    Xg, Yg = np.meshgrid(x_vals, -y_vals)
    Xs, Ys = Xg[::stride, ::stride], Yg[::stride, ::stride]
    gt_s = gt[::stride, ::stride]
    rgba = np.empty(gt_s.shape + (4,))
    rgba[gt_s == 0] = to_rgba("yellow")
    rgba[gt_s == 1] = to_rgba("darkgreen")
    extent = [-GRID.x / 2, GRID.x / 2, -GRID.y / 2, GRID.y / 2]
    top_cmap = plt.get_cmap("viridis")
    belief_cmap = LinearSegmentedColormap.from_list("belief", ["lemonchiffon", "darkgreen"])
    names = list(paths)

    fig = plt.figure(figsize=(14, 4.6 * len(names)))
    for row, name in enumerate(names):
        xs, ys, zs = zip(*paths[name])
        n = len(xs)

        ax3 = fig.add_subplot(len(names), 2, 2 * row + 1, projection="3d")
        ax3.plot_surface(Xs, Ys, np.zeros_like(Xs), facecolors=rgba, alpha=0.3,
                         linewidth=0, antialiased=False, shade=False)
        for i in range(n - 1):
            ax3.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2], color=top_cmap(i / max(1, n - 2)), linewidth=2)
        ax3.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=50, label="start")
        ax3.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="blue", marker="^", s=80, label="end")
        ax3.set_xlim(extent[0], extent[1])
        ax3.set_ylim(extent[2], extent[3])
        ax3.set_zlim(0, max(zs) + 2)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Altitude (m)")
        ax3.set_title(f"{name}: 3D terrain & path")
        ax3.view_init(elev=35, azim=-60)
        ax3.legend(loc="upper left", fontsize=8)

        ax2 = fig.add_subplot(len(names), 2, 2 * row + 2)
        im = ax2.imshow(beliefs[name], cmap=belief_cmap, origin="upper", extent=extent, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        for i in range(n - 1):
            ax2.plot(xs[i:i + 2], ys[i:i + 2], color=top_cmap(i / max(1, n - 2)), linewidth=2.5)
        fx, fy = footprints[name][-1]
        ax2.plot([fx[0], fx[1], fx[1], fx[0], fx[0]], [fy[0], fy[0], fy[1], fy[1], fy[0]],
                 color="red", lw=1.5)
        ax2.scatter([xs[0]], [ys[0]], c="green", s=80, zorder=5)
        ax2.scatter([xs[-1]], [ys[-1]], c="blue", marker="X", s=80, zorder=5)
        for k, (x, y) in enumerate(zip(xs, ys)):
            ax2.annotate(str(k), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax2.set_xlim(extent[0] - 1, extent[1] + 1)
        ax2.set_ylim(extent[2] - 1, extent[3] + 1)
        ax2.set_aspect("equal")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"{name}: top view, final belief P(occupied)")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_metrics_comparison(runs_by_planner, out_path):
    """Analysis figure in the layout of viewer.plot_metrics, mean +- std over runs, planners overlaid."""
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    markers = ["^", "s", "d", "o"]
    panels = [
        ("mse", "MSE", "MSE over Steps"),
        ("alt", "Altitude (m)", "Altitude over Steps"),
        ("coverage", "Coverage", "Coverage over Steps"),
        ("dH", "IG per step (bits)", "Information Gain per Step"),
    ]
    n_runs = len(next(iter(runs_by_planner.values())))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for k, (name, runs) in enumerate(runs_by_planner.items()):
        color, marker = colors[k % len(colors)], markers[k % len(markers)]
        for ax, (key, _, _) in zip(axes.flat, panels):
            arr = np.array([[rec[key] for rec in run] for run in runs])
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            steps = np.arange(1, arr.shape[1] + 1)
            ax.plot(steps, mean, color=color, marker=marker, markersize=4, linewidth=2, label=name)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.25)
    for ax, (_, ylabel, title) in zip(axes.flat, panels):
        ax.set_xlabel("Number of steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Mean with +- std band over {n_runs} runs")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_step_table(runs_by_planner, out_path):
    """CSV with one row per (run, step); each planner column holds 'action | one-step IG per action'."""
    names = list(runs_by_planner)
    n_runs = len(next(iter(runs_by_planner.values())))
    n_steps = len(next(iter(runs_by_planner.values()))[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step"] + names)
        for run in range(n_runs):
            for step in range(n_steps):
                row = [run, step]
                for name in names:
                    rec = runs_by_planner[name][run][step]
                    row.append(f"{rec['action']} | {format_ig(rec['ig'])}")
                writer.writerow(row)


def save_planner_log(name, runs, out_path):
    """Readable text log for one planner: per run, per step action, altitude, IG and metrics."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"planner: {name}\nruns: {len(runs)} (seeds {SEED}..{SEED + len(runs) - 1}), steps: {N_STEPS}\n")
        for run_idx, records in enumerate(runs):
            last = records[-1]
            f.write(f"\n=== run {run_idx} (seed {SEED + run_idx}) final: mse={last['mse']:.5f} "
                    f"coverage={last['coverage']:.4f} mean IG={np.mean([r['dH'] for r in records]):.1f} bits ===\n")
            for step, rec in enumerate(records):
                f.write(f"step {step:3d} action={rec['action']:<6} alt={rec['alt']:6.2f} dH={rec['dH']:10.2f} "
                        f"mse={rec['mse']:.5f} coverage={rec['coverage']:.4f} | IG: {format_ig(rec['ig'])}\n")
                if rec["own"] is not None:
                    f.write(f"         planner's own scores: {format_ig(rec['own'])}\n")


def print_summary(runs_by_planner):
    print("\n=== summary over runs (final step, mean +- std) ===")
    for name, runs in runs_by_planner.items():
        finals = {key: np.array([run[-1][key] for run in runs]) for key in ("mse", "coverage")}
        ig = np.array([np.mean([rec["dH"] for rec in run]) for run in runs])
        print(f"  {name:<24} mse={finals['mse'].mean():.3f}+-{finals['mse'].std():.3f}  "
              f"coverage={finals['coverage'].mean():.3f}+-{finals['coverage'].std():.3f}  "
              f"mean IG/step={ig.mean():.1f}+-{ig.std():.1f} bits")


def save_outputs_after_planner(name, runs, run0, runs_by_planner, out_dir):
    """Write this planner's log, run-0 path figure, and the cumulative CSV and metrics plot."""
    path0, fp0, belief0, gt0 = run0
    log_path = os.path.join(out_dir, f"planner_{name}.txt")
    save_planner_log(name, runs, log_path)
    print(f"  saved {os.path.normpath(log_path)}", flush=True)

    fig_path = os.path.join(out_dir, f"planner_path_{name}_run0.png")
    plot_viewer_style({name: path0}, {name: fp0}, {name: belief0}, gt0, fig_path,
                      title=f"{name}: run 0 (seed {SEED}) path on the Gaussian map")
    print(f"  saved {os.path.normpath(fig_path)}", flush=True)

    csv_path = os.path.join(out_dir, "planner_steps.csv")
    save_step_table(runs_by_planner, csv_path)
    metrics_path = os.path.join(out_dir, "planner_metrics.png")
    plot_metrics_comparison(runs_by_planner, metrics_path)
    print(f"  updated {os.path.normpath(csv_path)} and {os.path.normpath(metrics_path)}", flush=True)


def test_planners_run_on_same_map(out_dir=None, n_iters=N_ITERS):
    out_dir = out_dir or OUT_DIR
    runs_by_planner, run0_by_planner = {}, {}
    for name, plan_fn in PLANNERS.items():
        print(f"\nRunning {name} ({n_iters} runs)...", flush=True)
        runs = []
        for iter_idx in range(n_iters):
            seed = SEED + iter_idx
            gt = make_ground_truth(seed)
            records, path, belief = run_planner(name, plan_fn, gt, seed)
            runs.append(records)
            print(f"  run {iter_idx + 1}/{n_iters} (seed {seed}): final mse={records[-1]['mse']:.3f} "
                  f"coverage={records[-1]['coverage']:.3f}", flush=True)
            if iter_idx == 0:
                run0_by_planner[name] = ([(x, y, z) for x, y, z in path],
                                         [rec["fp"] for rec in records], belief, gt)
        runs_by_planner[name] = runs
        save_outputs_after_planner(name, runs, run0_by_planner[name], runs_by_planner, out_dir)

    print_summary(runs_by_planner)
    paths = {name: run0[0] for name, run0 in run0_by_planner.items()}
    footprints = {name: run0[1] for name, run0 in run0_by_planner.items()}
    beliefs = {name: run0[2] for name, run0 in run0_by_planner.items()}
    gt0 = next(iter(run0_by_planner.values()))[3]
    all_path = os.path.join(out_dir, "planner_viewer_plot.png")
    plot_viewer_style(paths, footprints, beliefs, gt0, all_path, title="Planner paths on run 0 (Gaussian map)")
    print(f"\nall-planner run-0 figure saved to {os.path.normpath(all_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=N_ITERS, help="number of runs to average over")
    args = parser.parse_args()
    test_planners_run_on_same_map(n_iters=args.iters)
    print("✓ planner smoke test passed")

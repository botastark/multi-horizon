"""
Smoke test: run greedy IG, Dec-MCTS and both MH-Dec-MCTS variants through the real pipeline
(experiment_runner.run_multi_agent_experiment, Simulator, OG + LBP mapper, news fusion) so it
measures what the experiments report.

Cases (field r=4, adaptive correlation, one greedy reference per case):
    n1      N=1: IG only. No neighbours, so IGd equals IG and BM vs BS makes no difference.
    case1   N=4, one UAV per corner, R=5,   BM news, greedy IGd
    case2   N=4, one UAV per corner, R=inf, BS news, greedy IG

Planner settings come from configs/strategies/*.json unchanged; communication, news and field
settings come from the case. Each planner is built once per run, so its RNG stream, intents and
HLP plan persist across steps.

Seeds: --split tune uses 1000-1019 and is the default. --split report uses 42-61 (the seeds
main.py runs) and is for final tables only, so tuning never sees them.

Metrics use one formula for every planner: MSE of the mean belief across agents, entropy of that
mean belief, and coverage as the fraction of cells inside any footprint so far. Simulator's own
fused_* histories cannot be compared across planners, because it uses a different MSE and
coverage for greedy than for the MCTS planners. A step means one round of observe, update, fuse.

Summary per case: MSE area, steps to MSE<0.05 and <0.01, steps to full coverage. Planners show
the per-seed difference from greedy on the same seed as mean +- SE, so negative is better than
greedy. A run that never reaches a target counts as n_steps and "reached" shows how many did, so
a difference is a lower bound when reached is below the run count.

Compute: one reward evaluation is one footprint IG calculation or one g2 call, whichever planner
makes it. Evaluations and planning wall time are logged per step. --eval-budget E sets each MCTS
planner's iterations so it spends about E evaluations per decision, for comparison at equal
evaluations, using pilots on a tune seed. Greedy stays at one evaluation per admissible action.
The MH HLP replans on a wall-clock timer, so its share depends on machine speed and load: keep
--jobs low when wall time or MH evaluation counts matter.

Outputs go to experiments/temp/smoke/<split>/<case>[_e<budget>] after each planner finishes all its
runs: planner_<name>.txt (step log), planner_path_<name>_run0.png (run 0 paths), planner_steps.csv,
planner_compute.csv, planner_metrics.png (mean +- std over runs), planner_summary.csv, and
planner_viewer_plot.png (all planners at run 0).

Usage: python tests/test_planners_smoke.py [--case all] [--split tune] [--iters 20] [--steps 100]
                                           [--planners a b] [--eval-budget E] [--jobs 1]
"""

import argparse
import csv
import functools
import json
import os
import sys
import tempfile
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import experiment_runner
import g2_evaluator
import simulator as simulator_module
from config_loader import load_config
from dec_mcts import DecMCTSNode, DecMCTSPlanner
from greedy_ig_planner import GreedyIGPlanner, create_greedy_ig_planner
from helper import compute_entropy, compute_mse, gaussian_random_field
from hierarchical_dec_mcts import HierarchicalDecMCTSPlanner, LowLevelPlanner
from mapper_LBP import OccupancyMap
from orthomap import Field
from simulator import Simulator
from uav_camera import Camera

N_STEPS = 100
N_ITERS = 20
CLUSTER_RADIUS = 4
SENSOR_A, SENSOR_B = 1.0, 0.015
GRID_SHAPE = (400, 400)
GRID = SimpleNamespace(x=50, y=50, shape=GRID_SHAPE, length=0.125, center=True)
ALLOWED_ACTIONS = {"up", "down", "front", "back", "right", "left", "hover"}
OUT_DIR = os.path.join(ROOT, "experiments", "temp", "smoke")

TUNE_SEEDS = tuple(range(1000, 1020))
REPORT_SEEDS = tuple(range(42, 62))
SEED_SPLITS = {"tune": TUNE_SEEDS, "report": REPORT_SEEDS}

Case = namedtuple("Case", "label num_agents mode config")
CASES = {
    "n1": Case("N=1: IG", 1, "IG", "configs/baseline_greedy_ig_bs_rinf.json"),
    "case1": Case("Case 1: N=4, R=5, BM, IGd", 4, "IGd_BM", "configs/baseline_greedy_igd_bm_r5.json"),
    "case2": Case("Case 2: N=4, R=inf, BS, IG", 4, "IG_BS", "configs/baseline_greedy_ig_bs_rinf.json"),
}

REFERENCE = "greedy_ig"
STRATEGY_CONFIGS = {
    "dec_mcts": "configs/strategies/dec_mcts.json",
    "mh_dec_mcts_full": "configs/strategies/mh_dec_mcts_full.json",
    "mh_dec_mcts_efficient": "configs/strategies/mh_dec_mcts_efficient.json",
}
PLANNERS = (REFERENCE, *STRATEGY_CONFIGS)
# Where each strategy keeps the iteration budget that --eval-budget scales.
ITERATION_KEYS = {
    "dec_mcts": ("dec_mcts", "iterations"),
    "mh_dec_mcts_full": ("hierarchical_dec_mcts", "llp_iterations"),
    "mh_dec_mcts_efficient": ("hierarchical_dec_mcts", "llp_iterations"),
}
CALIBRATION_STEPS = 20

MSE_TARGETS = {"steps_mse_05": 0.05, "steps_mse_01": 0.01}
METRICS = {
    "mse_area": "MSE area",
    "steps_mse_05": "steps MSE<0.05",
    "steps_mse_01": "steps MSE<0.01",
    "steps_full_cov": "steps full cov",
}


# ---------------------------------------------------------------------------
# Compute logging: no planner code changes, so every planner is counted the same way
# ---------------------------------------------------------------------------

_COUNTS = {"ig": 0, "g2": 0}
_NON_PLANNING_SECONDS = [0.0]


def _count_calls(owner, name, key, amount=lambda result: 1):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        _COUNTS[key] += amount(result)
        return result

    setattr(owner, name, wrapper)


def _exclude_from_planning_time(owner, name):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _NON_PLANNING_SECONDS[0] += time.perf_counter() - start

    setattr(owner, name, wrapper)


_count_calls(GreedyIGPlanner, "score_admissible_actions", "ig", len)
_count_calls(DecMCTSNode, "compute_ig_reward", "ig")
_count_calls(LowLevelPlanner, "_compute_ig", "ig")
_count_calls(g2_evaluator, "g2", "g2")
# Simulator computes per-agent metrics inside its action-selection phase.
_exclude_from_planning_time(simulator_module, "compute_metrics")
_exclude_from_planning_time(simulator_module, "observed_m_ids")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_config(case_name, strategy, seed, n_steps=N_STEPS, iterations=None):
    """Case settings from the baseline config, planner settings from the strategy JSON."""
    case = CASES[case_name]
    config = load_config(os.path.join(ROOT, case.config))[0]
    if config["cluster_radius"] != CLUSTER_RADIUS:
        raise ValueError(f"{case.config}: cluster_radius {config['cluster_radius']}, expected {CLUSTER_RADIUS}")
    config.update(
        action_strategy=strategy,
        num_agents=case.num_agents,
        n_steps=n_steps,
        seed=seed,
        enable_plotting=False,
        enable_logging=False,
    )
    config["multi_agent"]["num_agents"] = case.num_agents
    if strategy != REFERENCE:
        planner = load_config(os.path.join(ROOT, STRATEGY_CONFIGS[strategy]))[0]
        section, key = ITERATION_KEYS[strategy]
        config[section] = planner[section]
        config["decentralized"]["overlap_penalty_weight"] = planner["decentralized"]["overlap_penalty_weight"]
        if iterations is not None:
            config[section][key] = iterations
    return config


def build_simulator(case_name, strategy, seed, results_folder, n_steps=N_STEPS, iterations=None, config=None):
    """Everything run_multi_agent_experiment builds, stopped just before the run starts."""
    case = CASES[case_name]
    config = config or build_config(case_name, strategy, seed, n_steps, iterations)
    camera = Camera(GRID, 60, camera_altitude=None, f_overlap=None, s_overlap=None,
                    seed=seed, a=SENSOR_A, b=SENSOR_B)
    hrange = camera.get_hrange()
    field_map = Field(GRID, CLUSTER_RADIUS, sweep=strategy, h_range=hrange, seed=seed)
    field_map.reset(seed=seed)

    original = experiment_runner.Simulator
    experiment_runner.Simulator = MeasuredSimulator
    try:
        return experiment_runner.run_multi_agent_experiment(
            config=config, grid_info=GRID, map_obj=field_map,
            ground_truth_map=field_map.get_ground_truth(),
            conf_dict=camera.theoretical_conf_dict(), results_folder=results_folder,
            corr_type=config["correlation_types"][0], e_margin=None, grf_r=CLUSTER_RADIUS,
            iter_idx=seed, n_steps=n_steps, ENABLE_STEPWISE_PLOTTING=False, ENABLE_LOGGING=False,
            mcts_params={}, action_strategy=strategy, min_alt=hrange[0], overlap=None,
            optimal_alt=21.5, seed=seed, camera_hrange=hrange, news_mode=case.mode,
        )
    finally:
        experiment_runner.Simulator = original


def team_belief(agents):
    """P(occupied) averaged over agents, the one map every planner is scored on."""
    return np.mean([agent["belief_map"][:, :, 1] for agent in agents], axis=0)


def one_step_ig(agent):
    """One-step IG of every admissible action for this agent (same formula for all planners)."""
    scorer = create_greedy_ig_planner(
        agent_id=agent["agent_id"], camera=agent["camera"], grid_info=GRID,
        conf_dict=agent["planner"].conf_dict,
        config={"enable_discounting": False, "overlap_penalty_weight": 0.0}, seed=0,
    )
    scorer.update_belief(agent["belief_map"])
    scored = scorer.score_admissible_actions(agent["uav_pos"].position, agent["uav_pos"].altitude)
    return {action: data["ig"] for action, data in scored.items()}


def own_action_scores(agent):
    """What the planner itself thinks each action is worth."""
    planner = agent["planner"]
    if hasattr(planner, "_dec_mcts_planner"):
        values = planner._dec_mcts_planner.get_action_values()
    elif hasattr(planner, "_hierarchical_planner"):
        values = planner._hierarchical_planner.llp._mcts_action_values
    else:
        values = agent["info_gain_action"]
    return {str(a): float(v) for a, v in values.items() if not str(a).startswith("_") and np.isfinite(v)}


class MeasuredSimulator(Simulator):
    """The real simulator loop with per-step logging added around its phases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._observed = np.zeros(self.grid_info.shape, dtype=bool)

    def run(self):
        """run_multi_agent_experiment calls this: hand the built simulator back instead."""
        return self

    def run_measured(self):
        Simulator.run(self)
        return self.records

    def _select_agent_actions(self, step):
        counts, non_planning = dict(_COUNTS), _NON_PLANNING_SECONDS[0]
        start = time.perf_counter()
        super()._select_agent_actions(step)
        seconds = time.perf_counter() - start - (_NON_PLANNING_SECONDS[0] - non_planning)
        self._step_log = {
            "ig_evals": _COUNTS["ig"] - counts["ig"],
            "g2_evals": _COUNTS["g2"] - counts["g2"],
            "plan_ms": seconds * 1000.0,
            "agents": [self._decision(agent, step) for agent in self.agents],
            "entropy_before": compute_entropy(_as_belief(team_belief(self.agents))),
        }

    def _decision(self, agent, step):
        camera, position = agent["camera"], agent["uav_pos"]
        action = str(agent["_next_action"])
        name = f"{self.action_strategy} agent {agent['agent_id']} step {step}"
        assert action in ALLOWED_ACTIONS, f"{name}: unknown action {action!r}"
        assert action in camera.permitted_actions(camera.get_x()), f"{name}: {action} not permitted"
        dec_mcts = getattr(agent["planner"], "_dec_mcts_planner", None)
        if dec_mcts is not None and dec_mcts.current_intent is not None:
            intent = dec_mcts.current_intent
            assert len(intent.action_sequence) <= dec_mcts.config["horizon"], f"{name}: plan longer than horizon"
            assert len(intent.ig_sequence) == len(intent.action_sequence), f"{name}: ig/action length mismatch"
        return {"action": action, "from": (position.position[0], position.position[1], position.altitude),
                "ig": one_step_ig(agent), "own": own_action_scores(agent)}

    def _plot_step(self, step, agent_observations):
        """Last phase of every step, so everything here is after observation and fusion."""
        super()._plot_step(step, agent_observations)
        for agent in self.agents:
            position = agent["uav_pos"]
            [[imin, imax], [jmin, jmax]] = agent["camera"].get_range(
                position=position.position, altitude=position.altitude, index_form=True
            )
            self._observed[imin:imax, jmin:jmax] = True
        occupied = team_belief(self.agents)
        name = f"{self.action_strategy} step {step}"
        assert np.all(np.isfinite(occupied)), f"{name}: belief not finite"
        assert occupied.min() >= 0.0 and occupied.max() <= 1.0, f"{name}: belief out of [0,1]"
        entropy = compute_entropy(_as_belief(occupied))
        log = self._step_log
        self.records.append({
            "agents": [
                {**decision, "fp": agent["camera"].get_range(position=agent["uav_pos"].position,
                                                             altitude=agent["uav_pos"].altitude),
                 "to": (agent["uav_pos"].position[0], agent["uav_pos"].position[1], agent["uav_pos"].altitude)}
                for decision, agent in zip(log["agents"], self.agents)
            ],
            "alt": float(np.mean([agent["uav_pos"].altitude for agent in self.agents])),
            "dH": float(log["entropy_before"] - entropy),
            "entropy": float(entropy),
            "mse": float(compute_mse(self.ground_truth_map, occupied)),
            "coverage": float(self._observed.mean()),
            "ig_evals": log["ig_evals"],
            "g2_evals": log["g2_evals"],
            "plan_ms": log["plan_ms"],
        })


def _as_belief(occupied):
    return np.stack([1.0 - occupied, occupied], axis=2)


def run_planner(case_name, strategy, seed, n_steps=N_STEPS, iterations=None):
    """One full run. Returns step records, per-agent paths, final team belief, ground truth."""
    with tempfile.TemporaryDirectory() as tmp:
        simulator = build_simulator(case_name, strategy, seed, tmp, n_steps, iterations)
        records = simulator.run_measured()
    occupied = team_belief(simulator.agents)
    assert not np.all(occupied == 0.5), f"{strategy}: belief never updated"
    paths = [[rec["agents"][i]["from"] for rec in records[:1]] + [rec["agents"][i]["to"] for rec in records]
             for i in range(len(simulator.agents))]
    return records, paths, occupied, simulator.ground_truth_map


def _run_planner_task(task):
    return run_planner(*task)


# ---------------------------------------------------------------------------
# Compute: equal evaluations
# ---------------------------------------------------------------------------


def evaluations_per_decision(runs):
    """Mean IG + g2 calls per agent per step over a list of runs."""
    decisions = sum(len(rec["agents"]) for run in runs for rec in run)
    return sum(rec["ig_evals"] + rec["g2_evals"] for run in runs for rec in run) / max(1, decisions)


def calibrate_iterations(case_name, strategy, target, steps=CALIBRATION_STEPS, seed=TUNE_SEEDS[0]):
    """Iterations that spend about `target` evaluations per decision, from pilots on a tune seed."""
    section, key = ITERATION_KEYS[strategy]
    base = build_config(case_name, strategy, seed)[section][key]
    measured = {}
    for iterations in (base, 2 * base):
        records = run_planner(case_name, strategy, seed, steps, iterations)[0]
        measured[iterations] = evaluations_per_decision([records])
    slope = (measured[2 * base] - measured[base]) / base
    intercept = measured[base] - slope * base
    return max(1, int(round((target - intercept) / slope)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def first_step(condition):
    """1-based step at which condition first holds, or None if it never does."""
    hits = np.flatnonzero(condition)
    return int(hits[0]) + 1 if hits.size else None


def run_metrics(records):
    mse = np.array([rec["mse"] for rec in records])
    coverage = np.array([rec["coverage"] for rec in records])
    metrics = {"mse_area": float(np.sum((mse[1:] + mse[:-1]) / 2.0))}
    for key, target in MSE_TARGETS.items():
        metrics[key] = first_step(mse < target)
    metrics["steps_full_cov"] = first_step(coverage >= 1.0)
    return metrics


def metric_values(runs, key):
    """Per-run values of a metric, with a missed target counted as n_steps, and how many reached."""
    values, reached = [], 0
    for records in runs:
        value = run_metrics(records)[key]
        reached += value is not None
        values.append(len(records) if value is None else value)
    return np.array(values, dtype=float), reached


def paired_difference(values, reference):
    """Mean and standard error of the per-seed difference from the reference."""
    difference = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    se = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else float("nan")
    return float(difference.mean()), float(se)


def compute_summary(runs):
    decisions = sum(len(rec["agents"]) for run in runs for rec in run)
    return {
        "evals": evaluations_per_decision(runs),
        "g2": sum(rec["g2_evals"] for run in runs for rec in run) / max(1, decisions),
        "ms": sum(rec["plan_ms"] for run in runs for rec in run) / max(1, decisions),
    }


def summary_rows(case_name, runs_by_planner):
    """One row per planner and metric: absolute mean, difference from greedy, reached count."""
    reference = runs_by_planner[REFERENCE]
    rows = []
    for name, runs in runs_by_planner.items():
        for key in METRICS:
            values, reached = metric_values(runs, key)
            base, _ = metric_values(reference, key)
            diff, se = paired_difference(values, base)
            rows.append({"case": case_name, "planner": name, "metric": key, "mean": values.mean(),
                         "diff_from_greedy": diff, "diff_se": se, "reached": reached, "runs": len(runs)})
        compute = compute_summary(runs)
        for key, label in (("evals", "evals_per_decision"), ("g2", "g2_per_decision"), ("ms", "plan_ms_per_decision")):
            rows.append({"case": case_name, "planner": name, "metric": label, "mean": compute[key],
                         "diff_from_greedy": "", "diff_se": "", "reached": "", "runs": len(runs)})
    return rows


def print_summary(case_name, runs_by_planner, seeds, iterations_by_planner=None):
    n_steps = len(next(iter(runs_by_planner.values()))[0])
    print(f"\n=== {CASES[case_name].label} | seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} runs), {n_steps} steps ===")
    print("Planners show the per-seed difference from greedy as mean +- SE, negative is better.")
    print(f"{'planner':<23}" + "".join(f"{label:>24}" for label in METRICS.values())
          + f"{'evals/dec':>11}{'g2/dec':>9}{'ms/dec':>9}")
    reference = runs_by_planner[REFERENCE]
    for name, runs in runs_by_planner.items():
        cells = []
        for key in METRICS:
            values, reached = metric_values(runs, key)
            base, _ = metric_values(reference, key)
            tail = "" if key == "mse_area" else f" [{reached}/{len(runs)}]"
            if name == REFERENCE:
                text = f"{values.mean():.2f}{tail}"
            else:
                diff, se = paired_difference(values, base)
                text = f"{diff:+.2f}+-{se:.2f}{tail}"
            cells.append(f"{text:>24}")
        compute = compute_summary(runs)
        print(f"{name:<23}" + "".join(cells) + f"{compute['evals']:>11.1f}{compute['g2']:>9.1f}{compute['ms']:>9.1f}")
    if iterations_by_planner:
        print("Iterations used for the evaluation budget: "
              + ", ".join(f"{name}={i}" for name, i in iterations_by_planner.items()))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def format_ig(table):
    return ", ".join(f"{a}={v:.1f}" for a, v in sorted(table.items()))


def plot_viewer_style(paths, footprints, beliefs, gt, out_path, title):
    """One row per planner: 3D terrain with paths, top view with final footprints, final belief map."""
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
        ax3 = fig.add_subplot(len(names), 2, 2 * row + 1, projection="3d")
        ax3.plot_surface(Xs, Ys, np.zeros_like(Xs), facecolors=rgba, alpha=0.3,
                         linewidth=0, antialiased=False, shade=False)
        ax2 = fig.add_subplot(len(names), 2, 2 * row + 2)
        im = ax2.imshow(beliefs[name], cmap=belief_cmap, origin="upper", extent=extent, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        top = 0.0
        for agent, path in enumerate(paths[name]):
            xs, ys, zs = zip(*path)
            n = len(xs)
            top = max(top, max(zs))
            for i in range(n - 1):
                color = top_cmap(i / max(1, n - 2))
                ax3.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2], color=color, linewidth=2)
                ax2.plot(xs[i:i + 2], ys[i:i + 2], color=color, linewidth=2.5)
            labels = ("start", "end") if agent == 0 else (None, None)
            ax3.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=50, label=labels[0])
            ax3.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="blue", marker="^", s=80, label=labels[1])
            fx, fy = footprints[name][agent]
            ax2.plot([fx[0], fx[1], fx[1], fx[0], fx[0]], [fy[0], fy[0], fy[1], fy[1], fy[0]], color="red", lw=1.5)
            ax2.scatter([xs[0]], [ys[0]], c="green", s=80, zorder=5)
            ax2.scatter([xs[-1]], [ys[-1]], c="blue", marker="X", s=80, zorder=5)
            if len(paths[name]) == 1:
                for k, (x, y) in enumerate(zip(xs, ys)):
                    ax2.annotate(str(k), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax3.set_xlim(extent[0], extent[1])
        ax3.set_ylim(extent[2], extent[3])
        ax3.set_zlim(0, top + 2)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Altitude (m)")
        ax3.set_title(f"{name}: 3D terrain & path")
        ax3.view_init(elev=35, azim=-60)
        ax3.legend(loc="upper left", fontsize=8)
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
                    agents = runs_by_planner[name][run][step]["agents"]
                    row.append(" || ".join(f"{a['action']} | {format_ig(a['ig'])}" for a in agents))
                writer.writerow(row)


def save_compute_log(runs_by_planner, seeds, out_path):
    """CSV with one row per (planner, seed, step): reward evaluations and planning wall time."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", "seed", "step", "ig_evals", "g2_evals", "plan_ms"])
        for name, runs in runs_by_planner.items():
            for seed, records in zip(seeds, runs):
                for step, rec in enumerate(records):
                    writer.writerow([name, seed, step, rec["ig_evals"], rec["g2_evals"], f"{rec['plan_ms']:.3f}"])


def save_summary_table(case_name, runs_by_planner, out_path):
    rows = summary_rows(case_name, runs_by_planner)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_planner_log(name, runs, seeds, out_path):
    """Readable text log for one planner: per run, per step actions, altitude, IG and metrics."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"planner: {name}\nruns: {len(runs)} (seeds {seeds[0]}..{seeds[-1]}), steps: {len(runs[0])}\n")
        for seed, records in zip(seeds, runs):
            last = records[-1]
            f.write(f"\n=== seed {seed} final: mse={last['mse']:.5f} coverage={last['coverage']:.4f} "
                    f"mean IG={np.mean([r['dH'] for r in records]):.1f} bits ===\n")
            for step, rec in enumerate(records):
                f.write(f"step {step:3d} dH={rec['dH']:10.2f} mse={rec['mse']:.5f} coverage={rec['coverage']:.4f} "
                        f"ig_evals={rec['ig_evals']} g2_evals={rec['g2_evals']} plan_ms={rec['plan_ms']:.1f}\n")
                for i, agent in enumerate(rec["agents"]):
                    f.write(f"  agent {i} action={agent['action']:<6} alt={agent['to'][2]:6.2f} "
                            f"| IG: {format_ig(agent['ig'])}\n")
                    if agent["own"]:
                        f.write(f"         planner's own scores: {format_ig(agent['own'])}\n")


def save_outputs_after_planner(case_name, name, runs, seeds, run0, runs_by_planner, out_dir):
    """Write this planner's log, run-0 path figure, and the cumulative tables and metrics plot."""
    paths0, belief0, gt0 = run0
    log_path = os.path.join(out_dir, f"planner_{name}.txt")
    save_planner_log(name, runs, seeds, log_path)
    print(f"  saved {os.path.normpath(log_path)}", flush=True)

    fig_path = os.path.join(out_dir, f"planner_path_{name}_run0.png")
    footprints = {name: [agent["fp"] for agent in runs[0][-1]["agents"]]}
    plot_viewer_style({name: paths0}, footprints, {name: belief0}, gt0, fig_path,
                      title=f"{name}: {CASES[case_name].label}, seed {seeds[0]}")
    print(f"  saved {os.path.normpath(fig_path)}", flush=True)

    save_step_table(runs_by_planner, os.path.join(out_dir, "planner_steps.csv"))
    save_compute_log(runs_by_planner, seeds, os.path.join(out_dir, "planner_compute.csv"))
    plot_metrics_comparison(runs_by_planner, os.path.join(out_dir, "planner_metrics.png"))
    if REFERENCE in runs_by_planner:
        save_summary_table(case_name, runs_by_planner, os.path.join(out_dir, "planner_summary.csv"))
    print(f"  updated tables and metrics plot in {os.path.normpath(out_dir)}", flush=True)


def run_case(case_name, seeds, out_dir, n_steps=N_STEPS, planners=PLANNERS, eval_budget=None, jobs=1):
    """Run every planner on the same seeds, greedy first as the reference."""
    planners = (REFERENCE, *[p for p in planners if p != REFERENCE])
    iterations = {}
    if eval_budget is not None:
        iterations = {p: calibrate_iterations(case_name, p, eval_budget) for p in planners if p != REFERENCE}
    runs_by_planner, run0_by_planner = {}, {}
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for name in planners:
            print(f"\nRunning {name} on {CASES[case_name].label} ({len(seeds)} runs)...", flush=True)
            tasks = [(case_name, name, seed, n_steps, iterations.get(name)) for seed in seeds]
            results = pool.map(_run_planner_task, tasks) if pool else map(_run_planner_task, tasks)
            runs = []
            for seed, (records, paths, belief, gt) in zip(seeds, results):
                runs.append(records)
                print(f"  seed {seed}: final mse={records[-1]['mse']:.3f} coverage={records[-1]['coverage']:.3f}",
                      flush=True)
                if name not in run0_by_planner:
                    run0_by_planner[name] = (paths, belief, gt)
            runs_by_planner[name] = runs
            save_outputs_after_planner(case_name, name, runs, seeds, run0_by_planner[name], runs_by_planner, out_dir)
    finally:
        if pool:
            pool.shutdown()

    print_summary(case_name, runs_by_planner, seeds, iterations)
    paths = {name: run0[0] for name, run0 in run0_by_planner.items()}
    footprints = {name: [a["fp"] for a in runs_by_planner[name][0][-1]["agents"]] for name in runs_by_planner}
    beliefs = {name: run0[1] for name, run0 in run0_by_planner.items()}
    plot_viewer_style(paths, footprints, beliefs, run0_by_planner[REFERENCE][2],
                      os.path.join(out_dir, "planner_viewer_plot.png"),
                      title=f"Planner paths, {CASES[case_name].label}, seed {seeds[0]}")
    return runs_by_planner


# ---------------------------------------------------------------------------
# Tests of the measurement itself (fast; the full comparison runs from the command line)
# ---------------------------------------------------------------------------


def test_planners_run_on_same_map(tmp_path):
    runs = run_case("case1", TUNE_SEEDS[:2], str(tmp_path), n_steps=3)
    assert set(runs) == set(PLANNERS)
    for name in ("planner_summary.csv", "planner_compute.csv", "planner_steps.csv", "planner_metrics.png"):
        assert (tmp_path / name).exists()


def test_seed_splits_are_disjoint():
    assert REPORT_SEEDS == tuple(range(42, 62))
    assert min(TUNE_SEEDS) >= 1000
    assert not set(TUNE_SEEDS) & set(REPORT_SEEDS)


@pytest.mark.parametrize("sweep", ["sweep.json", "sweep_dec.json"])
def test_sweeps_tune_on_separate_seeds(sweep):
    with open(os.path.join(ROOT, sweep)) as f:
        experiments = json.load(f)["experiments"]
    assert experiments
    for experiment in experiments:
        shared = experiment["overrides"]["shared"]
        first, last = shared["seed"] + shared["iters"][0], shared["seed"] + shared["iters"][1] - 1
        assert not set(range(first, last + 1)) & set(REPORT_SEEDS), experiment["name"]


@pytest.mark.parametrize("case_name", CASES)
def test_case_definition_reaches_the_pipeline(case_name, tmp_path):
    simulator = build_simulator(case_name, REFERENCE, TUNE_SEEDS[0], str(tmp_path))
    np.testing.assert_array_equal(
        simulator.ground_truth_map, gaussian_random_field(CLUSTER_RADIUS, GRID_SHAPE, TUNE_SEEDS[0])
    )
    assert len(simulator.agents) == CASES[case_name].num_agents
    coordinator = simulator.coordinator
    if case_name == "n1":
        assert coordinator is None
        return
    corners = {(x, y) for x in (-25.0, 25.0) for y in (-25.0, 25.0)}
    assert {tuple(a["uav_pos"].position) for a in simulator.agents} == corners
    if case_name == "case1":
        assert (coordinator.communication_range, coordinator.news_mode, coordinator.mode) == (15.625, "BM", "IGd_BM")
    else:
        assert (coordinator.communication_range, coordinator.news_mode, coordinator.mode) == (-1, "BS", "IG_BS")


def test_belief_comes_from_the_pipeline_mapper(monkeypatch, tmp_path):
    calls = []
    original_og = OccupancyMap.update_belief_OG
    original_lbp = OccupancyMap.propagate_messages

    def spy_og(self, *args, **kwargs):
        calls.append("OG")
        return original_og(self, *args, **kwargs)

    def spy_lbp(self, *args, **kwargs):
        calls.append(("LBP", kwargs.get("max_iterations"), kwargs.get("reset_msgs")))
        return original_lbp(self, *args, **kwargs)

    monkeypatch.setattr(OccupancyMap, "update_belief_OG", spy_og)
    monkeypatch.setattr(OccupancyMap, "propagate_messages", spy_lbp)
    build_simulator("n1", REFERENCE, TUNE_SEEDS[0], str(tmp_path), n_steps=2).run_measured()
    assert calls == ["OG", ("LBP", 1, True)] * 2


def _read_strategy_json(strategy):
    with open(os.path.join(ROOT, STRATEGY_CONFIGS[strategy])) as f:
        return json.load(f)


@pytest.mark.parametrize("strategy", ["mh_dec_mcts_full", "mh_dec_mcts_efficient"])
def test_mh_json_settings_reach_the_planner(strategy, tmp_path):
    settings = _read_strategy_json(strategy)["hierarchical_dec_mcts"]
    simulator = build_simulator("n1", strategy, TUNE_SEEDS[0], str(tmp_path), n_steps=1)
    simulator.run_measured()
    planner = simulator.agents[0]["planner"]._hierarchical_planner
    llp, hlp = planner.llp, planner.hlp
    assert llp.use_g2 is settings["use_g2"] is True
    assert llp.use_mcts_llp is settings["use_mcts_llp"]
    assert (llp.horizon, llp.num_iterations) == (settings["llp"]["horizon"], settings["llp"]["iterations"])
    assert llp.discount == settings["llp"]["discount_factor"]
    assert (hlp.horizon, hlp.num_iterations) == (settings["hlp"]["horizon"], settings["hlp"]["iterations"])
    assert hlp.replan_interval == settings["hlp"]["replan_interval"]
    assert list(hlp.tile_size) == settings["hlp"]["tile_size"]


def test_dec_mcts_json_settings_reach_the_planner(tmp_path):
    settings = _read_strategy_json("dec_mcts")
    simulator = build_simulator("n1", "dec_mcts", TUNE_SEEDS[0], str(tmp_path), n_steps=1)
    simulator.run_measured()
    config = simulator.agents[0]["planner"]._dec_mcts_planner.config
    for key in ("horizon", "iterations", "ucb_c", "discount_factor", "timeout"):
        assert config[key] == settings["dec_mcts"][key], key
    assert config["overlap_penalty_weight"] == settings["decentralized"]["overlap_penalty_weight"]


@pytest.mark.parametrize("strategy,planner_class,attribute", [
    ("dec_mcts", DecMCTSPlanner, "_dec_mcts_planner"),
    ("mh_dec_mcts_full", HierarchicalDecMCTSPlanner, "_hierarchical_planner"),
    ("mh_dec_mcts_efficient", HierarchicalDecMCTSPlanner, "_hierarchical_planner"),
])
def test_planner_is_built_once_and_its_state_persists(strategy, planner_class, attribute, monkeypatch, tmp_path):
    built = []
    original_init = planner_class.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(planner_class, "__init__", counting_init)
    simulator = build_simulator("case2", strategy, TUNE_SEEDS[0], str(tmp_path), n_steps=2)

    def rng_states():
        states = []
        for agent in simulator.agents:
            planner = getattr(agent["planner"], attribute)
            rng = planner._rng if strategy == "dec_mcts" else planner.llp._rng
            states.append(rng.bit_generator.state["state"]["state"])
        return states

    simulator._select_agent_actions(0)
    first = rng_states()
    planners = [getattr(agent["planner"], attribute) for agent in simulator.agents]
    simulator._update_agent_positions()
    simulator._perform_belief_fusion(simulator._process_agent_observations(), 0)
    simulator._select_agent_actions(1)

    assert len(built) == len(simulator.agents)
    assert [getattr(agent["planner"], attribute) for agent in simulator.agents] == planners
    assert all(before != after for before, after in zip(first, rng_states())), "RNG restarted between steps"
    if strategy != "dec_mcts":
        bus = planners[0].intent_bus
        assert all(planner.intent_bus is bus for planner in planners), "agents must share one intent bus"
        assert len(bus.get_all_ll_intents()) == len(planners)
        assert all(planner.hlp.current_intent is not None for planner in planners)


def test_metrics_on_known_curves():
    records = [{"mse": m, "coverage": c} for m, c in [(0.2, 0.5), (0.04, 0.9), (0.04, 1.0), (0.02, 1.0)]]
    metrics = run_metrics(records)
    assert metrics["mse_area"] == pytest.approx(0.12 + 0.04 + 0.03)
    assert metrics["steps_mse_05"] == 2
    assert metrics["steps_mse_01"] is None
    assert metrics["steps_full_cov"] == 3
    values, reached = metric_values([records, records], "steps_mse_01")
    assert list(values) == [4.0, 4.0] and reached == 0
    diff, se = paired_difference([3.0, 5.0], [1.0, 1.0])
    assert diff == 3.0 and se == pytest.approx(1.0)


@pytest.mark.parametrize("case_name", ["n1", "case1"])
def test_every_planner_is_scored_with_the_same_formula(case_name, tmp_path):
    for strategy in PLANNERS:
        simulator = build_simulator(case_name, strategy, TUNE_SEEDS[0], str(tmp_path), n_steps=2)
        records = simulator.run_measured()
        expected = np.mean((simulator.ground_truth_map - team_belief(simulator.agents)) ** 2)
        assert records[-1]["mse"] == pytest.approx(expected), strategy


def test_reward_evaluations_are_counted(tmp_path):
    greedy = run_planner("n1", REFERENCE, TUNE_SEEDS[0], n_steps=3)[0]
    assert all(1 <= rec["ig_evals"] <= len(ALLOWED_ACTIONS) and rec["g2_evals"] == 0 for rec in greedy)
    settings = _read_strategy_json("mh_dec_mcts_efficient")["hierarchical_dec_mcts"]["llp"]
    efficient = run_planner("n1", "mh_dec_mcts_efficient", TUNE_SEEDS[0], n_steps=3)[0]
    # A rollout can stop early, because the LLP's admissible-action cache can disagree with its
    # exact altitude check, so this is an upper bound rather than an equality.
    budget = settings["horizon"] * settings["iterations"]
    assert all(0.9 * budget <= rec["ig_evals"] <= budget for rec in efficient)
    assert all(rec["g2_evals"] >= settings["iterations"] for rec in efficient)
    dec_mcts = run_planner("n1", "dec_mcts", TUNE_SEEDS[0], n_steps=3)[0]
    assert all(rec["ig_evals"] > 0 and rec["g2_evals"] == 0 and rec["plan_ms"] > 0 for rec in dec_mcts)


def test_eval_budget_brings_planners_to_equal_evaluations():
    target = 200
    for strategy in STRATEGY_CONFIGS:
        iterations = calibrate_iterations("n1", strategy, target, steps=12)
        records = run_planner("n1", strategy, TUNE_SEEDS[0], n_steps=12, iterations=iterations)[0]
        assert evaluations_per_decision([records]) == pytest.approx(target, rel=0.25), strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--split", choices=SEED_SPLITS, default="tune")
    parser.add_argument("--iters", type=int, default=N_ITERS, help="number of runs per planner")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=list(PLANNERS))
    parser.add_argument("--eval-budget", type=int, help="reward evaluations per decision for the MCTS planners")
    parser.add_argument("--jobs", type=int, default=1, help="parallel runs; wall time and MH counts shift under load")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    seeds = SEED_SPLITS[args.split][: args.iters]
    if len(seeds) < args.iters:
        parser.error(f"split {args.split} has only {len(seeds)} seeds")
    for case_name in CASES if args.case == "all" else [args.case]:
        suffix = f"_e{args.eval_budget}" if args.eval_budget is not None else ""
        out_dir = os.path.join(args.out, args.split, case_name + suffix)
        run_case(case_name, seeds, out_dir, args.steps, tuple(args.planners), args.eval_budget, args.jobs)
    print("✓ planner smoke test passed")


if __name__ == "__main__":
    main()

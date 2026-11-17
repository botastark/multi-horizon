"""
Collect per-run metrics from experiment folders and summarise them per strategy.

Each run folder (experiments/runs/<strategy>/run_<timestamp>_<run_base>/) holds
  txt/run.log         one block per iteration ("Iteration: N" + step table:
                      Step, Entropy, MSE, Coverage, Heights, ...)
  txt/timestamps.csv  planning start/end per (run_number, step, agent)

Row k of the step table is the state after k steps (row 0 = prior). Coverage is
the fraction of cells observed at least once by any agent.

Per run:
  steps_to_full_coverage   first step with coverage >= 99.99 % (NaN if never)
  coverage_at_25/50        coverage after 25 / 50 steps
  final_coverage, final_mse, final_entropy   last logged step
  mean_mse, mean_entropy   averaged over all logged steps (area under the curve / n)
  mean_altitude            mean altitude over agents and steps
  plan_ms_mean / plan_ms_p95   planning time per agent-step

Outputs (in --out): per_run_metrics.csv, summary.csv, summary.md,
curves.png (mean +- std over runs) and steps_to_full_coverage.png.

Usage:
  python collect_metrics.py --runs-root experiments/runs \\
      --match ortomap_rorto_sample_N4_IG_BS_commRinf --out experiments/summaries/orthomap_ig_bs_n4
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np

STRATEGY_ORDER = ["greedy_ig", "dec_mcts", "mh_dec_mcts_full", "mh_dec_mcts_efficient"]
STRATEGY_LABEL = {
    "greedy_ig": "Greedy IG",
    "dec_mcts": "Dec-MCTS + g2",
    "mh_dec_mcts_full": "MH-Dec-MCTS (full)",
    "mh_dec_mcts_efficient": "MH-Dec-MCTS (efficient)",
}
# Categorical slots 1-4 of the reference palette, fixed order per strategy
STRATEGY_COLOR = {
    "greedy_ig": "#2a78d6",
    "dec_mcts": "#eb6834",
    "mh_dec_mcts_full": "#1baf7a",
    "mh_dec_mcts_efficient": "#eda100",
}
FULL = 0.9999
ROW = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[([^\]]*)\]")


def parse_run_log(path):
    """Return {iteration: {"step": [...], "entropy": [...], "mse": [...], "coverage": [...], "alt": [...]}}."""
    runs, current = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("Iteration:"):
                current = int(line.split(":")[1])
                runs[current] = defaultdict(list)
                continue
            m = ROW.match(line)
            if m and current is not None:
                step, ent, mse, cov, heights = m.groups()
                run = runs[current]
                run["step"].append(int(step))
                run["entropy"].append(float(ent))
                run["mse"].append(float(mse))
                run["coverage"].append(float(cov))
                alts = [float(h) for h in heights.split(",") if h.strip()]
                run["alt"].append(float(np.mean(alts)) if alts else math.nan)
    return runs


def parse_timestamps(path):
    """Return {run_number: [duration_ms, ...]} (one entry per agent-step)."""
    out = defaultdict(list)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["run_number"])].append(float(r["end_ms"]) - float(r["start_ms"]))
            except (KeyError, ValueError):
                continue
    return out


def run_metrics(run, durations):
    cov = np.asarray(run["coverage"])
    steps = np.asarray(run["step"])
    reached = np.flatnonzero(cov >= FULL)

    def at(k):
        idx = np.flatnonzero(steps == k)
        return float(cov[idx[0]]) if idx.size else math.nan

    d = np.asarray(durations) if durations else np.array([math.nan])
    return {
        "n_steps_logged": int(len(steps)),
        "steps_to_full_coverage": float(steps[reached[0]]) if reached.size else math.nan,
        "coverage_at_25": at(25),
        "coverage_at_50": at(50),
        "final_coverage": float(cov[-1]),
        "final_mse": float(run["mse"][-1]),
        "final_entropy": float(run["entropy"][-1]),
        "mean_mse": float(np.mean(run["mse"])),
        "mean_entropy": float(np.mean(run["entropy"])),
        "mean_altitude": float(np.nanmean(run["alt"])),
        "plan_ms_mean": float(np.nanmean(d)),
        "plan_ms_p95": float(np.nanpercentile(d, 95)),
    }


def collect(runs_root, match):
    rows, curves = [], defaultdict(list)
    for log in sorted(glob.glob(os.path.join(runs_root, "*", "*", "txt", "run.log"))):
        run_dir = os.path.dirname(os.path.dirname(log))
        if match and match not in os.path.basename(run_dir):
            continue
        strategy = os.path.basename(os.path.dirname(run_dir))
        durations = parse_timestamps(os.path.join(run_dir, "txt", "timestamps.csv"))
        for it, run in sorted(parse_run_log(log).items()):
            if not run["step"]:
                continue
            m = run_metrics(run, durations.get(it, []))
            rows.append({"strategy": strategy, "iteration": it, "run_dir": os.path.basename(run_dir), **m})
            curves[strategy].append(run)
    return rows, curves


METRICS = [
    ("steps_to_full_coverage", "Step at 100 % coverage", "{:.1f}"),
    ("coverage_at_25", "Coverage @25", "{:.3f}"),
    ("coverage_at_50", "Coverage @50", "{:.3f}"),
    ("final_mse", "Final MSE", "{:.4f}"),
    ("mean_mse", "Mean MSE (all steps)", "{:.4f}"),
    ("final_entropy", "Final entropy", "{:.0f}"),
    ("mean_entropy", "Mean entropy", "{:.0f}"),
    ("mean_altitude", "Mean altitude (m)", "{:.1f}"),
    ("plan_ms_mean", "Planning ms / agent-step", "{:.0f}"),
    ("plan_ms_p95", "Planning ms p95", "{:.0f}"),
]


def summarise(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)
    order = [s for s in STRATEGY_ORDER if s in by] + sorted(s for s in by if s not in STRATEGY_ORDER)
    summary = []
    for s in order:
        rs = by[s]
        entry = {"strategy": s, "runs": len(rs),
                 "runs_reaching_full": int(sum(not math.isnan(r["steps_to_full_coverage"]) for r in rs))}
        for key, _, _ in METRICS:
            vals = np.array([r[key] for r in rs], dtype=float)
            vals = vals[~np.isnan(vals)]
            entry[f"{key}_mean"] = float(vals.mean()) if vals.size else math.nan
            entry[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else math.nan
            entry[f"{key}_median"] = float(np.median(vals)) if vals.size else math.nan
        summary.append(entry)
    return summary


def write_outputs(rows, summary, curves, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_run_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    header = ["Metric (mean ± std over runs)"] + [STRATEGY_LABEL.get(s["strategy"], s["strategy"]) for s in summary]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.append("| Runs (reaching 100 % coverage) | " + " | ".join(
        f"{s['runs']} ({s['runs_reaching_full']})" for s in summary) + " |")
    for key, label, fmt in METRICS:
        cells = []
        for s in summary:
            m, sd = s[f"{key}_mean"], s[f"{key}_std"]
            cells.append("–" if math.isnan(m) else fmt.format(m) + ("" if math.isnan(sd) else " ± " + fmt.format(sd)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    _plot_curves(curves, os.path.join(out, "curves.png"))
    _plot_steps_to_full(rows, os.path.join(out, "steps_to_full_coverage.png"))
    return "\n".join(lines)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#5f5e5a", labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_curves(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("coverage", "Coverage (observed at least once)"), ("mse", "MSE"), ("entropy", "Entropy (bits)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    order = [s for s in STRATEGY_ORDER if s in curves] + sorted(s for s in curves if s not in STRATEGY_ORDER)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        _style(ax)
        for s in order:
            n = min(len(r["step"]) for r in curves[s])
            data = np.array([r[key][:n] for r in curves[s]])
            x = np.asarray(curves[s][0]["step"][:n])
            mean, std = data.mean(0), data.std(0)
            color = STRATEGY_COLOR.get(s, "#6250d6")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, label=STRATEGY_LABEL.get(s, s))
        ax.set_title(title, fontsize=10, color="#1f1e1c", loc="left")
        ax.set_xlabel("Step", fontsize=9, color="#5f5e5a")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Mean ± 1 std over runs", fontsize=9, color="#5f5e5a", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_steps_to_full(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r["steps_to_full_coverage"])
    order = [s for s in STRATEGY_ORDER if s in by] + sorted(s for s in by if s not in STRATEGY_ORDER)
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style(ax)
    rng = np.random.default_rng(0)
    for i, s in enumerate(order):
        vals = np.array(by[s], dtype=float)
        ok = vals[~np.isnan(vals)]
        color = STRATEGY_COLOR.get(s, "#6250d6")
        ax.scatter(np.full(ok.size, i) + rng.uniform(-0.12, 0.12, ok.size), ok, s=28,
                   color=color, edgecolor="#fcfcfb", linewidth=1, zorder=3)
        if ok.size:
            ax.plot([i - 0.25, i + 0.25], [ok.mean()] * 2, color="#1f1e1c", linewidth=2, zorder=4)
            ax.text(i + 0.28, ok.mean(), f"{ok.mean():.1f}", va="center", fontsize=8, color="#1f1e1c")
        missing = int(np.isnan(vals).sum())
        if missing:
            ax.text(i, ax.get_ylim()[1] if ok.size else 0, f"{missing} never", ha="center",
                    va="bottom", fontsize=8, color="#5f5e5a")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([STRATEGY_LABEL.get(s, s) for s in order], fontsize=8, color="#1f1e1c")
    ax.set_ylabel("Step at which 100 % of cells were observed", fontsize=8, color="#5f5e5a")
    ax.set_title("Steps to full coverage (dots = runs, bar = mean)", fontsize=10, color="#1f1e1c", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="experiments/runs")
    ap.add_argument("--match", default="", help="substring the run folder name must contain")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, curves = collect(args.runs_root, args.match)
    if not rows:
        raise SystemExit(f"no runs found under {args.runs_root} matching '{args.match}'")
    summary = summarise(rows)
    print(write_outputs(rows, summary, curves, args.out))
    print(f"\nwrote {args.out}/per_run_metrics.csv, summary.csv, summary.md, curves.png, steps_to_full_coverage.png")


if __name__ == "__main__":
    main()

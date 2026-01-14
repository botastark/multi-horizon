# multi-horizon — Run & Plot (MH-Dec-MCTS primary)

This repository contains several planning strategies. The primary method is **MH-Dec-MCTS** (Multi-Horizon Dec-MCTS). Use the other two strategies as benchmarks for comparison: **Dec-MCTS** (single-level) and **Greedy IG** (baseline).

Prerequisites
- Activate your Python environment and install requirements:

```bash
conda activate active_sensing
pip install -r requirements.txt
```

VS Code users: the workspace config will auto-activate the `active_sensing` Conda environment
for new integrated terminals. If you prefer a manual step, run `conda activate active_sensing`.

Main method — MH-Dec-MCTS (recommended)
- Run the main hierarchical planner:

```bash
python src/main.py --config configs/benchmark_mh_dec_mcts.json
```
- Plot results:

```bash
python plotter.py trials/mh_dec_mcts_gaussian_*_N4_*/txt/ --radius 5
```
- Docs: [MH-Dec-MCTS Benchmark](docs/mh_dec_mcts_multiagent.md)

Benchmark — Dec-MCTS (single-level)
- Run:

```bash
python src/main.py --config configs/benchmark_dec_mcts.json
```
- Plot results:

```bash
python plotter.py trials/dec_mcts_gaussian_*_N4_*/txt/ --radius 5
```
- Docs: [Dec-MCTS Benchmark](docs/dec_mcts_multiagent.md)

Benchmark — Greedy IG (baseline)
- Run (all 6 news modes: IG, IGd, IG_BS, IG_BM, IGd_BS, IGd_BM):

```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```
- Plot results (example for specific mode):

```bash
python plotter.py trials/greedy_ig_gaussian_r5_corner_N4_IG_commR5*/txt/ --radius 5
```
- Docs: [Greedy IG Benchmark](docs/greedy_ig_multiagent.md)

**Available Modes for All Strategies:**
All three strategies (Greedy IG, Dec-MCTS, and MH-Dec-MCTS) support the same 6 modes configured via `mode_labels`:
- `IG`: No information sharing (baseline)
- `IGd`: Position sharing only (with footprint IoU discounting)
- `IG_BS`: IG + broadcast single news (all neighbors get same news)
- `IG_BM`: IG + per-neighbor news (each neighbor gets private news)
- `IGd_BS`: Position sharing + broadcast single news
- `IGd_BM`: Position sharing + per-neighbor news

The config files run all modes automatically. Results are stored in separate trial folders with mode suffix in folder name.

Notes
- `plotter.py` expects `txt/` trial outputs produced by `src/main.py`.
- Use `--paths` and `--compare-news` with `plotter.py` for multi-run comparisons (useful for comparing different news modes).
- Each config file specifies `mode_labels` to run multiple modes in sequence. Results are stored in separate trial folders.
- Communication range is calculated dynamically based on `radius_multiplier` (for Greedy IG) or set explicitly (for MCTS methods).

For algorithmic details, reward decomposition, and implementation notes see the benchmark docs linked above.

Plotter: Compare Methods
- Use `--compare-methods` to compare Dec-MCTS, MH-Dec-MCTS and Greedy IG trial outputs in a single figure. The tool auto-discovers matching `trials/` folders, validates common metadata (e.g. `NumAgents`, `communication_range`), and writes a 2×2 panel (Entropy / Height / MSE / Coverage) to `plots/method_comparison_r_<radius>[_pairwise_<pairwise>][_N<num_agents>].png`.

Examples:

```bash
# basic (auto-detects available methods)
python plotter.py --compare-methods --radius 5

# filter by pairwise correlation and select agent count
python plotter.py --compare-methods --radius 5 --pairwise adaptive --num-agents 4

# compare specific news mode across methods
python plotter.py --compare-methods --radius 5 --pairwise adaptive --num-agents 4 --news-mode IG_BM
```

Notes:
- `--pairwise` filters trials by pairwise correlation type (e.g. `adaptive`, `equal`, `biased`).
- `--num-agents` ensures the comparison uses trials with the same number of agents when multiple counts exist.
- `--news-mode` filters comparisons to a single mapping/news mode (e.g. `IG`, `IGd`, `IG_BM`, `IGd_BM`). When multiple `NewsMode` values are present across trial folders, `plotter.py --compare-methods` will require `--news-mode` to be specified.
- If trial folders lack explicit communication metadata, the plotter attempts to infer `communication_range` from folder names; verify results when in doubt.

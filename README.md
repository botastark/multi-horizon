# Multi-Horizon Active Sensing

Multi-agent active sensing system with decentralized planning and information sharing.

## Quick Start

### Running Benchmarks

#### Greedy IG Benchmark

Run the greedy information gain benchmark with various news modes (IG, IG+BS, IG+BM, IGd, IGd+BS, IGd+BM):

```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```

This will run experiments for all configured news modes and save results to `trials/greedy_ig_gaussian_r5_corner_N8_<NEWS_MODE>_commR<MULTIPLIER>/`.

**Communication Range Configuration:**

The communication range is controlled by `radius_multiplier` in the config file:
- `radius_multiplier: 5` → 5 × h_displacement = 5 × 3.125m = 15.625m
- `radius_multiplier: -1` → Unlimited range

where `h_displacement = (field_len/2) / n_h_act` and `n_h_act = 8` for 8 agents, otherwise 5.

### Plotting Results

#### Plot Greedy IG with Limited Range (R=5)

```bash
python plotter.py --compare-news --paths trials/*_commR5/txt/ --radius 5 --comm-range 5
```

This generates `plots/greedy_ig_info_sharing_comparison_r5_commR5_N8.png` showing:
- Left column: IG, IG+BS, IG+BM
- Right column: IGd, IGd+BS, IGd+BM
- Title: "Greedy Ig - Effect of information sharing... (Comm Range = 5×3.125m = 15.6m)"

#### Plot Greedy IG with Unlimited Range

```bash
python plotter.py --compare-news --paths trials/*_commRinf/txt/ --radius 5 --comm-range -1
```

This generates `plots/greedy_ig_info_sharing_comparison_r5_commRinf_N8.png` with unlimited communication range.

#### Compare Both Communication Ranges

Generate both plots to compare limited vs unlimited range:

```bash
# Limited range (15.625m)
python plotter.py --compare-news --paths trials/*_commR5/txt/ --radius 5 --comm-range 5

# Unlimited range
python plotter.py --compare-news --paths trials/*_commRinf/txt/ --radius 5 --comm-range -1
```

## Configuration

- **Config files**: `configs/benchmark_greedy_ig.json`
- **Results directory**: `trials/`
- **Plots directory**: `plots/`

## Strategy Comparison

| Aspect | Greedy IG | Dec-MCTS | MH-Dec-MCTS |
|--------|-----------|----------|-------------|
| **Planning levels** | 1 | 1 | 2 (LLP + HLP) |
| **Lookahead** | 1 step | 10 steps | 3-7 steps + 3-10 regions |
| **Search method** | Enumerate | MCTS | MCTS (both levels) |
| **Agents** | 8 | 4 | 4 |
| **Comm range** | 15.625m | 15.0m | 150.0m |
| **What's shared** | Footprint + news | LL trajectory | LL + HL intents |
| **Coordination** | Overlap avoidance | Trajectory penalty | g₂-based coupling |
| **Reward** | IG only | IG + overlap | g₁(IG) + g₂(time) |
# multi-horizon — Run & Plot Quick Reference

Minimal instructions to run each benchmark and plot results. For detailed design and algorithm notes see the linked docs for each benchmark.

Prerequisites
- Activate your Python environment (example):

```bash
conda activate active_sensing
pip install -r requirements.txt
```

Greedy IG (single-step IG baseline)
- Run:

```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```
- Plot results (example):

```bash
python plotter.py trials/greedy_ig_gaussian_r5_corner_N8_IG*/txt/ --radius 5
```
- Docs: [Greedy IG Benchmark](docs/greedy_ig_multiagent.md)

Decentralized MCTS (single-level Dec-MCTS)
- Run:

```bash
python src/main.py --config configs/benchmark_dec_mcts.json
```
- Plot results (example):

```bash
python plotter.py trials/dec_mcts_gaussian_*_N4_*/txt/ --radius 5
```
- Docs: [Dec-MCTS Benchmark](docs/dec_mcts_multiagent.md)

MH-Dec-MCTS (Multi-Horizon hierarchical planner)
- Run:

```bash
python src/main.py --config configs/benchmark_mh_dec_mcts.json
```
- Plot results (example):

```bash
python plotter.py trials/mh_dec_mcts_gaussian_*_N4_*/txt/ --radius 5
```
- Docs: [MH-Dec-MCTS Benchmark](docs/mh_dec_mcts_multiagent.md)

Notes
- The `plotter.py` script expects trial `txt/` output directories created by `src/main.py`.
- `--paths` and `--compare-news` in `plotter.py` are used for multi-run/news-mode comparisons (mostly for Greedy IG).

That's it — these commands are all you need to run experiments and produce plots. See the linked docs for implementation details and algorithmic descriptions.
**Status:** ✅ Paper-correct implementation (Seiler et al., 2024)

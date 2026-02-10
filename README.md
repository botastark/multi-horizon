# multi-horizon — Multi-Agent Coverage with Hierarchical Planning

This repository implements **4 baseline planning strategies** for multi-agent coverage experiments. The primary method is **MH-Dec-MCTS** (Multi-Horizon Decentralized MCTS), a hierarchical planner with strategic region allocation (HLP) and tactical action selection (LLP).

## Quick Start

### Prerequisites
Activate your Python environment and install requirements:

```bash
conda activate active_sensing
pip install -r requirements.txt
```

VS Code users: the workspace config will auto-activate the `active_sensing` Conda environment
for new integrated terminals.

### Run All 4 Baselines (Recommended)

Run all strategies in sequence using the master config:

```bash
# From project root (NOT from src/)
python src/main.py --config configs/master_config.json
```

This will run:
1. **Greedy IG** - Single-step lookahead baseline
2. **Dec-MCTS** - Single-level MCTS planning
3. **MH-Dec-MCTS (full)** - Hierarchical with both planners using MCTS
4. **MH-Dec-MCTS (efficient)** - Hierarchical with optimized LLP (random rollout)

### Run Single Strategy

To run a single strategy, edit `configs/master_config.json` and set:
```json
"strategies": ["mh_dec_mcts_full"]
```

Then run:
```bash
python src/main.py --config configs/master_config.json
```

⚠️ **Note:** Individual strategy configs (e.g., `configs/strategies/mh_dec_mcts_full.json`) cannot be run directly—they are templates loaded by the master config.

### Example Runner

```bash
python run_benchmark.py  # Edit file to select strategies
```

## The 4 Baselines

| # | Strategy | LLP | HLP | Description |
|---|----------|-----|-----|-------------|
| 1 | **Greedy IG** | N/A | N/A | Reactive baseline, single-step IG maximization |
| 2 | **Dec-MCTS** | N/A | N/A | Single-level MCTS with 10-step planning horizon |
| 3 | **MH-Dec-MCTS (full)** | MCTS | MCTS | Both planners use UCB tree search |
| 4 | **MH-Dec-MCTS (efficient)** | Random | MCTS | LLP optimized with random rollout |

**Incremental progression:**
- 1 → 2: Add multi-step planning
- 2 → 3: Add hierarchical structure (both levels use MCTS)
- 3 → 4: Optimize LLP (test if HLP guidance makes LLP tree search unnecessary)

## Configuration System

The project uses a **master config + strategy-specific configs** architecture:

```
configs/
├── master_config.json          # Shared settings + strategy selection
├── strategies/                 # Strategy-specific parameters
│   ├── greedy_ig.json
│   ├── dec_mcts.json
│   ├── mh_dec_mcts_full.json
│   └── mh_dec_mcts_efficient.json
└── [other configs]            # Additional test/utility configs
```


📖 **See**: [Configuration Guide](configs/README.md) for complete details.

## Plotting Results

### Main Method — MH-Dec-MCTS

```bash
# Plot MH-Dec-MCTS full results
python plotter.py trials/mh_dec_mcts_both_gaussian_*_N4_*/txt/ --radius 4

# Plot MH-Dec-MCTS efficient results
python plotter.py trials/mh_dec_mcts_gaussian_*_N4_*/txt/ --radius 4
```

### Benchmark Methods

```bash
# Dec-MCTS
python plotter.py trials/dec_mcts_gaussian_*_N4_*/txt/ --radius 4

# Greedy IG
python plotter.py trials/greedy_ig_gaussian_*_N4_IG_*/txt/ --radius 4
```

### Compare All Methods

```bash
# Basic comparison
python plotter.py --compare-methods --radius 4

# Filter by pairwise correlation and agent count
python plotter.py --compare-methods --radius 4 --pairwise adaptive --num-agents 4

# Compare specific communication mode
python plotter.py --compare-methods --radius 4 --news-mode IG_BM
```

### Planning Time Comparison

Generate planning time comparison plots showing computational efficiency across methods:

```bash
# Open analysis.ipynb in Jupyter and run the planning time analysis cells
jupyter notebook analysis.ipynb

# The notebook will generate plots comparing:
# - Average Total Time per Run (total planning time across all steps)
# - Average Time per Step (planning time per decision cycle)
# 
# Plots are saved to plots/ directory
```

**Communication Modes:**
All strategies support 6 modes (configured in `master_config.json` under `mode_labels`):
- `IG`: Independent IG planner (no position or belief sharing)
- `IGd`: IG planner with position-based overlap discounting
- `IG_BS`: IG planner with broadcast belief ("news") sharing
- `IG_BM`: IG planner with per-neighbor belief ("news") sharing
- `IGd_BS`: Discounted IG planner with broadcast belief sharing
- `IGd_BM`: Discounted IG planner with per-neighbor belief sharing

## Documentation

- 📋 [Configuration Guide](configs/README.md) - Complete config system documentation
- 📈 [Method Comparison](docs/method_comparison_summary.md) - 4 baseline comparison
- 🎯 [MH-Dec-MCTS Architecture](docs/mh_dec_mcts_multiagent.md) - Hierarchical planner details

## Project Structure

```
multi-horizon/
├── configs/                   # Configuration files
│   ├── master_config.json    # Master config (v3.0)
│   ├── strategies/           # Strategy-specific configs
├── src/                      # Source code
│   ├── main.py              # Main entry point
│   ├── config_loader.py     # Multi-strategy config loader
│   ├── hierarchical_dec_mcts.py  # MH-Dec-MCTS implementation
│   ├── dec_mcts.py          # Dec-MCTS implementation
│   └── ...
├── docs/                     # Documentation
├── trials/                   # Experiment results
├── plots/                    # Generated plots
├── run_benchmark.py         # Example runner script
└── plotter.py               # Visualization tool
```
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

### Web Interface (Optional)

Launch the interactive web interface for easy parameter tuning and experiment management:

```bash
cd web_interface
./start_web_interface.sh
# Open http://localhost:5000 in your browser
```

**Features:**
- 🎛️ Interactive parameter controls for all methods
- 📊 Real-time progress monitoring and plotting
- 📁 Automatic experiment organization in `experiments/` directory
- 🔄 Multi-run comparison (overlay multiple experiments)
- ⚙️ Optional debug logs (disabled by default for cleaner output)

### Batch Experiments (Command Line)

For running multiple parameter combinations via command line:

```bash
# Edit sweep.json to configure parameter grid
./sweep.sh
```

**All methods use the same directory structure:**
- Experiments save to `experiments/runs/<method>/run_<timestamp>_*/`
- Each run contains: `txt/` (results), `plots/` (visualizations), `config.json`
- Web interface also creates `metadata.json` and moves failed runs to `experiments/failed/`

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

### Quick Start

```bash
# Plot specific method
python plotter.py experiments/runs/dec_mcts/*/txt/ --radius 4

# Compare all methods (auto-discovers all experiments)
python plotter.py --compare-methods --radius 4
```

### Method-Specific Plots

**MH-Dec-MCTS (Full and Efficient):**
```bash
python plotter.py experiments/runs/mh_dec_mcts_full/*/txt/ --radius 4
python plotter.py experiments/runs/mh_dec_mcts_efficient/*/txt/ --radius 4
```

**Benchmark Methods:**
```bash
python plotter.py experiments/runs/dec_mcts/*/txt/ --radius 4
python plotter.py experiments/runs/greedy_ig/*/txt/ --radius 4
```

### Compare All Methods

```bash
# Basic comparison
python plotter.py --compare-methods --radius 4

# Filter by communication mode
python plotter.py --compare-methods --radius 4 --news-mode IGd_BM

# Filter by agent count
python plotter.py --compare-methods --radius 4 --num-agents 4
```

### Planning Time Comparison

Generate planning time efficiency comparison plots across all methods:

```bash
# Generate timing comparison (mean planning time per step)
python plotter.py --compare-timing

# Use convenience script
./plot_timing.sh

# Show plot interactively
./plot_timing.sh --show

# Compare P95 latency instead of mean
python plotter.py --compare-timing --timing-metric P95_ms

# Filter by communication mode
python plotter.py --compare-timing --news-mode IGd_BM
```

**Available Metrics:**
- `Mean_ms`: Average planning time per step (default)
- `Median_ms`: Median planning time (robust to outliers)
- `P95_ms`: 95th percentile latency (captures worst-case performance)
- `P99_ms`: 99th percentile latency (extreme cases)

**Output:** Generates two plots side-by-side:
1. **Evolution Plot**: Planning time across all steps (with confidence intervals)
2. **Distribution Plot**: Box plots showing overall timing distributions per method

All timing data is automatically collected from `timestamps.csv` files in each experiment run.

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
├── web_interface/            # Optional web interface
│   ├── web_interface.py     # Flask backend
│   ├── templates/           # HTML templates
│   ├── start_web_interface.sh  # Launch script
│   └── web_requirements.txt # Web dependencies
├── docs/                     # Documentation
├── experiments/              # Experiment results
│   ├── temp/                # In-progress experiments
│   ├── runs/                # Completed experiments by method
│   │   ├── greedy_ig/       # run_<timestamp>_*/
│   │   ├── dec_mcts/
│   │   ├── mh_dec_mcts_efficient/
│   │   └── mh_dec_mcts_full/
│   └── failed/              # Failed/stopped experiments
├── plots/                    # Generated comparison plots
├── sweep.sh                  # Batch experiment runner
├── sweep.json               # Parameter grid config
├── run_benchmark.py         # Example runner script
└── plotter.py               # Visualization tool
```
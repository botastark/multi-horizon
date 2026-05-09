# Web Interface for Multi-Horizon Experiments

This directory contains the web-based **Parameter Tuning Dashboard** for multi-horizon experiments.

## Quick Start

```bash
cd web_interface
./start_web_interface.sh
# Open http://localhost:5000 in your browser
```

## Features

- 📊 **Results Dashboard**: Browse and compare all experiments (ongoing and completed) in one view
- 🔍 **Filter & Search**: Filter by method and communication mode to find optimal parameters
- 📈 **Real-time Monitoring**: View results from CLI-started experiments as they run
- 🎛️ **Interactive Controls**: Easy parameter tuning for new experiments
- 📂 **Browse CLI Runs**: View and visualize ongoing experiments started from command line
- 🔄 **Auto-refresh**: Dashboard updates every 10 seconds with latest results
- 📁 **Auto Organization**: Experiments automatically saved to `experiments/runs/<method>/`
- ⚙️ **Optional Debug Logs**: Enable/disable detailed logging (off by default)

## Directory Structure

The web interface operates from this subdirectory but manages experiments in the project root:

```
multi-horizon/                    # Project root
├── web_interface/               # This directory
│   ├── web_interface.py        # Flask backend
│   ├── templates/              # HTML templates
│   ├── start_web_interface.sh  # Launch script
│   └── web_requirements.txt    # Dependencies
├── experiments/                # Experiment results (managed by web interface)
│   ├── temp/                   # In-progress experiments
│   ├── runs/                   # Completed experiments
│   └── failed/                 # Failed/stopped experiments
├── plots/                      # Generated plots
└── src/                        # Source code (called by web interface)
```

## How It Works

### Primary Use: Parameter Tuning Dashboard

The web interface is designed as a **comparison dashboard** for finding optimal parameters:

1. **Run experiments** from command line with different parameter configurations:
   ```bash
   python src/main.py --config configs/master_config.json
   ```

2. **Open dashboard** in browser (`http://localhost:5000`)

3. **Browse all runs** - see both ongoing and completed experiments in one view

4. **Filter and compare**:
   - Filter by method (Greedy IG, Dec-MCTS, MH Full, MH Efficient)
   - Filter by mode (IG_BS, IGd_BM, etc.)
   - View results side-by-side to compare performance

5. **Find optimal parameters** by comparing metrics across runs

6. **Monitor in real-time** - dashboard auto-refreshes every 10 seconds

### Alternative: Run New Experiments from UI

You can also start new experiments from the web interface:

1. Scroll down to "Start New Experiment" section
2. Select method and configure parameters
3. Click "Run Experiment"
4. Monitor progress in real-time

### Viewing Results

- **Click "View Results"** on any run card to see plots
- Results show even for incomplete/ongoing experiments
- Selected run is highlighted for easy tracking
- Results update automatically with latest data

## Path Management

The web interface calculates `PROJECT_ROOT` from its location and uses absolute paths:
- All experiment directories: `PROJECT_ROOT/experiments/`
- All plots: `PROJECT_ROOT/plots/`
- Source code: `PROJECT_ROOT/src/main.py`

This ensures correct operation regardless of where the web interface is started from.

## Alternative: Command Line

This web interface is optional. You can also run experiments via:

**Single runs:**
```bash
python src/main.py --config configs/master_config.json
```

**Batch parameter sweeps:**
```bash
./sweep.sh  # Edit sweep.json for parameter grid
```

## Dependencies

Install web interface dependencies:
```bash
pip install -r web_requirements.txt
```

Main project dependencies are in the root `requirements.txt`.

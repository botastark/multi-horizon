#!/bin/bash
# Launcher script for MCTS Web Interface

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================="
echo "  MCTS Experiment Web Interface"
echo "========================================="
echo ""

# Check if we're in the active_sensing environment
if [[ "$CONDA_DEFAULT_ENV" != "active_sensing" ]] && [[ -z "$VIRTUAL_ENV" || "$VIRTUAL_ENV" != *"active_sensing"* ]]; then
    echo "WARNING: You should activate the active_sensing conda environment first!"
    echo "Run: conda activate active_sensing"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate the environment and try again."
        exit 1
    fi
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Flask is not installed. Installing dependencies..."
    pip install -r "$SCRIPT_DIR/web_requirements.txt"
    echo ""
fi

# Check if numpy is installed (needed for experiments)
if ! python -c "import numpy" 2>/dev/null; then
    echo "NumPy is not installed. Installing main dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo ""
fi

# Create required directories in project root
cd "$PROJECT_ROOT"
mkdir -p plots
mkdir -p experiments/temp
mkdir -p experiments/runs
mkdir -p experiments/failed

echo "Starting web server from: $PROJECT_ROOT"
echo "Open your browser and navigate to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python "$SCRIPT_DIR/web_interface.py"

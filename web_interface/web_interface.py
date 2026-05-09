#!/usr/bin/env python3
"""
Flask Web Interface for MCTS Experiment Configuration and Execution
"""

import json
import os
import sys
import subprocess
import threading
import time
import re
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

# Get project root directory (parent of web_interface/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__)

# Configure logging to reduce verbosity
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)  # Only show errors, not INFO logs

# Global state for tracking experiments
experiment_state = {
    "running": False,
    "current_iter": 0,
    "total_iters": 0,
    "start_time": None,
    "config": None,
    "results_path": None,
    "log_output": [],
    "error": None,
    "process": None,  # Store process handle for stopping
    "debug_logs_enabled": False,  # Toggle for verbose debug logs
    "debug_log_file": None,  # Path to debug log file for current run
}

# Method configurations with parameters and descriptions
# Ordered as: 1.IG, 2.DecMCTS, 3.MH_Full, 4.MH_Eff
METHOD_CONFIGS = {
    "greedy_ig": {
        "name": "1. IG (Greedy Information Gain)",
        "description": "Single-step lookahead baseline with pure belief-driven information gain maximization",
        "config_file": os.path.join(PROJECT_ROOT, "configs/strategies/greedy_ig.json"),
        "parameters": {
            "overlap_penalty_weight": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Penalty weight for overlapping coverage between agents",
            },
            "radius_multiplier": {
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
                "description": "Communication range = radius_multiplier × grid_step",
            },
        },
    },
    "dec_mcts": {
        "name": "2. DecMCTS (Decentralized MCTS)",
        "description": "Single-level MCTS with multi-step trajectory planning using UCB tree search",
        "config_file": os.path.join(PROJECT_ROOT, "configs/strategies/dec_mcts.json"),
        "parameters": {
            "horizon": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Planning horizon (number of steps to look ahead)",
            },
            "iterations": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 200,
                "step": 10,
                "description": "Number of MCTS iterations per planning step",
            },
            "ucb_c": {
                "type": "float",
                "default": 1.4,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "description": "UCB exploration constant (higher = more exploration)",
            },
            "discount_factor": {
                "type": "float",
                "default": 0.95,
                "min": 0.5,
                "max": 1.0,
                "step": 0.05,
                "description": "Reward discount factor for future steps",
            },
            "timeout": {
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
                "description": "Maximum planning time per step (seconds)",
            },
            "overlap_penalty_weight": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Penalty weight for overlapping coverage",
            },
        },
    },
    "mh_dec_mcts_full": {
        "name": "3. MH_Full (Multi-Horizon Full)",
        "description": "Hierarchical planning: Both HLP and LLP use MCTS tree search",
        "config_file": os.path.join(
            PROJECT_ROOT, "configs/strategies/mh_dec_mcts_full.json"
        ),
        "parameters": {
            "llp_horizon": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Low-Level Planner horizon (tactical action selection)",
            },
            "llp_iterations": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
                "step": 10,
                "description": "LLP iterations (random rollout sampling)",
            },
            "llp_discount": {
                "type": "float",
                "default": 0.95,
                "min": 0.5,
                "max": 1.0,
                "step": 0.05,
                "description": "LLP discount factor",
            },
            "hlp_horizon": {
                "type": "int",
                "default": 8,
                "min": 3,
                "max": 15,
                "step": 1,
                "description": "High-Level Planner horizon (strategic region allocation)",
            },
            "hlp_iterations": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
                "step": 10,
                "description": "HLP MCTS iterations",
            },
            "hlp_ucb_c": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "description": "HLP UCB exploration constant",
            },
            "hlp_discount": {
                "type": "float",
                "default": 0.98,
                "min": 0.5,
                "max": 1.0,
                "step": 0.05,
                "description": "HLP discount factor",
            },
            "replan_interval": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.5,
                "description": "HLP replan interval (seconds)",
            },
            "overlap_penalty_weight": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Penalty weight for overlapping coverage",
            },
        },
    },
    "mh_dec_mcts_efficient": {
        "name": "4. MH_Eff (Multi-Horizon Efficient)",
        "description": "Hierarchical planning: HLP uses MCTS for strategic regions, LLP uses random rollout for tactics",
        "config_file": os.path.join(
            PROJECT_ROOT, "configs/strategies/mh_dec_mcts_efficient.json"
        ),
        "parameters": {
            "llp_horizon": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Low-Level Planner horizon",
            },
            "llp_iterations": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
                "step": 10,
                "description": "LLP MCTS iterations",
            },
            "llp_ucb_c": {
                "type": "float",
                "default": 1.4,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "description": "LLP UCB exploration constant",
            },
            "llp_discount": {
                "type": "float",
                "default": 0.95,
                "min": 0.5,
                "max": 1.0,
                "step": 0.05,
                "description": "LLP discount factor",
            },
            "hlp_horizon": {
                "type": "int",
                "default": 8,
                "min": 3,
                "max": 15,
                "step": 1,
                "description": "High-Level Planner horizon",
            },
            "hlp_iterations": {
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 100,
                "step": 10,
                "description": "HLP MCTS iterations",
            },
            "hlp_ucb_c": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "description": "HLP UCB exploration constant",
            },
            "hlp_discount": {
                "type": "float",
                "default": 0.98,
                "min": 0.5,
                "max": 1.0,
                "step": 0.05,
                "description": "HLP discount factor",
            },
            "replan_interval": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.5,
                "description": "HLP replan interval (seconds)",
            },
            "overlap_penalty_weight": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Penalty weight for overlapping coverage",
            },
        },
    },
}

# Common experiment parameters
COMMON_PARAMS = {
    "num_agents": {
        "type": "int",
        "default": 4,
        "min": 1,
        "max": 10,
        "step": 1,
        "description": "Number of UAV agents",
    },
    "n_steps": {
        "type": "int",
        "default": 15,
        "min": 5,
        "max": 50,
        "step": 5,
        "description": "Number of simulation steps per iteration",
    },
    "iters": {
        "type": "int",
        "default": 20,
        "min": 1,
        "max": 100,
        "step": 1,
        "description": "Number of experiment iterations (for averaging)",
    },
    "cluster_radius": {
        "type": "float",
        "default": 4.0,
        "min": 1.0,
        "max": 10.0,
        "step": 0.5,
        "description": "Gaussian field cluster radius",
    },
    "communication_range": {
        "type": "float",
        "default": 15.625,
        "min": 5.0,
        "max": 50.0,
        "step": 2.5,
        "description": "Agent communication range (meters)",
    },
    "mode_label": {
        "type": "select",
        "default": "IGd_BM",
        "options": ["IG_BS", "IGd_BM"],
        "description": "Information sharing mode (IG_BS=IG+BeliefSync, IGd_BM=IGd+BeliefMerge)",
    },
    "debug_logs": {
        "type": "select",
        "default": False,
        "options": [True, False],
        "description": "Enable detailed debug logs in logs/ directory (optional, for debugging only)",
    },
}


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/methods", methods=["GET"])
def get_methods():
    """Return available methods and their configurations in order."""
    # Return as a list to preserve order
    methods = []
    for key, config in METHOD_CONFIGS.items():
        methods.append(
            {"key": key, "name": config["name"], "description": config["description"]}
        )
    return jsonify(methods)


@app.route("/api/parameters/<method>", methods=["GET"])
def get_parameters(method):
    """Return parameters for a specific method."""
    if method not in METHOD_CONFIGS:
        return jsonify({"error": "Method not found"}), 404

    return jsonify(
        {
            "method_params": METHOD_CONFIGS[method]["parameters"],
            "common_params": COMMON_PARAMS,
        }
    )


@app.route("/api/run", methods=["POST"])
def run_experiment():
    """Start an experiment with given parameters."""
    global experiment_state

    if experiment_state["running"]:
        return jsonify({"error": "Experiment already running"}), 400

    data = request.json
    method = data.get("method")
    params = data.get("parameters", {})

    # Get debug_logs from parameters (it's a common parameter)
    debug_logs_enabled = params.get("debug_logs", False)
    # Convert string "true"/"false" to boolean if needed
    if isinstance(debug_logs_enabled, str):
        debug_logs_enabled = debug_logs_enabled.lower() == "true"

    if method not in METHOD_CONFIGS:
        return jsonify({"error": "Invalid method"}), 400

    # Ensure trials directory exists
    os.makedirs("trials", exist_ok=True)

    # Create custom config
    config = create_experiment_config(method, params)

    # Generate experiment metadata for tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_metadata = {
        "timestamp": timestamp,
        "method": method,
        "num_agents": params.get("num_agents", 4),
        "n_steps": params.get("n_steps", 15),
        "iters": params.get("iters", 20),
        "mode": params.get("mode_label", "IGd_BM"),
        "comm_range": params.get("radius_multiplier", "R5"),
    }

    # Create temp directory for this run
    run_dirname = create_run_dirname(exp_metadata)
    temp_run_path = os.path.join(PROJECT_ROOT, "experiments", "temp", run_dirname)
    os.makedirs(temp_run_path, exist_ok=True)

    # Set output directory in config so main.py uses it
    if "experiment" not in config:
        config["experiment"] = {}
    config["experiment"]["output_dir"] = temp_run_path

    # Save config to temp run directory
    config_path = os.path.join(temp_run_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save metadata for tracking and comparison
    metadata_path = os.path.join(temp_run_path, "metadata.json")
    full_metadata = {
        **exp_metadata,
        "parameters": params,
        "status": "running",
        "start_time": datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    # Create debug log file if debug logs enabled
    debug_log_file = None
    if debug_logs_enabled:
        # Create logs directory in temp_run_path
        logs_dir = os.path.join(temp_run_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        debug_log_file = os.path.join(logs_dir, f"debug_{run_dirname}.log")
        with open(debug_log_file, "w") as f:
            f.write(f"Debug logs for run: {run_dirname}\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    # Reset state
    experiment_state = {
        "running": True,
        "current_iter": 0,
        "total_iters": params.get("iters", 20),
        "start_time": datetime.now().isoformat(),
        "config": config,
        "metadata": exp_metadata,
        "config_file": config_path,
        "temp_run_path": temp_run_path,
        "run_dirname": run_dirname,
        "results_path": None,
        "log_output": [],
        "error": None,
        "debug_logs_enabled": debug_logs_enabled,
        "debug_log_file": debug_log_file,
    }

    # Run experiment in background thread
    thread = threading.Thread(target=run_experiment_background, args=(config_path,))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "config": config, "metadata": exp_metadata})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get current experiment status."""
    # Don't send process object in JSON
    state_copy = {k: v for k, v in experiment_state.items() if k != "process"}
    return jsonify(state_copy)


@app.route("/api/stop", methods=["POST"])
def stop_experiment():
    """Stop currently running experiment."""
    global experiment_state

    if not experiment_state.get("running"):
        return jsonify({"error": "No experiment is running"}), 400

    process = experiment_state.get("process")
    if process:
        try:
            import signal

            process.send_signal(signal.SIGTERM)
            experiment_state["log_output"].append(
                "[INFO] Stop requested - terminating process..."
            )
            return jsonify({"status": "stopping"})
        except Exception as e:
            error_msg = f"Failed to stop process: {str(e)}"
            experiment_state["log_output"].append(f"[ERROR] {error_msg}")
            return jsonify({"error": error_msg}), 500
    else:
        return jsonify({"error": "Process handle not found"}), 500


@app.route("/api/results", methods=["GET"])
def get_results():
    """Generate and return plots for current results (works with incomplete iterations)."""
    # Check if a specific run path is requested
    requested_run = request.args.get("run_path")

    if requested_run:
        # Load results from specified ongoing/temp run
        if os.path.exists(requested_run) and os.path.isdir(requested_run):
            results_path = requested_run
        else:
            return jsonify({"error": f"Run path not found: {requested_run}"}), 404
    else:
        # First try to get results_path from state
        results_path = experiment_state.get("results_path")

    # If not available, try temp_run_path directly (results are written there during run)
    if not results_path and experiment_state.get("temp_run_path"):
        temp_path = experiment_state["temp_run_path"]
        # Check if txt directory exists directly in temp_run_path
        txt_dir = os.path.join(temp_path, "txt")
        if os.path.exists(txt_dir) and os.path.isdir(txt_dir):
            # Results are being written directly to temp_run_path
            results_path = temp_path
            experiment_state["results_path"] = results_path
        else:
            # Check if trials subdirectory exists (older structure)
            trials_dir = os.path.join(temp_path, "trials")
            if os.path.exists(trials_dir):
                # Find the first (and usually only) trial run directory
                trial_dirs = [
                    d
                    for d in os.listdir(trials_dir)
                    if os.path.isdir(os.path.join(trials_dir, d))
                ]
                if trial_dirs:
                    results_path = os.path.join(trials_dir, trial_dirs[0])
                    experiment_state["results_path"] = results_path

    # If still not found, try lazy resolution
    if not results_path and experiment_state.get("config"):
        results_path = find_results_path(experiment_state["config"])
        if results_path:
            experiment_state["results_path"] = results_path

    if not results_path:
        return (
            jsonify(
                {
                    "error": "No results available yet. Please wait for first iteration to complete."
                }
            ),
            404,
        )

    try:
        # Call plotter to generate plots (works with partial data)
        plot_path = generate_plots(results_path)
        if plot_path:
            return jsonify({"plot_url": f"/api/plot/{plot_path}"})
        else:
            return (
                jsonify(
                    {"error": "Failed to generate plots. Results may not be ready yet."}
                ),
                500,
            )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/plot/<path:filename>")
def serve_plot(filename):
    """Serve generated plot image."""
    if not filename or filename == "None":
        return jsonify({"error": "No plot available"}), 404

    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    file_path = os.path.join(plot_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"Plot file not found: {filename}"}), 404

    return send_file(file_path, mimetype="image/png")


@app.route("/api/compare_methods", methods=["POST"])
def compare_methods():
    """Generate comparison plots for selected methods and runs."""
    try:
        data = request.get_json()
        run_paths = data.get("run_paths", [])
        include_timing = data.get("include_timing", True)
        timing_metric = data.get("timing_metric", "Mean_ms")

        if not run_paths:
            return jsonify({"error": "Please select at least 1 run to compare"}), 400

        # Generate comparison plots
        plot_urls = generate_comparison_plots(run_paths, include_timing, timing_metric)

        if plot_urls:
            return jsonify({"plots": plot_urls})
        else:
            return jsonify({"error": "Failed to generate comparison plots"}), 500

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ongoing_runs", methods=["GET"])
def get_ongoing_runs():
    """Get list of ongoing/temp runs AND completed runs (for dashboard comparison)."""
    all_runs = []

    try:
        # 1. Get ongoing runs from experiments/temp/
        temp_dir = os.path.join(PROJECT_ROOT, "experiments", "temp")
        if os.path.exists(temp_dir):
            for run_name in os.listdir(temp_dir):
                run_path = os.path.join(temp_dir, run_name)

                if os.path.isdir(run_path):
                    try:
                        run_info = extract_run_info(
                            run_name, run_path, status="ongoing"
                        )
                        if run_info:
                            all_runs.append(run_info)
                    except Exception as e:
                        print(f"Error processing temp run {run_name}: {e}")
                        continue

        # 2. Get completed runs from experiments/runs/<method>/
        runs_base = os.path.join(PROJECT_ROOT, "experiments", "runs")
        if os.path.exists(runs_base):
            for method in [
                "greedy_ig",
                "dec_mcts",
                "mh_dec_mcts_full",
                "mh_dec_mcts_efficient",
                "mh_dec_mcts_both",
                "mh_dec_mcts",  # For backward compatibility
            ]:
                method_dir = os.path.join(runs_base, method)
                if os.path.exists(method_dir):
                    for run_name in os.listdir(method_dir):
                        run_path = os.path.join(method_dir, run_name)
                        if os.path.isdir(run_path):
                            try:
                                run_info = extract_run_info(
                                    run_name,
                                    run_path,
                                    status="completed",
                                    method_hint=method,
                                )
                                if run_info:
                                    all_runs.append(run_info)
                            except Exception as e:
                                print(f"Error processing completed run {run_name}: {e}")
                                continue

        # Sort by modification time (most recent first)
        all_runs.sort(key=lambda x: x.get("modified", ""), reverse=True)

    except Exception as e:
        print(f"Error in get_ongoing_runs: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "runs": all_runs}), 500

    return jsonify({"runs": all_runs})


def extract_run_info(run_name, run_path, status="ongoing", method_hint=None):
    """Extract run information from directory name and contents."""
    # Extract method from directory name or hint
    method = method_hint
    if not method:
        # Check most specific patterns first
        if "greedy_ig" in run_name:
            method = "greedy_ig"
        elif "mh_dec_mcts_both" in run_name:
            method = "mh_dec_mcts_both"
        elif "mh_dec_mcts_full" in run_name:
            method = "mh_dec_mcts_full"
        elif "mh_dec_mcts" in run_name:
            # Plain mh_dec_mcts defaults to efficient
            method = "mh_dec_mcts_efficient"
        elif "dec_mcts" in run_name:
            method = "dec_mcts"
        else:
            method = "unknown"

    # Extract mode from directory name
    mode = "unknown"
    for m in ["IG_BS", "IGd_BM", "IG_BM", "IGd_BS", "IGd", "IG"]:
        if m in run_name:
            mode = m
            break

    # Extract parameters from run name
    params = {}

    # Extract num agents (N4, N8, etc)
    import re

    n_match = re.search(r"_N(\d+)", run_name)
    if n_match:
        params["num_agents"] = int(n_match.group(1))

    # Extract radius (r2, r4, r6, etc)
    r_match = re.search(r"_r(\d+(?:\.\d+)?)", run_name)
    if r_match:
        params["radius"] = float(r_match.group(1))

    # Extract pairwise mode (gaussian, corner, etc)
    if "gaussian" in run_name.lower():
        params["pairwise"] = "gaussian"
    elif "corner" in run_name.lower():
        params["pairwise"] = "corner"
    elif "adaptive" in run_name.lower():
        params["pairwise"] = "adaptive"

    # Extract comm range (commRinf, commR10, etc)
    comm_match = re.search(r"commR(inf|\d+(?:\.\d+)?)", run_name)
    if comm_match:
        params["comm_range"] = comm_match.group(1)

    # Extract method-specific hyperparameters from saved config or run.log if available
    hyperparams = _extract_hyperparams_from_log(run_path, method)
    if not hyperparams:
        default_config = _load_method_default_config(method)
        if default_config:
            hyperparams = _extract_hyperparams_from_config(default_config, method)

    # Get modification time
    try:
        mtime = os.path.getmtime(run_path)
        mtime_str = datetime.fromtimestamp(mtime).isoformat()
    except:
        mtime_str = datetime.now().isoformat()

    # Check if has results data
    txt_dir = os.path.join(run_path, "txt")
    has_data = os.path.exists(txt_dir)

    # Count log files if available
    file_count = 0
    if has_data:
        try:
            file_count = len(
                [
                    f
                    for f in os.listdir(txt_dir)
                    if os.path.isfile(os.path.join(txt_dir, f))
                ]
            )
        except:
            pass

    # Extract performance metrics (final coverage, entropy, etc.)
    metrics = {}
    if has_data:
        try:
            # Try to read from txt files
            txt_files = [
                f
                for f in os.listdir(txt_dir)
                if f.endswith(".txt") or f.endswith(".log")
            ]
            if txt_files:
                # Use first txt file to get final metrics
                first_file = os.path.join(txt_dir, txt_files[0])
                metrics = _extract_final_metrics(first_file)
        except Exception as e:
            pass  # Metrics will remain empty

    # Map method to display name
    method_display_map = {
        "greedy_ig": "Greedy-IG",
        "dec_mcts": "Dec-MCTS",
        "mh_dec_mcts_full": "MH-Full",
        "mh_dec_mcts_efficient": "MH-Eff",
        "mh_dec_mcts": "MH-Eff",  # Plain mh_dec_mcts defaults to efficient
        "mh_dec_mcts_both": "MH-Both",
        "unknown": "Unknown",
    }
    method_display = method_display_map.get(method, method)
    hyperparam_summary = _summarize_hyperparams(method, hyperparams)
    comparison_label = _build_run_comparison_label(
        {
            "id": run_name,
            "method": method,
            "method_display": method_display,
            "mode": mode,
            "hyperparams": hyperparams,
        }
    )

    return {
        "id": run_name,
        "method": method,
        "method_display": method_display,
        "mode": mode,
        "path": run_path,
        "modified": mtime_str,
        "has_data": has_data,
        "file_count": file_count,
        "status": status,
        "params": params,
        "hyperparams": hyperparams,
        "hyperparam_summary": hyperparam_summary,
        "comparison_label": comparison_label,
        "metrics": metrics,
    }


def _extract_hyperparams_from_log(run_path, method):
    """Extract method-specific hyperparameters from run.log file."""
    hyperparams = {}

    config_snapshot = _load_saved_run_config(run_path)
    if config_snapshot:
        hyperparams.update(_extract_hyperparams_from_config(config_snapshot, method))
        if hyperparams:
            return hyperparams

    # Try to read from run.log
    log_file = os.path.join(run_path, "txt", "run.log")
    if not os.path.exists(log_file):
        return hyperparams

    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Preferred format: explicit JSON line written by experiment runner
        # Example: hyperparams: {"horizon": 10, "iterations": 40, ...}
        hyper_match = re.search(r"hyperparams:\s*(\{[^\n]*\})", content)
        if hyper_match:
            try:
                parsed = json.loads(hyper_match.group(1))
                if isinstance(parsed, dict):
                    hyperparams.update(parsed)
                    return hyperparams
            except Exception:
                pass

        # Look for mcts_params line
        params_match = re.search(r"mcts_params:\s*\{([^}]*)\}", content)
        if params_match:
            params_str = params_match.group(1)
            # Parse simple key: value pairs
            for param_pair in params_str.split(","):
                if ":" in param_pair:
                    key, value = param_pair.split(":", 1)
                    key = key.strip().strip("'\"")
                    value = value.strip().strip("'\"")
                    try:
                        # Try to convert to number
                        if "." in value:
                            hyperparams[key] = float(value)
                        else:
                            hyperparams[key] = int(value)
                    except ValueError:
                        hyperparams[key] = value

        # Extract specific hyperparams based on method
        if method in ["mh_dec_mcts_full", "mh_dec_mcts_efficient", "mh_dec_mcts"]:
            # Look for horizon, iterations, etc in the log
            horizon_match = re.search(r"horizon[:\s]+([\d.]+)", content, re.IGNORECASE)
            if horizon_match:
                hyperparams["horizon"] = int(horizon_match.group(1))

            iter_match = re.search(r"iterations[:\s]+([\d.]+)", content, re.IGNORECASE)
            if iter_match:
                hyperparams["iterations"] = int(iter_match.group(1))

        elif method == "dec_mcts":
            horizon_match = re.search(r"horizon[:\s]+([\d.]+)", content, re.IGNORECASE)
            if horizon_match:
                hyperparams["horizon"] = int(horizon_match.group(1))

            iter_match = re.search(r"iterations[:\s]+([\d.]+)", content, re.IGNORECASE)
            if iter_match:
                hyperparams["iterations"] = int(iter_match.group(1))

    except Exception as e:
        pass

    return hyperparams


def _load_saved_run_config(run_path):
    """Load a saved config snapshot for a completed run if available."""
    for filename in ["config_resolved.json", "config.json"]:
        candidate_path = os.path.join(run_path, filename)
        if os.path.exists(candidate_path):
            try:
                with open(candidate_path, "r") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _load_method_default_config(method):
    """Load the default strategy config for a method as a fallback summary source."""
    method_aliases = {
        "mh_dec_mcts": "mh_dec_mcts_efficient",
        "mh_dec_mcts_both": "mh_dec_mcts_full",
    }
    canonical_method = method_aliases.get(method, method)
    config_path = METHOD_CONFIGS.get(canonical_method, {}).get("config_file")

    if not config_path or not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_hyperparams_from_config(config, method):
    """Extract method-specific hyperparameters from a saved config snapshot."""
    hyperparams = {}
    decentralized_cfg = config.get("decentralized", {})

    if method == "greedy_ig":
        greedy_cfg = config.get("greedy_ig", {})
        for key in ["overlap_penalty_weight"]:
            if key in greedy_cfg:
                hyperparams[key] = greedy_cfg[key]
        for key in ["radius_multiplier"]:
            if key in decentralized_cfg:
                hyperparams[key] = decentralized_cfg[key]
    elif method == "dec_mcts":
        dec_cfg = config.get("dec_mcts", {})
        for key in ["horizon", "iterations", "ucb_c", "discount_factor", "timeout"]:
            if key in dec_cfg:
                hyperparams[key] = dec_cfg[key]
        for key in ["overlap_penalty_weight", "radius_multiplier"]:
            if key in decentralized_cfg:
                hyperparams[key] = decentralized_cfg[key]
    elif method in [
        "mh_dec_mcts_efficient",
        "mh_dec_mcts_full",
        "mh_dec_mcts",
        "mh_dec_mcts_both",
    ]:
        hier_cfg = config.get("hierarchical_dec_mcts", {})
        llp_cfg = hier_cfg.get("llp", {})
        hlp_cfg = hier_cfg.get("hlp", {})

        if llp_cfg:
            for key in ["horizon", "iterations", "ucb_c", "discount_factor"]:
                if key in llp_cfg:
                    hyperparams[f"llp_{key}"] = llp_cfg[key]
        else:
            for key, value in hier_cfg.items():
                if key.startswith("llp_"):
                    hyperparams[key] = value

        if hlp_cfg:
            for key in [
                "horizon",
                "iterations",
                "ucb_c",
                "discount_factor",
                "replan_interval",
            ]:
                if key in hlp_cfg:
                    mapped_key = (
                        "hlp_replan_interval"
                        if key == "replan_interval"
                        else f"hlp_{key}"
                    )
                    hyperparams[mapped_key] = hlp_cfg[key]
        else:
            for key, value in hier_cfg.items():
                if key.startswith("hlp_"):
                    hyperparams[key] = value

        use_mcts_llp = hier_cfg.get("use_mcts_llp")
        if use_mcts_llp is not None:
            hyperparams["use_mcts_llp"] = use_mcts_llp

        for key in ["overlap_penalty_weight", "radius_multiplier"]:
            if key in decentralized_cfg:
                hyperparams[key] = decentralized_cfg[key]

    return hyperparams


def _format_hyperparam_value(value):
    """Format hyperparameter values compactly for legends/UI."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _summarize_hyperparams(method, hyperparams):
    """Build a compact hyperparameter summary string for legends/UI."""
    if method == "greedy_ig":
        if not hyperparams:
            return "single-step greedy"
        key_map = [
            ("overlap_penalty_weight", "pen"),
            ("radius_multiplier", "rm"),
        ]
    elif method == "dec_mcts":
        if not hyperparams:
            return None
        key_map = [
            ("horizon", "h"),
            ("iterations", "it"),
            ("ucb_c", "c"),
            ("discount_factor", "df"),
            ("timeout", "to"),
            ("overlap_penalty_weight", "pen"),
            ("radius_multiplier", "rm"),
        ]
    elif method in [
        "mh_dec_mcts_efficient",
        "mh_dec_mcts",
        "mh_dec_mcts_full",
        "mh_dec_mcts_both",
    ]:
        if not hyperparams:
            return None
        key_map = [
            ("llp_horizon", "llp_h"),
            ("llp_iterations", "llp_it"),
            ("llp_ucb_c", "llp_c"),
            ("hlp_horizon", "hlp_h"),
            ("hlp_iterations", "hlp_it"),
            ("hlp_ucb_c", "hlp_c"),
            ("hlp_replan_interval", "rp"),
            ("use_mcts_llp", "llp"),
            ("overlap_penalty_weight", "pen"),
            ("radius_multiplier", "rm"),
        ]
    else:
        if not hyperparams:
            return None
        key_map = [(key, key) for key in sorted(hyperparams.keys())]

    tokens = []
    for key, short_key in key_map:
        if key not in hyperparams:
            continue
        value = hyperparams[key]
        if key == "use_mcts_llp":
            tokens.append("llp=mcts" if value else "llp=rollout")
        else:
            tokens.append(f"{short_key}={_format_hyperparam_value(value)}")

    return ", ".join(tokens) if tokens else None


def _build_run_comparison_label(run_info, include_method=True):
    """Build a concise label that distinguishes runs by mode and hyperparameters."""
    parts = []

    if include_method:
        method_display = run_info.get("method_display") or run_info.get("method")
        if method_display:
            parts.append(method_display)

    mode = run_info.get("mode")
    if mode and mode != "unknown":
        parts.append(mode)

    hyperparam_summary = _summarize_hyperparams(
        run_info.get("method"), run_info.get("hyperparams", {})
    )
    if hyperparam_summary:
        parts.append(hyperparam_summary)
    elif run_info.get("id"):
        parts.append(run_info["id"].replace("run_", "", 1)[:20])

    return " | ".join(parts) if parts else run_info.get("id", "run")


def _extract_final_metrics(file_path):
    """Extract final step metrics from a txt/log file."""
    metrics = {}
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Find the last data row (last line starting with a number)
        last_data_line = None
        for line in reversed(lines):
            line = line.strip()
            if line and line[0].isdigit():
                last_data_line = line
                break

        if last_data_line:
            # Parse the line - format: Step Entropy MSE Height Coverage ...
            parts = last_data_line.split()
            if len(parts) >= 5:
                try:
                    metrics["final_step"] = int(parts[0])
                    metrics["final_entropy"] = float(parts[1])
                    metrics["final_mse"] = float(parts[2])
                    metrics["final_height"] = float(parts[3])
                    metrics["final_coverage"] = float(parts[4])
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        pass

    return metrics


@app.route("/api/completed_runs", methods=["GET"])
def get_completed_runs():
    """Get list of completed runs matching current method and settings."""
    method = request.args.get("method")
    num_agents = request.args.get("num_agents")
    mode = request.args.get("mode")

    if not method:
        return jsonify({"error": "Method parameter required"}), 400

    # Search in experiments/runs/<method>/
    runs_dir = os.path.join(PROJECT_ROOT, "experiments", "runs", method)
    completed_runs = []

    if os.path.exists(runs_dir):
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            metadata_path = os.path.join(run_path, "metadata.json")

            if os.path.isdir(run_path):
                try:
                    run_info = extract_run_info(
                        run_name, run_path, status="completed", method_hint=method
                    )

                    meta = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)

                    # Filter by settings (not all params)
                    if num_agents and str(
                        meta.get(
                            "num_agents", run_info.get("params", {}).get("num_agents")
                        )
                    ) != str(num_agents):
                        continue
                    if mode and meta.get("mode", run_info.get("mode")) != mode:
                        continue

                    # Check if run has results
                    has_results = False
                    for subdir in ["txt", "logs", "trials"]:
                        if os.path.exists(os.path.join(run_path, subdir)):
                            has_results = True
                            break

                    if has_results or meta.get("status") == "completed":
                        completed_runs.append(
                            {
                                "id": run_name,
                                "timestamp": meta.get(
                                    "timestamp", run_info.get("modified")
                                ),
                                "start_time": meta.get(
                                    "start_time", run_info.get("modified")
                                ),
                                "end_time": meta.get("end_time"),
                                "num_agents": meta.get(
                                    "num_agents",
                                    run_info.get("params", {}).get("num_agents"),
                                ),
                                "n_steps": meta.get("n_steps"),
                                "iters": meta.get("iters"),
                                "mode": meta.get("mode", run_info.get("mode")),
                                "comm_range": meta.get(
                                    "comm_range",
                                    run_info.get("params", {}).get("comm_range"),
                                ),
                                "path": run_path,
                                "hyperparams": run_info.get("hyperparams", {}),
                                "comparison_label": run_info.get("comparison_label"),
                            }
                        )
                except Exception as e:
                    print(f"Error reading metadata from {run_name}: {e}")
                    continue

    # Sort by timestamp (most recent first)
    completed_runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return jsonify({"runs": completed_runs})


@app.route("/api/best_runs", methods=["GET"])
def get_best_runs():
    """Get performance comparison metrics grouped by mode and hyperparameter variant."""

    method_config_map = {
        "greedy_ig": "greedy_ig",
        "dec_mcts": "dec_mcts",
        "mh_dec_mcts_full": "mh_dec_mcts_both",
        "mh_dec_mcts_efficient": "mh_dec_mcts",
    }

    variant_groups = {}
    runs_base = os.path.join(PROJECT_ROOT, "experiments", "runs")

    if os.path.exists(runs_base):
        for method_key, folder_name in method_config_map.items():
            method_dir = os.path.join(runs_base, folder_name)
            if not os.path.exists(method_dir):
                continue

            for run_name in os.listdir(method_dir):
                run_path = os.path.join(method_dir, run_name)
                if not os.path.isdir(run_path):
                    continue

                try:
                    run_info = extract_run_info(
                        run_name,
                        run_path,
                        status="completed",
                        method_hint=method_key,
                    )
                    if not run_info or not run_info.get("has_data"):
                        continue

                    run_info["early_metrics"] = _calculate_early_metrics(run_path)
                    hyperparam_summary = (
                        run_info.get("hyperparam_summary") or "config unavailable"
                    )
                    variant_key = (
                        run_info.get("mode", "unknown"),
                        run_info.get("method", method_key),
                        hyperparam_summary,
                    )
                    variant_groups.setdefault(variant_key, []).append(run_info)
                except Exception as e:
                    print(f"Error processing run {run_name}: {e}")
                    continue

    variant_rows = []
    for (mode, method_key, hyperparam_summary), runs in variant_groups.items():
        sorted_runs = sorted(
            runs,
            key=lambda run: run.get("modified", ""),
            reverse=True,
        )
        sample_run = sorted_runs[0]
        variant_rows.append(
            {
                "mode": mode,
                "method": method_key,
                "method_display": sample_run.get("method_display", method_key),
                "hyperparam_summary": hyperparam_summary,
                "comparison_label": sample_run.get("comparison_label"),
                "metrics": _aggregate_method_metrics(runs),
                "num_runs": len(runs),
                "run_paths": [r.get("path") for r in sorted_runs if r.get("path")],
                "runs": [
                    {
                        "id": run.get("id"),
                        "path": run.get("path"),
                        "modified": run.get("modified"),
                        "mode": run.get("mode"),
                        "method": run.get("method"),
                        "method_display": run.get("method_display"),
                        "comparison_label": run.get("comparison_label"),
                        "hyperparam_summary": run.get("hyperparam_summary"),
                        "metrics": run.get("metrics", {}),
                        "early_metrics": run.get("early_metrics", {}),
                    }
                    for run in sorted_runs
                    if run.get("path")
                ],
            }
        )

    mode_order = {"IG_BS": 0, "IGd_BM": 1, "IG_BM": 2, "IGd_BS": 3, "IGd": 4, "IG": 5}
    method_order = {
        "Greedy-IG": 0,
        "Dec-MCTS": 1,
        "MH-Full": 2,
        "MH-Eff": 3,
        "MH-Both": 4,
    }
    variant_rows.sort(
        key=lambda row: (
            mode_order.get(row.get("mode"), 99),
            method_order.get(row.get("method_display"), 99),
            row.get("hyperparam_summary", ""),
        )
    )

    return jsonify({"variants": variant_rows})


def _calculate_early_metrics(run_path):
    """Calculate metrics for early steps (step 0-20) to measure convergence speed."""
    metrics = {
        "entropy_drop_rate": None,  # How fast entropy decreases
        "mse_drop_rate": None,  # How fast MSE decreases
        "avg_planning_time": None,  # Average planning time
    }

    txt_dir = os.path.join(run_path, "txt")
    if not os.path.exists(txt_dir):
        return metrics

    try:
        # Read txt files to get step-by-step data
        txt_files = [
            f for f in os.listdir(txt_dir) if f.endswith(".txt") or f.endswith(".log")
        ]
        if not txt_files:
            return metrics

        first_file = os.path.join(txt_dir, txt_files[0])

        with open(first_file, "r") as f:
            lines = f.readlines()

        # Parse data lines
        data_rows = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        step = int(parts[0])
                        entropy = float(parts[1])
                        mse = float(parts[2])
                        data_rows.append({"step": step, "entropy": entropy, "mse": mse})
                    except (ValueError, IndexError):
                        continue

        if len(data_rows) < 2:
            return metrics

        # Focus on first 20 steps
        early_rows = [r for r in data_rows if r["step"] <= 20]
        if len(early_rows) < 2:
            early_rows = data_rows[: min(20, len(data_rows))]

        # Calculate drop rates (initial - final) / steps
        if len(early_rows) >= 2:
            initial = early_rows[0]
            final = early_rows[-1]
            num_steps = final["step"] - initial["step"]

            if num_steps > 0:
                metrics["entropy_drop_rate"] = (
                    initial["entropy"] - final["entropy"]
                ) / num_steps
                metrics["mse_drop_rate"] = (initial["mse"] - final["mse"]) / num_steps

        # Calculate average planning time from timestamps.csv
        timing_csv = os.path.join(txt_dir, "timestamps.csv")
        if os.path.exists(timing_csv):
            import csv

            times = []
            with open(timing_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        step_raw = row.get("step")
                        if step_raw in (None, ""):
                            continue

                        step = int(float(step_raw))
                        if step > 20:
                            continue

                        planning_time_raw = row.get("planning_time_ms")
                        if planning_time_raw not in (None, ""):
                            time_ms = float(planning_time_raw)
                        else:
                            start_raw = row.get("start_ms")
                            end_raw = row.get("end_ms")
                            if start_raw in (None, "") or end_raw in (None, ""):
                                continue
                            time_ms = float(end_raw) - float(start_raw)

                        if time_ms >= 0:
                            times.append(time_ms)
                    except (TypeError, ValueError, KeyError):
                        continue

            if times:
                metrics["avg_planning_time"] = sum(times) / len(times)

    except Exception as e:
        print(f"Error calculating early metrics for {run_path}: {e}")

    return metrics


def _aggregate_method_metrics(runs):
    """Aggregate metrics across multiple runs of the same method."""
    aggregated = {
        "entropy_drop_rate": [],
        "mse_drop_rate": [],
        "avg_planning_time": [],
        "final_coverage": [],
        "final_entropy": [],
        "final_mse": [],
    }

    for run in runs:
        early = run.get("early_metrics", {})
        final = run.get("metrics", {})

        if early.get("entropy_drop_rate") is not None:
            aggregated["entropy_drop_rate"].append(early["entropy_drop_rate"])
        if early.get("mse_drop_rate") is not None:
            aggregated["mse_drop_rate"].append(early["mse_drop_rate"])
        if early.get("avg_planning_time") is not None:
            aggregated["avg_planning_time"].append(early["avg_planning_time"])

        if final.get("final_coverage") is not None:
            aggregated["final_coverage"].append(final["final_coverage"])
        if final.get("final_entropy") is not None:
            aggregated["final_entropy"].append(final["final_entropy"])
        if final.get("final_mse") is not None:
            aggregated["final_mse"].append(final["final_mse"])

    # Calculate means
    result = {}
    for key, values in aggregated.items():
        if values:
            result[key] = sum(values) / len(values)
            result[f"{key}_std"] = (
                sum((x - result[key]) ** 2 for x in values) / len(values)
            ) ** 0.5
        else:
            result[key] = None
            result[f"{key}_std"] = None

    return result


@app.route("/api/runs_by_hyperparam", methods=["GET"])
def get_runs_by_hyperparam():
    """Get runs grouped by a specific hyperparameter value for comparison within a method."""
    method = request.args.get("method")
    hyperparam = request.args.get(
        "hyperparam"
    )  # e.g., 'horizon', 'iterations', 'ucb_c'
    metric = request.args.get("metric", "final_coverage")

    if not method or not hyperparam:
        return jsonify({"error": "method and hyperparam parameters required"}), 400

    # Collect all runs for the specified method
    runs_by_param_value = {}
    runs_base = os.path.join(PROJECT_ROOT, "experiments", "runs", method)

    if os.path.exists(runs_base):
        for run_name in os.listdir(runs_base):
            run_path = os.path.join(runs_base, run_name)
            if os.path.isdir(run_path):
                run_info = extract_run_info(
                    run_name, run_path, status="completed", method_hint=method
                )

                if run_info and run_info.get("has_data"):
                    # Get hyperparam value
                    param_value = run_info.get("hyperparams", {}).get(hyperparam)

                    if param_value is not None:
                        # Group by parameter value
                        key = str(param_value)
                        if key not in runs_by_param_value:
                            runs_by_param_value[key] = []
                        runs_by_param_value[key].append(run_info)

    # Sort runs within each parameter value by metric
    reverse = True if "mse" not in metric.lower() else False
    for param_val in runs_by_param_value:
        runs_by_param_value[param_val].sort(
            key=lambda x: x.get("metrics", {}).get(
                metric, -float("inf") if reverse else float("inf")
            ),
            reverse=reverse,
        )

    return jsonify(
        {
            "runs_by_hyperparam": runs_by_param_value,
            "method": method,
            "hyperparam": hyperparam,
            "metric": metric,
        }
    )


@app.route("/api/multi_plot", methods=["POST"])
def generate_multi_plot():
    """Generate plot overlaying multiple runs."""
    data = request.json
    run_ids = data.get("run_ids", [])
    method = data.get("method")

    if not run_ids or not method:
        return jsonify({"error": "run_ids and method required"}), 400

    # Resolve run paths
    run_paths = []
    for run_id in run_ids:
        if run_id == "current":
            # Include current running/completed experiment
            if experiment_state.get("results_path"):
                run_paths.append(
                    {"id": "current", "path": experiment_state["results_path"]}
                )
            elif experiment_state.get("temp_run_path"):
                run_paths.append(
                    {"id": "current", "path": experiment_state["temp_run_path"]}
                )
        else:
            # Find in experiments/runs/<method>/
            run_path = os.path.join(PROJECT_ROOT, "experiments", "runs", method, run_id)
            if os.path.exists(run_path):
                run_paths.append({"id": run_id, "path": run_path})

    if not run_paths:
        return jsonify({"error": "No valid run paths found"}), 404

    # Generate multi-run plot - use current experiment's timestamp as ID
    try:
        # Get current experiment timestamp
        run_id = None
        if experiment_state.get("metadata"):
            run_id = experiment_state["metadata"].get("timestamp")

        # Fallback to current timestamp if no experiment running
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = f"run_{run_id}.png"
        output_path = os.path.join("plots", output_file)

        result = create_multi_run_plot(run_paths, output_path)
        if result:
            return jsonify({"plot_url": f"/api/plot/{output_file}"})
        else:
            return jsonify({"error": "Failed to generate multi-run plot"}), 500
    except Exception as e:
        print(f"Error generating multi-run plot: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def create_experiment_config(method, params):
    """Create experiment configuration from method and parameters."""
    # Load base config
    base_config_path = METHOD_CONFIGS[method]["config_file"]
    with open(base_config_path, "r") as f:
        config = json.load(f)

    # Generate experiment name with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_short = method.replace("_", "-")
    n_agents = params.get("num_agents", 4)
    mode = params.get("mode_label", "IGd_BM")

    # Update experiment metadata
    if "experiment" not in config:
        config["experiment"] = {}

    # Create descriptive experiment name
    exp_name = f"{method_short}_N{n_agents}_{mode}_{timestamp}"
    config["experiment"]["name"] = exp_name
    config["experiment"]["timestamp"] = timestamp
    config["experiment"]["web_interface"] = True

    # Update method-specific parameters
    method_key = config.get("action_strategy")

    if method == "greedy_ig":
        if "greedy_ig" in config:
            config["greedy_ig"]["overlap_penalty_weight"] = params.get(
                "overlap_penalty_weight", 0.0
            )
        if "decentralized" in config:
            config["decentralized"]["radius_multiplier"] = params.get(
                "radius_multiplier", 5.0
            )

    elif method == "dec_mcts":
        if "dec_mcts" in config:
            config["dec_mcts"].update(
                {
                    "horizon": params.get("horizon", 10),
                    "iterations": params.get("iterations", 30),
                    "ucb_c": params.get("ucb_c", 1.4),
                    "discount_factor": params.get("discount_factor", 0.95),
                    "timeout": params.get("timeout", 5.0),
                }
            )
        if "decentralized" in config:
            config["decentralized"]["overlap_penalty_weight"] = params.get(
                "overlap_penalty_weight", 0.3
            )

    elif method in ["mh_dec_mcts_efficient", "mh_dec_mcts_full"]:
        if "hierarchical_dec_mcts" in config:
            config["hierarchical_dec_mcts"]["llp"].update(
                {
                    "horizon": params.get("llp_horizon", 3),
                    "iterations": params.get("llp_iterations", 30),
                    "discount_factor": params.get("llp_discount", 0.95),
                }
            )
            config["hierarchical_dec_mcts"]["hlp"].update(
                {
                    "horizon": params.get("hlp_horizon", 8),
                    "iterations": params.get("hlp_iterations", 30),
                    "ucb_c": params.get("hlp_ucb_c", 1.0),
                    "discount_factor": params.get("hlp_discount", 0.98),
                    "replan_interval": params.get("replan_interval", 2.0),
                }
            )

            # Full version has LLP UCB
            if method == "mh_dec_mcts_full":
                config["hierarchical_dec_mcts"]["llp"]["ucb_c"] = params.get(
                    "llp_ucb_c", 1.4
                )

        if "decentralized" in config:
            config["decentralized"]["overlap_penalty_weight"] = params.get(
                "overlap_penalty_weight", 0.3
            )

    # Update common parameters (create shared section if not exists)
    if "shared" not in config:
        config["shared"] = {}

    # Get and convert debug_logs to boolean
    debug_logs_param = params.get("debug_logs", False)
    if isinstance(debug_logs_param, str):
        debug_logs_param = debug_logs_param.lower() == "true"

    # Get cluster_radius as integer (this is what gets passed as field_type to Field class)
    cluster_radius = int(params.get("cluster_radius", 4))

    config["shared"].update(
        {
            "project_path": "./",
            "num_agents": params.get("num_agents", 4),
            "n_steps": params.get("n_steps", 15),
            "iters": [0, int(params.get("iters", 20))],  # Ensure integer
            "cluster_radius": cluster_radius,
            "mode_labels": [params.get("mode_label", "IGd_BM")],
            "enable_plotting": True,
            "enable_logging": True,
            "debug_logs": debug_logs_param,  # Boolean, not string
            "field_type": "Gaussian",
            "start_position": params.get("start_position", "corner"),
            "correlation_types": ["adaptive"],
            "error_margins": [None],
        }
    )

    # Also set these at root level for main.py compatibility
    # For Gaussian fields, field_type should be the cluster_radius INTEGER (not "Gaussian" string)
    # This matches how main.py works: field_type = grf_r (the radius integer)
    config["field_type"] = (
        "Gaussian"  # Keep as string for main.py to detect and convert
    )
    config["cluster_radius"] = cluster_radius
    config["num_agents"] = params.get("num_agents", 4)
    config["n_steps"] = params.get("n_steps", 15)
    config["iters"] = [0, int(params.get("iters", 20))]
    config["mode_labels"] = [params.get("mode_label", "IGd_BM")]
    config["start_position"] = params.get("start_position", "corner")
    config["correlation_types"] = ["adaptive"]
    config["error_margins"] = [None]

    # Update communication range
    if "decentralized" not in config:
        config["decentralized"] = {}
    config["decentralized"]["communication_range"] = params.get(
        "communication_range", 15.625
    )

    return config


def _is_debug_log(line: str) -> bool:
    """Check if a log line is verbose debug info that should be filtered."""
    if not line:
        return False
    line_lower = line.lower()
    # Filter out step-by-step agent logs and verbose planning details
    debug_patterns = [
        "step " in line_lower
        and ("processing" in line_lower or "planning" in line_lower),
        "[agent" in line_lower and "step" in line_lower and "action=" in line_lower,
        "action=" in line_lower and "pos=(" in line_lower,
        line.startswith("────"),  # separator lines
        # Filter "Step X: Action=..., Entropy=..., MSE=..." logs
        (
            "step " in line_lower
            and "action=" in line_lower
            and ("entropy=" in line_lower or "mse=" in line_lower)
        ),
    ]
    return any(debug_patterns)


def _clean_stream_line(line: str) -> str:
    """Normalize subprocess stream lines for UI display/parsing."""
    if not line:
        return ""
    # Remove ANSI escape codes and normalize carriage returns
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
    line = line.replace("\r", "\n")
    # Keep only a single logical line at a time
    parts = [p.strip() for p in line.split("\n") if p.strip()]
    return parts[-1] if parts else ""


def _parse_progress_from_text(text: str):
    """Extract current/total progress from text like '9/20'."""
    if not text:
        return None, None
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _update_progress_from_runlog(results_path: str):
    """Fallback progress update by reading txt/run.log while process is running."""
    if not results_path:
        return
    run_log = os.path.join(results_path, "txt", "run.log")
    if not os.path.exists(run_log):
        return
    try:
        with open(run_log, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        latest_iter = None
        for line in lines:
            m = re.search(r"^\s*Iteration:\s*(\d+)", line)
            if m:
                latest_iter = int(m.group(1))
        if latest_iter is not None:
            # Iterations are typically zero-indexed in logs; UI is one-indexed
            shown_iter = latest_iter + 1
            total = experiment_state.get("total_iters", 0)
            if total > 0:
                shown_iter = min(shown_iter, total)
            experiment_state["current_iter"] = max(
                experiment_state.get("current_iter", 0), shown_iter
            )
    except Exception:
        pass


def run_experiment_background(config_path):
    """Run experiment in background and update state."""
    global experiment_state

    try:
        # Determine Python executable - prefer current environment's python
        python_exe = sys.executable

        # Check if we're in base/wrong environment, try to find active_sensing
        if "base" in python_exe or "numpy" not in sys.modules:
            # Try to find active_sensing conda environment
            possible_paths = [
                os.path.expanduser("~/miniconda3/envs/active_sensing/bin/python"),
                os.path.expanduser("~/anaconda3/envs/active_sensing/bin/python"),
                "/home/bota/miniconda3/envs/active_sensing/bin/python",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    python_exe = path
                    print(f"Using Python from active_sensing environment: {python_exe}")
                    break

        # Run main.py with config
        cmd = [python_exe, "src/main.py", "--config", config_path]

        experiment_state["log_output"].append(f"Running command: {' '.join(cmd)}")
        experiment_state["log_output"].append(f"Python: {python_exe}")

        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=PROJECT_ROOT,
            env=run_env,
        )

        experiment_state["log_output"].append(
            f"[INFO] Process started with PID: {process.pid}"
        )

        # Store process handle for stopping
        experiment_state["process"] = process

        # Monitor output from both stdout and stderr
        import select

        last_runlog_poll = time.time()
        stderr_tail = []

        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [], 1.0)

            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    raw_line = process.stdout.readline()
                    line = _clean_stream_line(raw_line)
                    if line:
                        # Check if this is a debug log
                        is_debug = _is_debug_log(line)

                        # If debug logs enabled, write to file
                        if experiment_state.get(
                            "debug_logs_enabled"
                        ) and experiment_state.get("debug_log_file"):
                            try:
                                with open(experiment_state["debug_log_file"], "a") as f:
                                    f.write(line + "\n")
                            except Exception:
                                pass  # Don't fail if debug log write fails

                        # Only add to log_output (shown in HTML) if not a debug log OR if debug enabled
                        if not is_debug or experiment_state.get("debug_logs_enabled"):
                            experiment_state["log_output"].append(line)

                        # Capture explicit results path from main logger output
                        if line.startswith("Logging to:"):
                            log_path = line.split("Logging to:", 1)[1].strip()
                            if log_path:
                                # .../trials/<run>/logs/main_*.log -> .../trials/<run>
                                results_path = os.path.dirname(
                                    os.path.dirname(log_path)
                                )
                                if os.path.isdir(results_path):
                                    experiment_state["results_path"] = results_path
                                    experiment_state["log_output"].append(
                                        f"[INFO] Results path detected: {results_path}"
                                    )

                        # Try to parse iteration progress (NOT steps!)
                        # tqdm format: "Iters (e=None):  45%|████▌     | 9/20 [00:52<00:34,  3.15s/it]"
                        # Ignore lines like "steps:  20%|██        | 3/15" - we want iterations only
                        if "iters" in line.lower() and "iter" in line.lower():
                            # Only parse if line contains "Iters" (iteration tracking), not "steps"
                            if "steps" not in line.lower():
                                try:
                                    current, total = _parse_progress_from_text(line)
                                    if current is not None and total is not None:
                                        experiment_state["current_iter"] = current
                                        experiment_state["total_iters"] = total
                                except Exception as e:
                                    pass  # Ignore parsing errors

                        # Track results path from output
                        if "trials" in line.lower() or "saving" in line.lower():
                            experiment_state["log_output"].append(
                                f"[INFO] {line.strip()}"
                            )

                if fd == process.stderr.fileno():
                    raw_line = process.stderr.readline()
                    line = _clean_stream_line(raw_line)
                    if line:
                        # tqdm often writes to stderr; parse iteration progress (not steps!)
                        if "iters" in line.lower() and "steps" not in line.lower():
                            current, total = _parse_progress_from_text(line)
                            if current is not None and total is not None:
                                experiment_state["current_iter"] = current
                                experiment_state["total_iters"] = total

                        low = line.lower()
                        # Don't mark tqdm progress bars as errors
                        is_progress_bar = any(
                            x in line for x in ["|", "%|", "it/s", "s/it"]
                        )

                        if not is_progress_bar and (
                            "error" in low or "exception" in low or "traceback" in low
                        ):
                            msg = f"[ERROR] {line}"
                            stderr_tail.append(line)
                            stderr_tail = stderr_tail[-25:]
                        else:
                            msg = f"[STDERR] {line}"

                        experiment_state["log_output"].append(msg)

            # Fallback: poll run.log for iteration updates while running
            if time.time() - last_runlog_poll >= 1.0:
                _update_progress_from_runlog(experiment_state.get("results_path"))
                last_runlog_poll = time.time()

            # Check if process finished
            if process.poll() is not None:
                break

        # Get remaining output
        for line in process.stdout.readlines():
            clean = _clean_stream_line(line)
            if clean:
                experiment_state["log_output"].append(clean)

        for line in process.stderr.readlines():
            clean = _clean_stream_line(line)
            if clean:
                low = clean.lower()
                is_progress_bar = any(x in clean for x in ["|", "%|", "it/s", "s/it"])
                if not is_progress_bar and (
                    "error" in low or "exception" in low or "traceback" in low
                ):
                    experiment_state["log_output"].append(f"[ERROR] {clean}")
                    stderr_tail.append(clean)
                    stderr_tail = stderr_tail[-25:]
                else:
                    experiment_state["log_output"].append(f"[STDERR] {clean}")

        returncode = process.wait()
        temp_run_path = experiment_state.get("temp_run_path")
        run_dirname = experiment_state.get("run_dirname")
        method = experiment_state.get("metadata", {}).get("method", "unknown")

        # Close debug log file if it exists
        if experiment_state.get("debug_log_file"):
            try:
                with open(experiment_state["debug_log_file"], "a") as f:
                    f.write(
                        f"\n\nExperiment finished at: {datetime.now().isoformat()}\n"
                    )
            except Exception:
                pass

        if returncode != 0:
            # SIGTERM (-15) is expected from stop button, not an error
            if returncode == -15:
                experiment_state["log_output"].append(
                    "[INFO] Experiment stopped by user"
                )
                # Move to failed directory
                if temp_run_path and os.path.exists(temp_run_path):
                    failed_dir = os.path.join(PROJECT_ROOT, "experiments", "failed")
                    os.makedirs(failed_dir, exist_ok=True)
                    failed_path = os.path.join(failed_dir, run_dirname)
                    try:
                        import shutil

                        shutil.move(temp_run_path, failed_path)
                        experiment_state["log_output"].append(
                            f"[INFO] Moved to failed: {failed_path}"
                        )
                        # Update metadata
                        metadata_path = os.path.join(failed_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                meta = json.load(f)
                            meta["status"] = "stopped"
                            meta["end_time"] = datetime.now().isoformat()
                            with open(metadata_path, "w") as f:
                                json.dump(meta, f, indent=2)
                    except Exception as e:
                        experiment_state["log_output"].append(
                            f"[ERROR] Failed to move to failed dir: {e}"
                        )
            else:
                detailed_error = f"Process exited with code {returncode}"
                if stderr_tail:
                    # Filter out tqdm from error tail
                    actual_errors = [
                        l
                        for l in stderr_tail
                        if not any(x in l for x in ["|", "%|", "it/s", "s/it"])
                    ]
                    if actual_errors:
                        detailed_error += f": {actual_errors[-1]}"
                experiment_state["error"] = detailed_error
                experiment_state["log_output"].append(
                    f"[ERROR] Process failed with code {returncode}"
                )
                if stderr_tail:
                    actual_errors = [
                        l
                        for l in stderr_tail[-10:]
                        if not any(x in l for x in ["|", "%|", "it/s", "s/it"])
                    ]
                    if actual_errors:
                        experiment_state["log_output"].append(
                            "[ERROR] Last stderr lines:"
                        )
                        experiment_state["log_output"].extend(
                            [f"[ERROR] {line}" for line in actual_errors]
                        )
                # Move to failed directory
                if temp_run_path and os.path.exists(temp_run_path):
                    failed_dir = os.path.join(PROJECT_ROOT, "experiments", "failed")
                    os.makedirs(failed_dir, exist_ok=True)
                    failed_path = os.path.join(failed_dir, run_dirname)
                    try:
                        import shutil

                        shutil.move(temp_run_path, failed_path)
                        experiment_state["log_output"].append(
                            f"[INFO] Moved to failed: {failed_path}"
                        )
                        # Update metadata
                        metadata_path = os.path.join(failed_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                meta = json.load(f)
                            meta["status"] = "failed"
                            meta["end_time"] = datetime.now().isoformat()
                            meta["error"] = detailed_error
                            with open(metadata_path, "w") as f:
                                json.dump(meta, f, indent=2)
                    except Exception as e:
                        experiment_state["log_output"].append(
                            f"[ERROR] Failed to move to failed dir: {e}"
                        )
        else:
            # Success! Move from temp to runs/<method>/
            if temp_run_path and os.path.exists(temp_run_path):
                runs_dir = os.path.join(PROJECT_ROOT, "experiments", "runs", method)
                os.makedirs(runs_dir, exist_ok=True)
                final_path = os.path.join(runs_dir, run_dirname)
                try:
                    import shutil

                    shutil.move(temp_run_path, final_path)
                    experiment_state["results_path"] = final_path
                    experiment_state["log_output"].append(
                        f"[SUCCESS] Results saved to: {final_path}"
                    )
                    # Update metadata
                    metadata_path = os.path.join(final_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)
                        meta["status"] = "completed"
                        meta["end_time"] = datetime.now().isoformat()
                        with open(metadata_path, "w") as f:
                            json.dump(meta, f, indent=2)
                except Exception as e:
                    experiment_state["log_output"].append(
                        f"[ERROR] Failed to move to runs dir: {e}"
                    )
                    # Fall back to old results path finder
                    results_path = find_results_path(experiment_state["config"])
                    experiment_state["results_path"] = results_path
            else:
                # Fallback for old structure
                results_path = find_results_path(experiment_state["config"])
                experiment_state["results_path"] = results_path
                if results_path:
                    experiment_state["log_output"].append(
                        f"[SUCCESS] Results path: {results_path}"
                    )
                    print(f"Results path set to: {results_path}")

    except Exception as e:
        experiment_state["error"] = str(e)
        experiment_state["log_output"].append(f"[EXCEPTION] {str(e)}")
        print(f"Exception in run_experiment_background: {e}")
        import traceback

        traceback.print_exc()

    finally:
        experiment_state["running"] = False
        experiment_state["process"] = None
        experiment_state["log_output"].append("[INFO] Experiment finished")


def create_run_dirname(metadata):
    """Create directory name with timestamp and critical params only."""
    timestamp = metadata["timestamp"]
    method = metadata["method"]
    num_agents = metadata["num_agents"]
    mode = metadata["mode"]
    iters = metadata["iters"]

    # Extract communication range if available
    comm = metadata.get("comm_range", "R5")

    # Format: run_<timestamp>_<method>_N<agents>_<mode>_comm<range>_i<iters>
    dirname = f"run_{timestamp}_{method}_N{num_agents}_{mode}_comm{comm}_i{iters}"
    return dirname


def find_results_path(config):
    """Find the path to experiment results (checks both trials/ and temp directories)."""
    # Results can be in multiple locations:
    # 1. PROJECT_ROOT/trials/ (old structure)
    # 2. experiments/temp/<run_name>/trials/ (current run)

    search_dirs = [
        os.path.join(PROJECT_ROOT, "trials"),
        os.path.join(PROJECT_ROOT, "experiments", "temp"),
    ]

    all_results = []

    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue

        # For temp directory, search recursively for trials subdirs
        if "temp" in base_dir:
            try:
                for run_dir in os.listdir(base_dir):
                    run_path = os.path.join(base_dir, run_dir)
                    if os.path.isdir(run_path):
                        trials_subdir = os.path.join(run_path, "trials")
                        if os.path.exists(trials_subdir):
                            # Look inside trials subdirectory
                            for trial in os.listdir(trials_subdir):
                                trial_path = os.path.join(trials_subdir, trial)
                                if os.path.isdir(trial_path):
                                    has_results = os.path.exists(
                                        os.path.join(trial_path, "txt")
                                    )
                                    if has_results:
                                        mtime = os.path.getmtime(trial_path)
                                        all_results.append((trial_path, mtime))
            except Exception as e:
                print(f"Error searching {base_dir}: {e}")
                continue
        else:
            # For trials directory, look at top level
            try:
                for entry in os.listdir(base_dir):
                    full_path = os.path.join(base_dir, entry)
                    if os.path.isdir(full_path):
                        has_results = os.path.exists(os.path.join(full_path, "txt"))
                        if has_results:
                            mtime = os.path.getmtime(full_path)
                            all_results.append((full_path, mtime))
            except Exception as e:
                print(f"Error searching {base_dir}: {e}")
                continue

    if all_results:
        # Return most recent directory with results
        all_results.sort(key=lambda x: x[1], reverse=True)
        selected_path = all_results[0][0]
        print(f"Found results path: {selected_path}")
        return selected_path

    print("No results directories found")
    return None


def generate_comparison_plots(run_paths, include_timing=True, timing_metric="Mean_ms"):
    """Generate multi-method comparison plots including timing data.

    Args:
        run_paths: List of run directory paths to compare
        include_timing: Whether to include timing comparison plot
        timing_metric: Metric to use for timing (Mean_ms, Median_ms, P95_ms, P99_ms)

    Returns:
        Dictionary with plot URLs for {metrics, timing}
    """
    import sys
    import pandas as pd
    import numpy as np

    # Import plotter functions
    plotter_path = os.path.join(PROJECT_ROOT, "plotter.py")
    if os.path.exists(plotter_path):
        spec = importlib.util.spec_from_file_location("plotter", plotter_path)
        plotter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plotter)
    else:
        return None

    plot_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_urls = {}

    # 1. Generate metrics comparison (MSE, Coverage, etc.)
    try:
        # Aggregate data from all selected runs
        all_stats = pd.DataFrame()
        method_map = {}

        for run_path in run_paths:
            txt_path = os.path.join(run_path, "txt")
            if os.path.exists(txt_path):
                try:
                    stats = plotter.aggregate_data_by_settings(txt_path)

                    run_name = os.path.basename(os.path.normpath(run_path))
                    run_info = extract_run_info(run_name, run_path, status="completed")
                    method = run_info.get(
                        "comparison_label"
                    ) or extract_method_from_path(run_path)
                    stats["Method"] = method

                    all_stats = pd.concat([all_stats, stats], ignore_index=True)
                    method_map[method] = run_path
                except Exception as e:
                    print(f"Warning: Failed to aggregate {txt_path}: {e}")

        if not all_stats.empty:
            # Generate method comparison plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Get first radius and pairwise values
            radius = (
                str(all_stats["GaussianRadius"].iloc[0])
                if "GaussianRadius" in all_stats.columns
                else "4"
            )
            pairwise = (
                str(all_stats["Pairwise"].iloc[0])
                if "Pairwise" in all_stats.columns
                else "equal"
            )

            plotter.plot_method_comparison(
                all_stats, radius, plot_dir, show=False, pairwise=pairwise
            )

            plot_urls["metrics"] = (
                f"/api/plot/method_comparison_r_{radius}_news_{all_stats['NewsMode'].iloc[0] if 'NewsMode' in all_stats.columns else 'IG'}_pairwise_{pairwise}.png?t={timestamp}"
            )
    except Exception as e:
        print(f"Error generating metrics comparison: {e}")
        import traceback

        traceback.print_exc()

    # 2. Generate timing comparison if requested
    if include_timing:
        try:
            timing_data_dict = {}
            hlp_timing_dict = {}
            llp_timing_dict = {}
            has_hierarchical = False

            for run_path in run_paths:
                run_name = os.path.basename(os.path.normpath(run_path))
                run_info = extract_run_info(run_name, run_path, status="completed")
                method = run_info.get("comparison_label") or extract_method_from_path(
                    run_path
                )

                # Overall timing
                csv_path = os.path.join(run_path, "txt", "timestamps.csv")
                if os.path.exists(csv_path):
                    timing_df = plotter.parse_timing_csv(csv_path)
                    if not timing_df.empty:
                        timing_data_dict[method] = timing_df

                # HLP timing (hierarchical methods only)
                hlp_csv_path = os.path.join(run_path, "txt", "timestamps_hlp.csv")
                if os.path.exists(hlp_csv_path):
                    hlp_df = plotter.parse_timing_csv(hlp_csv_path)
                    if not hlp_df.empty:
                        hlp_timing_dict[method] = hlp_df
                        has_hierarchical = True

                # LLP timing (hierarchical methods only)
                llp_csv_path = os.path.join(run_path, "txt", "timestamps_llp.csv")
                if os.path.exists(llp_csv_path):
                    llp_df = plotter.parse_timing_csv(llp_csv_path)
                    if not llp_df.empty:
                        llp_timing_dict[method] = llp_df

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Overall timing plot
            if timing_data_dict:
                plotter.plot_timing_comparison(
                    timing_data_dict,
                    plot_dir,
                    show=False,
                    metric=timing_metric,
                    timing_type="overall",
                )
                plot_urls["timing"] = (
                    f"/api/plot/timing_comparison_{timing_metric.lower()}.png?t={timestamp}"
                )

            # HLP timing plot (if hierarchical methods present)
            if hlp_timing_dict:
                plotter.plot_timing_comparison(
                    hlp_timing_dict,
                    plot_dir,
                    show=False,
                    metric=timing_metric,
                    timing_type="hlp",
                )
                plot_urls["timing_hlp"] = (
                    f"/api/plot/timing_comparison_hlp_{timing_metric.lower()}.png?t={timestamp}"
                )

            # LLP timing plot (if hierarchical methods present)
            if llp_timing_dict:
                plotter.plot_timing_comparison(
                    llp_timing_dict,
                    plot_dir,
                    show=False,
                    metric=timing_metric,
                    timing_type="llp",
                )
                plot_urls["timing_llp"] = (
                    f"/api/plot/timing_comparison_llp_{timing_metric.lower()}.png?t={timestamp}"
                )

        except Exception as e:
            print(f"Error generating timing comparison: {e}")
            import traceback

            traceback.print_exc()

    return plot_urls if plot_urls else None


def extract_method_from_path(run_path):
    """Extract method name from run path."""
    # Method is either in path or can be extracted from directory name
    # Check most specific patterns first
    if "greedy_ig" in run_path:
        return "Greedy-IG"
    elif "mh_dec_mcts_both" in run_path:
        return "MH-Both"
    elif "mh_dec_mcts_full" in run_path:
        return "MH-Full"
    elif "mh_dec_mcts_efficient" in run_path:
        return "MH-Eff"
    elif "mh_dec_mcts" in run_path:
        # Plain mh_dec_mcts defaults to efficient
        return "MH-Eff"
    elif "dec_mcts" in run_path:
        return "Dec-MCTS"
    else:
        # Try to extract from directory name
        parts = run_path.split(os.sep)
        for part in parts:
            if "greedy_ig" in part:
                return "Greedy-IG"
            elif "mh_dec_mcts_both" in part:
                return "MH-Both"
            elif "mh_dec_mcts_full" in part:
                return "MH-Full"
            elif "mh_dec_mcts_efficient" in part:
                return "MH-Eff"
            elif "mh_dec_mcts" in part:
                # Plain mh_dec_mcts defaults to efficient
                return "MH-Eff"
            elif "dec_mcts" in part:
                return "Dec-MCTS"
        return "Unknown"


def generate_plots(results_path):
    """Generate plots using plotter.py for the given results."""
    if not results_path or not os.path.exists(results_path):
        return None

    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Use experiment run ID from metadata or directory name
    run_id = None
    metadata_path = os.path.join(results_path, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            run_id = meta.get("timestamp")
        except:
            pass

    # Fallback to directory name if metadata not available
    if not run_id:
        run_id = os.path.basename(results_path)
        # Extract timestamp from directory name if it follows pattern
        if run_id.startswith("run_"):
            parts = run_id.split("_")
            if len(parts) >= 2:
                run_id = f"{parts[1]}_{parts[2]}" if len(parts) > 2 else parts[1]

    output_file = f"run_{run_id}.png"
    output_path = os.path.join(output_dir, output_file)

    # Try simple plot first (more reliable)
    try:
        result = create_simple_plot(results_path, output_path)
        if result:
            return output_file  # Return the filename directly
    except Exception as e:
        import traceback

        traceback.print_exc()

    # Try using plotter.py as fallback
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import plotter

        # Aggregate data from the results path
        stats = plotter.aggregate_data_by_settings(results_path)

        if stats is not None and not stats.empty:
            # Determine strategy name
            strategy = None
            if "Strategy" in stats.columns:
                strategies = stats["Strategy"].unique()
                strategy = strategies[0] if len(strategies) == 1 else "web_experiment"

            # Get radius
            radius = "4"  # Default
            if "GaussianRadius" in stats.columns:
                radii = stats["GaussianRadius"].unique()
                if len(radii) > 0:
                    radius = str(radii[0])

            # Generate plots
            plotter.plot_all_settings(
                stats, radius, output_dir, strategy=strategy, show=False
            )

            # Find the generated plot file
            plot_files = [
                f
                for f in os.listdir(output_dir)
                if f.endswith(".png") and "web_results" not in f
            ]
            if plot_files:
                # Return the most recent one
                plot_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
                    reverse=True,
                )
                print(f"Found plotter-generated file: {plot_files[0]}")
                return plot_files[0]
    except Exception as e:
        print(f"Error using plotter functions: {e}")
        import traceback

        traceback.print_exc()

    print("Could not generate plots")
    return None


def create_simple_plot(results_path, output_path):
    """Create a simple plot from experiment text files."""
    print(f"Creating simple plot from {results_path}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Try to find and parse text files
        txt_dir = os.path.join(results_path, "txt")

        if not os.path.exists(txt_dir):
            print(f"Text directory does not exist: {txt_dir}")
            return None

        # Look for both .txt and .log files
        txt_files = [
            f
            for f in os.listdir(txt_dir)
            if (f.endswith(".txt") or f.endswith(".log"))
            and not f.startswith("timestamps")
        ]
        print(f"Found {len(txt_files)} result files: {txt_files}")

        if not txt_files:
            print("No text files found")
            return None

        # Collect all data from all iterations
        all_data = {
            "steps": [],
            "entropies": [],
            "mses": [],
            "coverages": [],
            "heights": [],
        }

        for txt_file in txt_files:
            file_path = os.path.join(txt_dir, txt_file)
            print(f"Parsing {txt_file}...")

            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()

                # Split file into iterations - each iteration section starts with "Iteration: X"
                iteration_sections = []
                current_section = []
                is_multi_agent = False

                for line in lines:
                    # Check for multi-agent format
                    if "Heights" in line or "Actions" in line:
                        is_multi_agent = True

                    # Check if this is start of new iteration
                    if line.strip().startswith("Iteration:"):
                        if current_section:
                            iteration_sections.append(current_section)
                        current_section = [line]
                    else:
                        current_section.append(line)

                # Don't forget the last section
                if current_section:
                    iteration_sections.append(current_section)

                # If no iteration markers found, treat entire file as one iteration
                if not iteration_sections:
                    iteration_sections = [lines]

                print(f"  Found {len(iteration_sections)} iteration(s) in file")

                # Parse each iteration separately
                for iter_idx, section_lines in enumerate(iteration_sections):
                    # Find data section within this iteration
                    data_start = None
                    for i, line in enumerate(section_lines):
                        if line.strip().startswith("Step"):
                            # Skip header and separator line (----)
                            data_start = i + 1
                            if (
                                i + 1 < len(section_lines)
                                and "---" in section_lines[i + 1]
                            ):
                                data_start = i + 2
                            break

                    if data_start:
                        steps = []
                        entropies = []
                        mses = []
                        coverages = []
                        heights_data = []

                        for line in section_lines[data_start:]:
                            line = line.strip()
                            # Stop if we hit next iteration or empty section
                            if not line or line.startswith("Iteration:"):
                                break
                            if not line[0].isdigit():
                                continue

                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    steps.append(float(parts[0]))
                                    entropies.append(float(parts[1]))
                                    mses.append(float(parts[2]))
                                    coverages.append(float(parts[3]))

                                    # Parse heights if multi-agent format (column 4 has [h1, h2, ...])
                                    if is_multi_agent and len(parts) > 4:
                                        # Extract heights list [h1, h2, h3, h4]
                                        heights_str = (
                                            " ".join(parts[4:])
                                            .split("]")[0]
                                            .replace("[", "")
                                        )
                                        if heights_str:
                                            heights_list = [
                                                float(h.strip(","))
                                                for h in heights_str.split()
                                                if h.strip(",")
                                                .replace(".", "")
                                                .isdigit()
                                            ]
                                            if heights_list:
                                                heights_data.append(
                                                    np.mean(heights_list)
                                                )  # Average across agents
                                except (ValueError, IndexError) as e:
                                    # Skip lines that don't parse correctly
                                    continue

                        if steps:
                            all_data["steps"].append(steps)
                            all_data["entropies"].append(entropies)
                            all_data["mses"].append(mses)
                            all_data["coverages"].append(coverages)
                            all_data["heights"].append(
                                heights_data if heights_data else [0] * len(steps)
                            )
                            print(f"  Iteration {iter_idx}: Parsed {len(steps)} steps")

            except Exception as e:
                print(f"Error parsing {txt_file}: {e}")
                continue

        if not all_data["steps"]:
            print("No data could be parsed from text files")
            return None

        # Average across iterations with std
        print(f"Averaging across {len(all_data['steps'])} iterations...")
        max_len = max(len(s) for s in all_data["steps"])
        steps = list(range(max_len))

        avg_entropies = []
        std_entropies = []
        avg_mses = []
        std_mses = []
        avg_coverages = []
        std_coverages = []
        avg_heights = []
        std_heights = []

        for i in range(max_len):
            ent_vals = [e[i] for e in all_data["entropies"] if i < len(e)]
            mse_vals = [m[i] for m in all_data["mses"] if i < len(m)]
            cov_vals = [c[i] for c in all_data["coverages"] if i < len(c)]
            hgt_vals = [h[i] for h in all_data["heights"] if i < len(h)]

            avg_entropies.append(np.mean(ent_vals) if ent_vals else 0)
            std_entropies.append(np.std(ent_vals) if len(ent_vals) > 1 else 0)
            avg_mses.append(np.mean(mse_vals) if mse_vals else 0)
            std_mses.append(np.std(mse_vals) if len(mse_vals) > 1 else 0)
            avg_coverages.append(np.mean(cov_vals) if cov_vals else 0)
            std_coverages.append(np.std(cov_vals) if len(cov_vals) > 1 else 0)
            avg_heights.append(np.mean(hgt_vals) if hgt_vals else 0)
            std_heights.append(np.std(hgt_vals) if len(hgt_vals) > 1 else 0)

        # Create plots with 2x2 grid + merged bottom row
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.25)

        fig.suptitle(
            f'Experiment Results ({len(all_data["steps"])} iterations)',
            fontsize=16,
            fontweight="bold",
        )

        steps_arr = np.array(steps)

        # Plot entropy with std shading
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps, avg_entropies, "b-", linewidth=2, label="Mean")
        ax1.fill_between(
            steps,
            np.array(avg_entropies) - np.array(std_entropies),
            np.array(avg_entropies) + np.array(std_entropies),
            alpha=0.3,
            color="blue",
            label="±1 Std",
        )
        ax1.set_xlabel("Step", fontsize=11)
        ax1.set_ylabel("Entropy", fontsize=11)
        ax1.set_title("Map Entropy Over Time", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot MSE with std shading
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(steps, avg_mses, "r-", linewidth=2, label="Mean")
        ax2.fill_between(
            steps,
            np.array(avg_mses) - np.array(std_mses),
            np.array(avg_mses) + np.array(std_mses),
            alpha=0.3,
            color="red",
            label="±1 Std",
        )
        ax2.set_xlabel("Step", fontsize=11)
        ax2.set_ylabel("MSE", fontsize=11)
        ax2.set_title("Mean Squared Error", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot coverage with std shading
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(steps, avg_coverages, "g-", linewidth=2, label="Mean")
        ax3.fill_between(
            steps,
            np.array(avg_coverages) - np.array(std_coverages),
            np.array(avg_coverages) + np.array(std_coverages),
            alpha=0.3,
            color="green",
            label="±1 Std",
        )
        ax3.set_xlabel("Step", fontsize=11)
        ax3.set_ylabel("Coverage Ratio", fontsize=11)
        ax3.set_title("Coverage Over Time", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot heights with std shading
        ax4 = fig.add_subplot(gs[1, 1])
        if any(h > 0 for h in avg_heights):
            ax4.plot(steps, avg_heights, "m-", linewidth=2, label="Mean")
            ax4.fill_between(
                steps,
                np.array(avg_heights) - np.array(std_heights),
                np.array(avg_heights) + np.array(std_heights),
                alpha=0.3,
                color="magenta",
                label="±1 Std",
            )
            ax4.set_xlabel("Step", fontsize=11)
            ax4.set_ylabel("Height (m)", fontsize=11)
            ax4.set_title("Average UAV Height", fontsize=12, fontweight="bold")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "Height data not available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax4.set_title("Average UAV Height", fontsize=12, fontweight="bold")

        # Summary stats in merged bottom row
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis("off")
        summary_text = f"""
Experiment Summary

Iterations: {len(all_data['steps'])}  |  Steps per iteration: {max_len}

Final Metrics (mean ± std):
  Entropy:  {avg_entropies[-1]:.4f} ± {std_entropies[-1]:.4f}
  MSE:      {avg_mses[-1]:.4f} ± {std_mses[-1]:.4f}
  Coverage: {avg_coverages[-1]:.4f} ± {std_coverages[-1]:.4f}
  Height:   {avg_heights[-1]:.4f} ± {std_heights[-1]:.4f} m

Results: {os.path.basename(results_path)}
        """
        ax_summary.text(
            0.5,
            0.5,
            summary_text,
            fontsize=11,
            verticalalignment="center",
            horizontalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to {output_path}")
        return True  # Indicate success, filename determined by caller

    except Exception as e:
        print(f"Error creating simple plot: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_multi_run_plot(run_paths, output_path):
    """Create plot overlaying multiple runs with separate curves."""
    print(f"Creating multi-run plot with {len(run_paths)} runs")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Parse data from each run
        runs_data = []
        for run_info in run_paths:
            run_id = run_info["id"]
            run_path = run_info["path"]
            extracted_run_info = extract_run_info(run_id, run_path, status="completed")
            label = extracted_run_info.get("comparison_label") or run_id

            # Look for txt directory or trials subdirectory
            txt_dir = None
            for subdir in ["txt", "trials", "logs"]:
                candidate = os.path.join(run_path, subdir)
                if os.path.exists(candidate):
                    # Check if it has txt files
                    try:
                        files = [
                            f
                            for f in os.listdir(candidate)
                            if f.endswith((".txt", ".log"))
                            and not f.startswith("timestamps")
                        ]
                        if files:
                            txt_dir = candidate
                            break
                    except:
                        continue

            if not txt_dir:
                print(f"No txt directory found for run {run_id}")
                continue

            # Parse data (reuse parsing logic from create_simple_plot)
            all_data = {
                "steps": [],
                "entropies": [],
                "mses": [],
                "coverages": [],
                "heights": [],
            }

            txt_files = [
                f
                for f in os.listdir(txt_dir)
                if f.endswith((".txt", ".log")) and not f.startswith("timestamps")
            ]

            for txt_file in txt_files:
                file_path = os.path.join(txt_dir, txt_file)

                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    # Split by iterations
                    iteration_sections = []
                    current_section = []
                    is_multi_agent = False

                    for line in lines:
                        if "Heights" in line or "Actions" in line:
                            is_multi_agent = True
                        if line.strip().startswith("Iteration:"):
                            if current_section:
                                iteration_sections.append(current_section)
                            current_section = [line]
                        else:
                            current_section.append(line)

                    if current_section:
                        iteration_sections.append(current_section)
                    if not iteration_sections:
                        iteration_sections = [lines]

                    # Parse each iteration
                    for section_lines in iteration_sections:
                        data_start = None
                        for i, line in enumerate(section_lines):
                            if line.strip().startswith("Step"):
                                data_start = i + 1
                                if (
                                    i + 1 < len(section_lines)
                                    and "---" in section_lines[i + 1]
                                ):
                                    data_start = i + 2
                                break

                        if data_start:
                            steps, entropies, mses, coverages, heights_data = (
                                [],
                                [],
                                [],
                                [],
                                [],
                            )

                            for line in section_lines[data_start:]:
                                line = line.strip()
                                if not line or line.startswith("Iteration:"):
                                    break
                                if not line[0].isdigit():
                                    continue

                                parts = line.split()
                                if len(parts) >= 4:
                                    try:
                                        steps.append(float(parts[0]))
                                        entropies.append(float(parts[1]))
                                        mses.append(float(parts[2]))
                                        coverages.append(float(parts[3]))

                                        if is_multi_agent and len(parts) > 4:
                                            heights_str = (
                                                " ".join(parts[4:])
                                                .split("]")[0]
                                                .replace("[", "")
                                            )
                                            if heights_str:
                                                heights_list = [
                                                    float(h.strip(","))
                                                    for h in heights_str.split()
                                                    if h.strip(",")
                                                    .replace(".", "")
                                                    .isdigit()
                                                ]
                                                if heights_list:
                                                    heights_data.append(
                                                        np.mean(heights_list)
                                                    )
                                    except (ValueError, IndexError):
                                        continue

                            if steps:
                                all_data["steps"].append(steps)
                                all_data["entropies"].append(entropies)
                                all_data["mses"].append(mses)
                                all_data["coverages"].append(coverages)
                                all_data["heights"].append(
                                    heights_data if heights_data else [0] * len(steps)
                                )

                except Exception as e:
                    print(f"Error parsing {txt_file}: {e}")
                    continue

            if all_data["steps"]:
                # Average across iterations for this run
                max_len = max(len(s) for s in all_data["steps"])
                steps = list(range(max_len))

                avg_entropies, avg_mses, avg_coverages, avg_heights = [], [], [], []
                std_entropies, std_mses, std_coverages, std_heights = [], [], [], []

                for i in range(max_len):
                    ent_vals = [e[i] for e in all_data["entropies"] if i < len(e)]
                    mse_vals = [m[i] for m in all_data["mses"] if i < len(m)]
                    cov_vals = [c[i] for c in all_data["coverages"] if i < len(c)]
                    hgt_vals = [h[i] for h in all_data["heights"] if i < len(h)]

                    avg_entropies.append(np.mean(ent_vals) if ent_vals else 0)
                    std_entropies.append(np.std(ent_vals) if len(ent_vals) > 1 else 0)
                    avg_mses.append(np.mean(mse_vals) if mse_vals else 0)
                    std_mses.append(np.std(mse_vals) if len(mse_vals) > 1 else 0)
                    avg_coverages.append(np.mean(cov_vals) if cov_vals else 0)
                    std_coverages.append(np.std(cov_vals) if len(cov_vals) > 1 else 0)
                    avg_heights.append(np.mean(hgt_vals) if hgt_vals else 0)
                    std_heights.append(np.std(hgt_vals) if len(hgt_vals) > 1 else 0)

                runs_data.append(
                    {
                        "id": run_id,
                        "label": label,
                        "steps": steps,
                        "avg_entropies": avg_entropies,
                        "std_entropies": std_entropies,
                        "avg_mses": avg_mses,
                        "std_mses": std_mses,
                        "avg_coverages": avg_coverages,
                        "std_coverages": std_coverages,
                        "avg_heights": avg_heights,
                        "std_heights": std_heights,
                        "n_iters": len(all_data["steps"]),
                    }
                )
                print(
                    f"  Run {run_id}: {len(all_data['steps'])} iterations, {max_len} steps"
                )

        if not runs_data:
            print("No valid run data found")
            return None

        # Create 2x2 plot with each run as separate curve
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        fig.suptitle(
            f"Multi-Run Comparison ({len(runs_data)} runs)",
            fontsize=16,
            fontweight="bold",
        )

        # Color palette for different runs
        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # Plot entropy
        ax1 = fig.add_subplot(gs[0, 0])
        for i, run in enumerate(runs_data):
            color = colors[i % len(colors)]
            label = f"{run['label']} ({run['n_iters']} iters)"
            ax1.plot(
                run["steps"],
                run["avg_entropies"],
                color=color,
                linewidth=2,
                label=label,
            )
            ax1.fill_between(
                run["steps"],
                np.array(run["avg_entropies"]) - np.array(run["std_entropies"]),
                np.array(run["avg_entropies"]) + np.array(run["std_entropies"]),
                alpha=0.2,
                color=color,
            )
        ax1.set_xlabel("Step", fontsize=11)
        ax1.set_ylabel("Entropy", fontsize=11)
        ax1.set_title("Map Entropy Over Time", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # Plot MSE
        ax2 = fig.add_subplot(gs[0, 1])
        for i, run in enumerate(runs_data):
            color = colors[i % len(colors)]
            label = f"{run['label']} ({run['n_iters']} iters)"
            ax2.plot(
                run["steps"], run["avg_mses"], color=color, linewidth=2, label=label
            )
            ax2.fill_between(
                run["steps"],
                np.array(run["avg_mses"]) - np.array(run["std_mses"]),
                np.array(run["avg_mses"]) + np.array(run["std_mses"]),
                alpha=0.2,
                color=color,
            )
        ax2.set_xlabel("Step", fontsize=11)
        ax2.set_ylabel("MSE", fontsize=11)
        ax2.set_title("Mean Squared Error", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # Plot coverage
        ax3 = fig.add_subplot(gs[1, 0])
        for i, run in enumerate(runs_data):
            color = colors[i % len(colors)]
            label = f"{run['label']} ({run['n_iters']} iters)"
            ax3.plot(
                run["steps"],
                run["avg_coverages"],
                color=color,
                linewidth=2,
                label=label,
            )
            ax3.fill_between(
                run["steps"],
                np.array(run["avg_coverages"]) - np.array(run["std_coverages"]),
                np.array(run["avg_coverages"]) + np.array(run["std_coverages"]),
                alpha=0.2,
                color=color,
            )
        ax3.set_xlabel("Step", fontsize=11)
        ax3.set_ylabel("Coverage", fontsize=11)
        ax3.set_title("Coverage Ratio", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # Plot heights
        ax4 = fig.add_subplot(gs[1, 1])
        for i, run in enumerate(runs_data):
            color = colors[i % len(colors)]
            label = f"{run['label']} ({run['n_iters']} iters)"
            if any(h > 0 for h in run["avg_heights"]):
                ax4.plot(
                    run["steps"],
                    run["avg_heights"],
                    color=color,
                    linewidth=2,
                    label=label,
                )
                ax4.fill_between(
                    run["steps"],
                    np.array(run["avg_heights"]) - np.array(run["std_heights"]),
                    np.array(run["avg_heights"]) + np.array(run["std_heights"]),
                    alpha=0.2,
                    color=color,
                )
        ax4.set_xlabel("Step", fontsize=11)
        ax4.set_ylabel("Height (m)", fontsize=11)
        ax4.set_title("Average UAV Height", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Multi-run plot saved to {output_path}")
        return True  # Indicate success, filename determined by caller

    except Exception as e:
        print(f"Error creating multi-run plot: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("trials", exist_ok=True)

    print("=" * 70)
    print("Starting MCTS Experiment Web Interface...")
    print("=" * 70)

    # Check Python environment
    python_exe = sys.executable
    print(f"Python: {python_exe}")

    # Check for required packages
    missing_packages = []
    try:
        import numpy

        print("✓ NumPy available")
    except ImportError:
        missing_packages.append("numpy")
        print("✗ NumPy not available - experiments will fail!")

    try:
        import matplotlib

        print("✓ Matplotlib available")
    except ImportError:
        missing_packages.append("matplotlib")
        print("✗ Matplotlib not available - plotting will fail!")

    try:
        import pandas

        print("✓ Pandas available")
    except ImportError:
        missing_packages.append("pandas")
        print("✗ Pandas not available - analysis will fail!")

    if missing_packages:
        print("\n" + "!" * 70)
        print("WARNING: Missing required packages!")
        print("!" * 70)
        print("\nYou need to activate the 'active_sensing' conda environment:")
        print("  conda activate active_sensing")
        print("\nOr install missing packages:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\n" + "!" * 70)

    print("\nOpen http://localhost:5000 in your browser")
    print("=" * 70 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)

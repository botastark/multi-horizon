#!/usr/bin/env python3
"""
Experiment Results Browser

Browse and analyze past experiments from the web interface.
Shows experiment metadata and allows easy comparison.
"""

import os
import json
from datetime import datetime
from pathlib import Path


def parse_experiment_metadata(exp_dir):
    """Extract metadata from experiment directory and logs."""
    metadata = {
        "path": exp_dir,
        "name": os.path.basename(exp_dir),
        "timestamp": None,
        "method": None,
        "num_agents": None,
        "mode": None,
        "modified": os.path.getmtime(exp_dir),
    }

    # Try to parse from directory name
    # Format: method_N{agents}_{mode}_{timestamp} or legacy format
    parts = metadata["name"].split("_")

    # Check for timestamp in name (YYYYMMDD_HHMMSS format)
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():  # YYYYMMDD
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                metadata["timestamp"] = f"{part}_{parts[i + 1]}"
                try:
                    dt = datetime.strptime(metadata["timestamp"], "%Y%m%d_%H%M%S")
                    metadata["datetime"] = dt
                except:
                    pass

    # Extract method (first part usually)
    if parts:
        metadata["method"] = parts[0]

    # Extract number of agents
    for part in parts:
        if part.startswith("N") and part[1:].isdigit():
            metadata["num_agents"] = int(part[1:])
            break

    # Extract mode (IG, IGd, IGd_BM, etc.)
    for part in parts:
        if part in ["IG", "IGd", "IG_BS", "IG_BM", "IGd_BS", "IGd_BM"]:
            metadata["mode"] = part
            break

    # Check for log files to get more info
    logs_dir = os.path.join(exp_dir, "logs")
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
        metadata["log_files"] = log_files

        # Parse first log file for additional metadata
        if log_files:
            log_path = os.path.join(logs_dir, log_files[0])
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()[:50]  # Read first 50 lines

                for line in lines:
                    if "Strategy:" in line:
                        metadata["strategy"] = line.split(":")[1].strip()
                    elif "Num agents:" in line:
                        metadata["num_agents"] = int(line.split(":")[1].strip())
                    elif "News mode:" in line:
                        metadata["mode"] = line.split(":")[1].strip()
            except:
                pass

    # Check for text results
    txt_dir = os.path.join(exp_dir, "txt")
    if os.path.exists(txt_dir):
        txt_files = [
            f
            for f in os.listdir(txt_dir)
            if (f.endswith(".txt") or f.endswith(".log"))
            and not f.startswith("timestamps")
        ]
        metadata["result_files"] = len(txt_files)

    return metadata


def list_experiments(trials_dir="experiments/runs", sort_by="timestamp"):
    """List all experiments in the experiments/runs directory."""
    if not os.path.exists(trials_dir):
        print(f"Experiments directory not found: {trials_dir}")
        return []

    experiments = []

    for entry in os.listdir(trials_dir):
        exp_path = os.path.join(trials_dir, entry)
        if os.path.isdir(exp_path):
            # Check if it has results
            has_results = os.path.exists(
                os.path.join(exp_path, "txt")
            ) or os.path.exists(os.path.join(exp_path, "logs"))
            if has_results:
                metadata = parse_experiment_metadata(exp_path)
                experiments.append(metadata)

    # Sort experiments
    if sort_by == "timestamp" and any(e.get("datetime") for e in experiments):
        experiments.sort(key=lambda x: x.get("datetime") or datetime.min, reverse=True)
    elif sort_by == "modified":
        experiments.sort(key=lambda x: x.get("modified", 0), reverse=True)
    elif sort_by == "name":
        experiments.sort(key=lambda x: x.get("name", ""))

    return experiments


def filter_experiments(experiments, method=None, num_agents=None, mode=None):
    """Filter experiments by criteria."""
    filtered = experiments

    if method:
        filtered = [
            e for e in filtered if method.lower() in e.get("method", "").lower()
        ]

    if num_agents is not None:
        filtered = [e for e in filtered if e.get("num_agents") == num_agents]

    if mode:
        filtered = [e for e in filtered if e.get("mode") == mode]

    return filtered


def print_experiments_table(experiments):
    """Print experiments in a nice table format."""
    if not experiments:
        print("No experiments found.")
        return

    print("\n" + "=" * 120)
    print(
        f"{'#':<4} {'Method':<20} {'Agents':<7} {'Mode':<10} {'Timestamp':<18} {'Results':<8} {'Directory':<40}"
    )
    print("=" * 120)

    for i, exp in enumerate(experiments, 1):
        method = exp.get("method", "unknown")[:20]
        agents = exp.get("num_agents", "?")
        mode = exp.get("mode", "?")[:10]

        if exp.get("datetime"):
            ts = exp["datetime"].strftime("%Y-%m-%d %H:%M")
        elif exp.get("timestamp"):
            ts = exp["timestamp"]
        else:
            ts = "-"

        results = exp.get("result_files", 0)
        name = exp.get("name", "")[:40]

        print(
            f"{i:<4} {method:<20} {agents:<7} {mode:<10} {ts:<18} {results:<8} {name:<40}"
        )

    print("=" * 120)
    print(f"Total: {len(experiments)} experiments\n")


def get_experiment_summary(exp_metadata):
    """Get detailed summary of an experiment."""
    print("\n" + "=" * 80)
    print(f"Experiment: {exp_metadata['name']}")
    print("=" * 80)

    print(f"Path: {exp_metadata['path']}")

    if exp_metadata.get("datetime"):
        print(f"Date: {exp_metadata['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Method: {exp_metadata.get('method', 'unknown')}")
    print(f"Strategy: {exp_metadata.get('strategy', 'unknown')}")
    print(f"Agents: {exp_metadata.get('num_agents', '?')}")
    print(f"Mode: {exp_metadata.get('mode', '?')}")
    print(f"Result files: {exp_metadata.get('result_files', 0)}")

    if exp_metadata.get("log_files"):
        print(f"Log files: {', '.join(exp_metadata['log_files'][:3])}")

    print("=" * 80 + "\n")


def compare_experiments(exp_list):
    """Compare multiple experiments side by side."""
    if not exp_list:
        print("No experiments to compare")
        return

    print("\n" + "=" * 100)
    print("Experiment Comparison")
    print("=" * 100)

    attributes = ["name", "method", "num_agents", "mode", "timestamp", "result_files"]

    for attr in attributes:
        print(f"\n{attr.upper()}:")
        for i, exp in enumerate(exp_list, 1):
            value = exp.get(attr, "N/A")
            print(f"  [{i}] {value}")

    print("\n" + "=" * 100 + "\n")


def main():
    """Main interactive browser."""
    import argparse

    parser = argparse.ArgumentParser(description="Browse experiment results")
    parser.add_argument("--trials-dir", default="experiments/runs", help="Experiments directory path")
    parser.add_argument("--method", help="Filter by method")
    parser.add_argument("--agents", type=int, help="Filter by number of agents")
    parser.add_argument("--mode", help="Filter by mode (IG, IGd, IGd_BM, etc.)")
    parser.add_argument(
        "--sort",
        choices=["timestamp", "modified", "name"],
        default="timestamp",
        help="Sort order",
    )
    parser.add_argument(
        "--detail", type=int, help="Show detailed info for experiment #"
    )
    parser.add_argument(
        "--compare", nargs="+", type=int, help="Compare experiments by their numbers"
    )

    args = parser.parse_args()

    # List all experiments
    experiments = list_experiments(args.trials_dir, sort_by=args.sort)

    # Filter if requested
    if args.method or args.agents or args.mode:
        experiments = filter_experiments(
            experiments, method=args.method, num_agents=args.agents, mode=args.mode
        )

    # Show table
    print_experiments_table(experiments)

    # Show detail if requested
    if args.detail:
        if 1 <= args.detail <= len(experiments):
            get_experiment_summary(experiments[args.detail - 1])
        else:
            print(f"Invalid experiment number: {args.detail}")

    # Compare if requested
    if args.compare:
        exp_to_compare = []
        for num in args.compare:
            if 1 <= num <= len(experiments):
                exp_to_compare.append(experiments[num - 1])
            else:
                print(f"Invalid experiment number: {num}")

        if exp_to_compare:
            compare_experiments(exp_to_compare)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example: Running Multi-Strategy Benchmark

This script demonstrates how to run experiments with the new config system.
"""

import subprocess
import sys
from pathlib import Path


def run_all_strategies():
    """Run all 4 baselines using master config."""
    print("=" * 80)
    print("Running all 4 baseline strategies")
    print("=" * 80)

    cmd = [sys.executable, "src/main.py", "--config", "configs/master_config.json"]

    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_single_strategy(strategy_name):
    """Run a single strategy using its config file."""
    strategy_configs = {
        "greedy_ig": "configs/strategies/greedy_ig.json",
        "dec_mcts": "configs/strategies/dec_mcts.json",
        "mh_full": "configs/strategies/mh_dec_mcts_full.json",
        "mh_efficient": "configs/strategies/mh_dec_mcts_efficient.json",
    }

    if strategy_name not in strategy_configs:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available: {list(strategy_configs.keys())}")
        return

    config_path = strategy_configs[strategy_name]

    print("=" * 80)
    print(f"Running single strategy: {strategy_name}")
    print("=" * 80)

    cmd = [sys.executable, "src/main.py", "--config", config_path]

    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_legacy_config():
    """Run using legacy single-file config."""
    print("=" * 80)
    print("Running with legacy config (backward compatibility)")
    print("=" * 80)

    cmd = [
        sys.executable,
        "src/main.py",
        "--config",
        "configs/benchmark_greedy_ig.json",
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    """Example usage."""
    print("\n" + "=" * 80)
    print("Multi-Strategy Experiment Runner")
    print("=" * 80 + "\n")

    # Example 1: Run all strategies
    print("Example 1: Run all 4 baselines")
    print("-" * 80)
    run_all_strategies()

    # Example 2: Run single strategy
    print("\n\nExample 2: Run single strategy (MH-Dec-MCTS full)")
    print("-" * 80)
    run_single_strategy("mh_full")

    # Example 3: Legacy config
    print("\n\nExample 3: Run with legacy config")
    print("-" * 80)
    run_legacy_config()


if __name__ == "__main__":
    # For manual testing, you can uncomment one of these:

    # Run all strategies
    run_all_strategies()

    # Or run specific strategy
    # run_single_strategy("greedy_ig")
    # run_single_strategy("dec_mcts")
    # run_single_strategy("mh_full")
    # run_single_strategy("mh_efficient")

    # Or run with legacy config
    # run_legacy_config()

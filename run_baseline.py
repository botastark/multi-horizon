#!/usr/bin/env python3
"""
Baseline Experiment Runner - Pre-Refactoring Validation

Runs two baseline configurations to establish ground truth:
1. IG_BS with Rinf (infinite communication)
2. IGd_BM with R=5 (limited communication)

Settings:
- Greedy IG planner
- Gaussian field with r=4
- Adaptive weights
- 20 runs each (100 steps per run)
- No debug plots

Results saved to experiments/baseline/ for comparison after refactoring.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Ensure we're in project root (script is already in root, so just use its directory)
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)


def run_baseline_experiment(config_name, description):
    """Run a single baseline experiment configuration."""
    print(f"\n{'='*70}")
    print(f"🚀 Running: {description}")
    print(f"{'='*70}")

    # Config path relative to current directory (already in project root)
    config_path = os.path.join("configs", f"{config_name}.json")

    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False

    print(f"📄 Using config: {config_path}")

    start_time = time.time()

    try:
        # Run the experiment (use relative path since we're in project root)
        result = subprocess.run(
            [sys.executable, "src/main.py", "--config", config_path],
            check=True,
            capture_output=False,  # Show live output
            text=True,
            cwd=os.getcwd()  # Explicitly use current working directory
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False


def main():
    """Run all baseline experiments."""
    print(f"\n{'#'*70}")
    print("# BASELINE EXPERIMENTS - PRE-REFACTORING VALIDATION")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    experiments = [
        ("baseline_greedy_ig_bs_rinf", "Config 1: IG_BS + Rinf (infinite comm)"),
        ("baseline_greedy_igd_bm_r5", "Config 2: IGd_BM + R=5 (limited comm)"),
    ]

    results = {}

    for config_name, description in experiments:
        success = run_baseline_experiment(config_name, description)
        results[description] = success

    # Summary
    print(f"\n{'='*70}")
    print("📊 BASELINE EXPERIMENTS SUMMARY")
    print(f"{'='*70}")

    for desc, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {desc}")

    # Check if all succeeded
    all_success = all(results.values())

    if all_success:
        print(f"\n🎉 All baseline experiments completed successfully!")
        print(f"📁 Results saved to: experiments/baseline/")
        print(f"\n💡 You can now proceed with refactoring.")
        print(f"   After refactoring, re-run these configs to validate results match.")
    else:
        print(f"\n⚠️  Some experiments failed. Please check the logs.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

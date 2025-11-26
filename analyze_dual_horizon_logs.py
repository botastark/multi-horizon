#!/usr/bin/env python3
"""
Analyze Dual-Horizon Planner Logs

This script parses and analyzes the dual-horizon planner log files to extract
key metrics and patterns.

Usage:
    python analyze_dual_horizon_logs.py <log_file_path>
    python analyze_dual_horizon_logs.py logs/dual_horizon/dual_horizon_experiment_20241126_123456.log
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_log_file(log_path):
    """Parse dual-horizon log file and extract metrics."""
    
    steps = []
    current_step = None
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect step start
            if match := re.match(r'.*STEP (\d+)', line):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'step_num': int(match.group(1)),
                    'metrics': {}
                }
            
            # Extract metrics
            elif current_step:
                if 'Coverage Progress:' in line:
                    val = float(re.search(r'([\d.]+)', line).group(1))
                    current_step['metrics']['coverage_progress'] = val
                
                elif 'Uncertainty Ratio:' in line:
                    val = float(re.search(r'([\d.]+)', line).group(1))
                    current_step['metrics']['uncertainty_ratio'] = val
                
                elif 'Fragmentation Score:' in line:
                    val = float(re.search(r'([\d.]+)', line).group(1))
                    current_step['metrics']['fragmentation_score'] = val
                
                elif 'w_short:' in line:
                    val = float(re.search(r'w_short: ([\d.]+)', line).group(1))
                    current_step['metrics']['w_short'] = val
                
                elif 'w_long:' in line:
                    val = float(re.search(r'w_long: ([\d.]+)', line).group(1))
                    current_step['metrics']['w_long'] = val
                
                elif 'Num Patches:' in line:
                    val = int(re.search(r'(\d+)', line).group(1))
                    current_step['metrics']['num_patches'] = val
                
                elif 'Selected Action:' in line:
                    if 'LLP' in line:
                        action = line.split(':')[-1].strip()
                        current_step['llp_action'] = action
                    elif 'HLP' in line:
                        action = line.split(':')[-1].strip()
                        current_step['hlp_action'] = action
                
                elif 'FINAL SELECTED ACTION:' in line:
                    action = line.split(':')[-1].strip()
                    current_step['final_action'] = action
                
                elif 'Agreement:' in line:
                    agreement = line.split(':')[-1].strip()
                    current_step['agreement'] = agreement
    
    # Add last step
    if current_step:
        steps.append(current_step)
    
    return steps


def analyze_steps(steps):
    """Analyze parsed steps and compute statistics."""
    
    if not steps:
        print("No steps found in log file!")
        return
    
    # Agreement statistics
    agreements = [s.get('agreement', 'UNKNOWN') for s in steps]
    agreement_counts = {
        'YES': agreements.count('YES'),
        'PARTIAL': agreements.count('PARTIAL'),
        'DIFFERENT': agreements.count('DIFFERENT')
    }
    
    total = len(steps)
    
    print("\n" + "="*80)
    print("DUAL-HORIZON PLANNER ANALYSIS")
    print("="*80)
    print(f"\nTotal Steps: {total}")
    
    print("\n--- Agreement Statistics ---")
    for key, count in agreement_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"{key:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Weight evolution
    print("\n--- Blend Weight Evolution ---")
    w_short_vals = [s['metrics'].get('w_short', 0) for s in steps if 'metrics' in s]
    w_long_vals = [s['metrics'].get('w_long', 0) for s in steps if 'metrics' in s]
    
    if w_short_vals:
        print(f"w_short - Mean: {np.mean(w_short_vals):.3f}, Std: {np.std(w_short_vals):.3f}, "
              f"Min: {np.min(w_short_vals):.3f}, Max: {np.max(w_short_vals):.3f}")
    if w_long_vals:
        print(f"w_long  - Mean: {np.mean(w_long_vals):.3f}, Std: {np.std(w_long_vals):.3f}, "
              f"Min: {np.min(w_long_vals):.3f}, Max: {np.max(w_long_vals):.3f}")
    
    # Coverage progress
    print("\n--- Coverage & Fragmentation ---")
    coverage_vals = [s['metrics'].get('coverage_progress', 0) for s in steps if 'metrics' in s]
    frag_vals = [s['metrics'].get('fragmentation_score', 0) for s in steps if 'metrics' in s]
    patch_vals = [s['metrics'].get('num_patches', 0) for s in steps if 'metrics' in s]
    
    if coverage_vals:
        print(f"Coverage - Start: {coverage_vals[0]:.3f}, End: {coverage_vals[-1]:.3f}, "
              f"Gain: {coverage_vals[-1] - coverage_vals[0]:.3f}")
    if frag_vals:
        print(f"Fragmentation Score - Mean: {np.mean(frag_vals):.3f}, Max: {np.max(frag_vals):.3f}")
    if patch_vals:
        print(f"Uncovered Patches - Mean: {np.mean(patch_vals):.1f}, Max: {np.max(patch_vals)}")
    
    # Action agreement patterns
    print("\n--- Action Selection Patterns ---")
    llp_actions = defaultdict(int)
    hlp_actions = defaultdict(int)
    final_actions = defaultdict(int)
    
    for step in steps:
        if 'llp_action' in step:
            llp_actions[step['llp_action']] += 1
        if 'hlp_action' in step:
            hlp_actions[step['hlp_action']] += 1
        if 'final_action' in step:
            final_actions[step['final_action']] += 1
    
    print("\nLLP Action Distribution:")
    for action, count in sorted(llp_actions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:10s}: {count:4d} ({count/total*100:5.1f}%)")
    
    print("\nHLP Action Distribution:")
    for action, count in sorted(hlp_actions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:10s}: {count:4d} ({count/total*100:5.1f}%)")
    
    print("\nFinal Action Distribution:")
    for action, count in sorted(final_actions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:10s}: {count:4d} ({count/total*100:5.1f}%)")
    
    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dual_horizon_logs.py <log_file_path>")
        print("\nExample:")
        print("  python analyze_dual_horizon_logs.py logs/dual_horizon/dual_horizon_experiment_20241126_123456.log")
        
        # Try to find latest log file
        log_dir = Path("logs/dual_horizon")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                print(f"\nLatest log file found: {log_files[0]}")
                print(f"Run: python analyze_dual_horizon_logs.py {log_files[0]}")
        
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing log file: {log_path}")
    
    steps = parse_log_file(log_path)
    analyze_steps(steps)


if __name__ == "__main__":
    main()

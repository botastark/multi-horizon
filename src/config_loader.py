"""
Configuration Loader for Multi-Strategy Experiments

This module loads master config + strategy-specific configs.

Config Structure (v3.0):
    Master config: Shared settings and strategy selection
    Strategy configs: Strategy-specific parameters
"""

import json
import os
import copy
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> List[Dict[str, Any]]:
    """
    Load configuration from master config file or standalone strategy config.

    Args:
        config_file: Path to master config JSON file or standalone strategy config

    Returns:
        List of fully resolved configurations (one per strategy)
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    # Check if this is a master config (has "strategies" key) or standalone strategy config
    if "strategies" in config:
        # Master config v3.0
        version = config.get("_version", "3.0")
        if version != "3.0":
            raise ValueError(f"Only v3.0 configs supported. Found version: {version}")
        return _load_master_config(config, os.path.dirname(config_file))
    else:
        # Standalone strategy config - merge shared section if present
        logger.info(f"Loading standalone strategy config from: {config_file}")
        
        # Default shared values for standalone configs
        default_shared = {
            'project_path': './',
            'field_type': 'Gaussian',
            'cluster_radius': 4,
            'start_position': 'corner',
            'num_agents': 4,
            'n_steps': 15,
            'iters': [0, 20],
            'correlation_types': ['adaptive'],
            'error_margins': [None],
            'enable_plotting': True,
            'enable_logging': True,
            'mode_labels': ['IGd_BM']
        }
        
        # If config has a "shared" section, merge it with defaults
        if "shared" in config:
            shared = {**default_shared, **config.pop("shared")}
        else:
            shared = default_shared
            
        # Merge shared into config (config values take precedence)
        for key, value in shared.items():
            if key not in config:
                config[key] = value
        
        # Flatten hierarchical configs
        config = flatten_hierarchical_config(config)
        return [config]


def _load_master_config(master: Dict[str, Any], base_dir: str) -> List[Dict[str, Any]]:
    """
    Load master config with multiple strategies.

    Args:
        master: Master configuration dict
        base_dir: Base directory for resolving relative paths

    Returns:
        List of fully resolved configurations (one per strategy)
    """
    strategies = master.get("strategies", [])
    strategy_configs = master.get("strategy_configs", {})
    shared = master.get("shared", {})
    shared_decentralized = master.get("decentralized", {})
    shared_experiment = master.get("experiment", {})

    configs = []

    for strategy_name in strategies:
        # Load strategy-specific config
        strategy_config_path = strategy_configs.get(strategy_name)
        if not strategy_config_path:
            logger.warning(f"No config path for strategy '{strategy_name}', skipping")
            continue

        # Resolve relative path
        if not os.path.isabs(strategy_config_path):
            strategy_config_path = os.path.join(base_dir, strategy_config_path)

        if not os.path.exists(strategy_config_path):
            logger.warning(
                f"Strategy config not found: {strategy_config_path}, skipping"
            )
            continue

        # Load strategy config
        with open(strategy_config_path, "r") as f:
            strategy_cfg = json.load(f)

        # Merge: shared <- strategy overrides
        merged = _merge_configs(
            shared, shared_decentralized, shared_experiment, strategy_cfg, master
        )

        # Filter comments
        merged = {k: v for k, v in merged.items() if not k.startswith("_")}

        # Normalize nested strategy configs before the planner sees them.
        merged = flatten_hierarchical_config(merged)

        configs.append(merged)
        logger.info(f"Loaded config for strategy: {strategy_name}")

    return configs


def _merge_configs(
    shared: Dict[str, Any],
    shared_decentralized: Dict[str, Any],
    shared_experiment: Dict[str, Any],
    strategy: Dict[str, Any],
    master: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Merge shared and strategy-specific configs.

    Strategy config takes precedence over shared config.
    """
    merged = copy.deepcopy(shared)
    
    # Copy limited_testing flag from master config if present
    if master and "limited_testing" in master:
        merged["limited_testing"] = master["limited_testing"]

    # Merge decentralized settings
    merged_decentralized = copy.deepcopy(shared_decentralized)
    strategy_decentralized = strategy.get("decentralized", {})
    merged_decentralized.update(strategy_decentralized)
    merged["decentralized"] = merged_decentralized

    # Merge experiment settings
    merged_experiment = copy.deepcopy(shared_experiment)
    strategy_experiment = strategy.get("experiment", {})

    # Special handling for log_dir_suffix
    if "log_dir_suffix" in strategy_experiment:
        base_log_dir = merged_experiment.get("base_log_dir", "trials")
        merged_experiment["log_dir"] = os.path.join(
            base_log_dir, strategy_experiment["log_dir_suffix"]
        )

    # Merge common_metrics + strategy_metrics
    common_metrics = merged_experiment.get("common_metrics", [])
    strategy_metrics = strategy_experiment.get("strategy_metrics", [])
    merged_experiment["metrics"] = common_metrics + strategy_metrics

    merged_experiment.update(strategy_experiment)
    merged["experiment"] = merged_experiment

    # Add strategy-specific sections
    for key, value in strategy.items():
        if key not in ["decentralized", "experiment"]:
            merged[key] = value

    return merged


def flatten_hierarchical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten hierarchical_dec_mcts nested structure for backward compatibility.

    Converts:
        hierarchical_dec_mcts.llp.horizon -> hierarchical_dec_mcts.llp_horizon
        hierarchical_dec_mcts.hlp.horizon -> hierarchical_dec_mcts.hlp_horizon
    """
    hier = config.get("hierarchical_dec_mcts", {})
    if not hier:
        return config

    # If already flat (has llp_horizon), return as-is
    if "llp_horizon" in hier:
        return config

    # Flatten nested structure
    flattened = {}

    # Copy top-level settings
    for key in ["use_mcts_llp", "use_g2", "mode_labels", "intent_sharing"]:
        if key in hier:
            flattened[key] = hier[key]

    # Flatten LLP settings
    llp = hier.get("llp", {})
    for key, value in llp.items():
        if not key.startswith("_"):
            flattened[f"llp_{key}"] = value

    # Flatten HLP settings
    hlp = hier.get("hlp", {})
    for key, value in hlp.items():
        if not key.startswith("_"):
            if key == "replan_interval":
                flattened["hlp_replan_interval"] = value
            elif key == "tile_size":
                flattened["tile_size"] = value
            else:
                flattened[f"hlp_{key}"] = value

    config["hierarchical_dec_mcts"] = flattened
    return config

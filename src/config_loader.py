"""
Configuration Loader with Backward Compatibility

This module provides a clean interface for loading configuration files
while maintaining backward compatibility with the legacy format.

New Config Structure (v2.0):
    - simulation: Core simulation settings
    - strategy: Action strategy selection
    - agents: Multi-agent configuration
    - planner: MCTS planner parameters
    - dual_horizon: LLP + HLP planning settings
    - belief: Belief fusion and LBP settings
    - decentralized: Decentralized agent communication
    - output: Logging and visualization

Legacy format keys are automatically mapped from the new structure.
"""

import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file with backward compatibility.

    Supports both new (v2.0) and legacy config formats.
    New format is automatically converted to include legacy keys.

    Args:
        config_file: Path to config JSON file

    Returns:
        Configuration dict with both new and legacy keys
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    # Check if this is the new format (has _schema_version)
    if config.get("_schema_version") == "2.0":
        config = _convert_v2_to_legacy(config)
    else:
        # Legacy format - just filter comments
        config = {k: v for k, v in config.items() if not k.startswith("_")}

    return config


def _convert_v2_to_legacy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert v2.0 config format to include legacy keys for backward compatibility.

    Args:
        config: New format configuration

    Returns:
        Configuration with both new structure and legacy flat keys
    """
    result = {}

    # Keep original structure (filtered)
    for key, value in config.items():
        if not key.startswith("_"):
            result[key] = value

    # === Map to legacy keys ===

    # Simulation settings
    sim = config.get("simulation", {})
    result["field_type"] = sim.get("field_type", "Gaussian")
    result["start_position"] = sim.get("start_position", "corner")
    result["n_steps"] = sim.get("num_steps", 100)
    result["iters"] = sim.get("iterations", [0, 1])
    result["correlation_types"] = sim.get("correlation_types", ["equal"])
    result["error_margins"] = [
        "None" if e is None else e for e in sim.get("error_margins", [None])
    ]
    result["project_path"] = sim.get("project_path", "./")

    # Strategy
    strategy = config.get("strategy", {})
    result["action_strategy"] = strategy.get("type", "hierarchical_dec_mcts")

    # Output settings
    output = config.get("output", {})
    result["enable_plotting"] = output.get("enable_plotting", True)
    result["enable_logging"] = output.get("enable_logging", True)

    # Build mcts_params (legacy format)
    planner = config.get("planner", {})
    dual = config.get("dual_horizon", {})
    weights = dual.get("weights", {})
    hysteresis = dual.get("hysteresis", {})
    llp = dual.get("llp", {})
    hlp = dual.get("hlp", {})

    result["mcts_params"] = {
        "planning_depth": planner.get("planning_depth", 15),
        "num_iterations": planner.get("num_iterations", 100),
        "ucb1_c": planner.get("ucb1_exploration", 0.95),
        "discount_factor": planner.get("discount_factor", 0.99),
        "timeout": planner.get("timeout_ms", 2000),
        "parallel": planner.get("parallel_simulations", 1),
        "horizon_weights": {
            "w_coverage": weights.get("coverage", 0.1),
            "w_fragmentation": weights.get("fragmentation", 0.5),
            "w_ig": weights.get("information_gain", 0.9),
            "w_distance": weights.get("distance", 0.5),
            "short_horizon_depth": llp.get("horizon", 5),
            "long_horizon_depth": hlp.get("horizon", 20),
            "tile_size": hlp.get("tile_size", [100, 100]),
            "current_target_bonus": hysteresis.get("current_target_bonus", 0.15),
            "coverage_delta_threshold": hysteresis.get(
                "coverage_delta_threshold", 0.05
            ),
            "position_delta_threshold": hysteresis.get("position_delta_threshold", 30),
        },
    }

    # Build hierarchical_dec_mcts (legacy format)
    dec_coord = config.get("decentralized", {}).get("coordination", {})
    belief = config.get("belief", {})
    d_uct = dec_coord.get("d_uct", {})

    result["hierarchical_dec_mcts"] = {
        "llp_horizon": llp.get("horizon", 5),
        "llp_iterations": llp.get("iterations", 100),
        "hlp_horizon": hlp.get("horizon", 3),
        "hlp_iterations": hlp.get("iterations", 50),
        "tile_size": hlp.get("tile_size", [100, 100]),
        "intent_discount": dec_coord.get("intent_discount", 0.8),
        "enable_belief_sharing": belief.get("fusion_enabled", True),
        # D-UCT settings for asynchronous drift handling
        "d_uct_enabled": d_uct.get("enabled", True),
        "d_uct_decay_factor": d_uct.get("decay_factor", 0.9),
        "d_uct_min_visits": d_uct.get("min_visits_before_decay", 5),
        "d_uct_stale_threshold": d_uct.get("stale_intent_threshold_sec", 2.0),
    }

    # Build multi_agent (legacy format)
    agents = config.get("agents", {})
    belief_cfg = config.get("belief", {})
    lbp = belief_cfg.get("lbp", {})

    result["multi_agent"] = {
        "num_agents": agents.get("num_agents", 4),
        "enable_coordination": True,
        "belief_fusion": belief_cfg.get("fusion_enabled", True),
        "fusion_method": belief_cfg.get("fusion_method", "lbp"),
        "use_lbp": lbp.get("enabled", True),
        "lbp_iterations": lbp.get("iterations", 1),
        "news_mode": belief_cfg.get("news_mode", "BM"),
        "pairwise_potential": lbp.get("pairwise_potential", [[0.6, 0.4], [0.4, 0.6]]),
        "region_allocation": "auction",
        "communication_range": agents.get("communication_range", -1),
        "collision_avoidance_distance": agents.get("collision_avoidance_distance", 5.0),
        "belief_fusion_weight": 0.7,
        "coordination_frequency": 5,
    }

    # Build decentralized (legacy format for decentralized_agent.py)
    dec = config.get("decentralized", {})
    sharing = dec.get("sharing", {})
    comm = dec.get("communication", {})
    coord = dec.get("coordination", {})
    d_uct_cfg = coord.get("d_uct", {})

    result["decentralized"] = {
        "enable_belief_fusion": sharing.get("belief_fusion", True),
        "enable_llp_intent_sharing": sharing.get("llp_intent", True),
        "enable_hlp_intent_sharing": sharing.get("hlp_intent", True),
        "enable_position_sharing": sharing.get("position", True),
        "news_mode": belief_cfg.get("news_mode", "BM"),
        "intent_horizon": coord.get("intent_horizon", 5),
        "intent_discount": coord.get("intent_discount", 0.8),
        "overlap_penalty_weight": coord.get("overlap_penalty_weight", 0.3),
        "stale_message_threshold": comm.get("stale_threshold_sec", 10.0),
        "message_queue_size": comm.get("message_queue_size", 100),
        "message_delay": comm.get("message_delay_sec", 0.0),
        "drop_probability": comm.get("drop_probability", 0.0),
        # D-UCT settings
        "d_uct_enabled": d_uct_cfg.get("enabled", True),
        "d_uct_decay_factor": d_uct_cfg.get("decay_factor", 0.9),
        "d_uct_stale_threshold": d_uct_cfg.get("stale_intent_threshold_sec", 2.0),
    }
    # Also provide flattened access for simpler code
    result["use_lbp"] = lbp.get("enabled", True)
    result["lbp_iterations"] = lbp.get("iterations", 1)
    result["communication_range"] = agents.get("communication_range", -1)

    logger.debug("Converted v2.0 config to legacy format")

    return result


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Get a specific section from the config.

    Works with both new and legacy formats.

    Args:
        config: Full configuration dict
        section: Section name (e.g., 'simulation', 'planner', 'decentralized')

    Returns:
        Section configuration dict
    """
    return config.get(section, {})


def get_decentralized_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get decentralized agent configuration.

    Handles both new nested format and legacy flat format.

    Args:
        config: Full configuration dict

    Returns:
        Decentralized configuration with flat keys for agent initialization
    """
    dec = config.get("decentralized", {})

    # If already in flat format (legacy), return as-is
    if "enable_belief_fusion" in dec:
        return dec

    # Convert from nested format
    sharing = dec.get("sharing", {})
    comm = dec.get("communication", {})
    coord = dec.get("coordination", {})

    return {
        "enable_belief_fusion": sharing.get("belief_fusion", True),
        "enable_llp_intent_sharing": sharing.get("llp_intent", True),
        "enable_hlp_intent_sharing": sharing.get("hlp_intent", True),
        "enable_position_sharing": sharing.get("position", True),
        "intent_horizon": coord.get("intent_horizon", 5),
        "overlap_penalty_weight": coord.get("overlap_penalty_weight", 0.3),
        "stale_message_threshold": comm.get("stale_threshold_sec", 10.0),
        "message_queue_size": comm.get("message_queue_size", 100),
        "message_delay": comm.get("message_delay_sec", 0.0),
        "drop_probability": comm.get("drop_probability", 0.0),
    }


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    # Simulation
    sim = config.get("simulation", {})
    print(f"\n[Simulation]")
    print(f"  Field type: {config.get('field_type', sim.get('field_type', 'N/A'))}")
    print(
        f"  Start position: {config.get('start_position', sim.get('start_position', 'N/A'))}"
    )
    print(f"  Steps: {config.get('n_steps', sim.get('num_steps', 'N/A'))}")

    # Strategy
    print(f"\n[Strategy]")
    print(f"  Type: {config.get('action_strategy', 'N/A')}")

    # Agents
    agents = config.get("agents", {})
    multi = config.get("multi_agent", {})
    print(f"\n[Agents]")
    print(f"  Number: {multi.get('num_agents', agents.get('num_agents', 'N/A'))}")
    print(
        f"  Communication range: {multi.get('communication_range', agents.get('communication_range', 'N/A'))}"
    )

    # Decentralized
    dec = config.get("decentralized", {})
    sharing = dec.get("sharing", dec)  # Handle both formats
    print(f"\n[Decentralized Sharing]")
    print(
        f"  Belief fusion: {sharing.get('belief_fusion', sharing.get('enable_belief_fusion', 'N/A'))}"
    )
    print(
        f"  LLP intent: {sharing.get('llp_intent', sharing.get('enable_llp_intent_sharing', 'N/A'))}"
    )
    print(
        f"  HLP intent: {sharing.get('hlp_intent', sharing.get('enable_hlp_intent_sharing', 'N/A'))}"
    )
    print(
        f"  Position: {sharing.get('position', sharing.get('enable_position_sharing', 'N/A'))}"
    )

    print("\n" + "=" * 60 + "\n")

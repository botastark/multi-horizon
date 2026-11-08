"""
Experiment Utilities for UAV Coverage Planning

Common utilities used across single-agent and multi-agent experiments:
- Agent initialization
- Metrics computation
- Logging helpers
- Visualization utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from helper import uav_position
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera



def initialize_agent(
    agent_id: int,
    grid_info,
    start_position: Tuple[float, float],
    action_strategy: str,
    conf_dict: Dict,
    corr_type: str,
    mcts_params: Dict,
    optimal_alt: float,
    min_alt: float,
    overlap: float,
    seed: int,
    coordinator=None,
    start_altitude: Optional[float] = None,
    debug_logs: bool = False,
    experiment_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Initialize a single agent with camera, planner, and state.

    Args:
        agent_id: Unique agent identifier
        grid_info: Grid information object
        start_position: (x, y) starting position
        action_strategy: Planning strategy name
        conf_dict: Configuration dictionary
        corr_type: Correlation type for occupancy map
        mcts_params: MCTS parameters
        optimal_alt: Optimal altitude
        min_alt: Minimum altitude
        overlap: Camera overlap setting
        seed: Random seed
        coordinator: Optional coordinator for multi-agent

    Returns:
        Dict containing agent state (camera, planner, maps, etc.)
    """
    # Create camera
    # rng = np.random.default_rng(seed + agent_id)
    # If a per-agent start altitude is provided, use it; otherwise fall back to min_alt
    cam_alt = start_altitude if start_altitude is not None else min_alt
    camera = Camera(
        grid_info,
        60,
        seed=seed + agent_id,
        camera_altitude=cam_alt,
        f_overlap=overlap,
        s_overlap=overlap,
    )

    # Create planner
    agent_planner = planning(
        grid_info,
        camera,
        action_strategy,
        conf_dict=conf_dict,
        optimal_alt=optimal_alt,
        mcts_params=mcts_params,
        agent_id=agent_id,
        coordinator=coordinator,
        seed=seed + agent_id,
        debug_logs=debug_logs,
        experiment_config=experiment_config,
    )

    # Initialize UAV position
    uav_pos = uav_position((start_position, camera.get_hrange()[0]))
    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)

    # Create occupancy map
    occupancy_map = OM(grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type)

    # Create agent state
    agent_state = {
        "agent_id": agent_id,
        "camera": camera,
        "planner": agent_planner,
        "occupancy_map": occupancy_map,
        "uav_pos": uav_pos,
        "uav_positions": [uav_pos],
        "actions": [],
        "belief_map": np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5),
        "observed_ids": set(),
        "entropy": [],
        "mse": [],
        "coverage": [],
        "height": [],
        "info_gain_action": {},
    }

    return agent_state


def finalize_planners(agents: List[Dict]) -> None:
    """
    Call finalize_episode on all agent planners.

    Args:
        agents: List of agent state dictionaries
    """
    for agent_state in agents:
        planner = agent_state["planner"]
        if hasattr(planner, "finalize_episode"):
            planner.finalize_episode()


def extract_region_metadata(planner, action_strategy: str) -> Tuple:
    """
    Extract region metadata from hierarchical planner if available.

    Args:
        planner: Planner object
        action_strategy: Strategy name

    Returns:
        Tuple of (region_metadata, selected_region_id, region_scores)
    """
    region_metadata = None
    selected_region_id = None
    region_scores = None

    if action_strategy in (
        "mh_dec_mcts_efficient",
        "mh_dec_mcts_full",
    ) and hasattr(planner, "_hierarchical_planner"):
        hp = planner._hierarchical_planner
        if hasattr(hp, "current_region_metadata"):
            region_metadata = hp.current_region_metadata
            selected_region_id = getattr(hp, "current_selected_region", None)
            region_scores = getattr(hp, "current_region_scores", None)

    return region_metadata, selected_region_id, region_scores

"""
Experiment Utilities for UAV Coverage Planning

Common utilities used across single-agent and multi-agent experiments:
- Agent initialization
- Metrics computation
- Logging helpers
- Visualization utilities
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from helper import compute_metrics, observed_m_ids, uav_position
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera


logger = logging.getLogger(__name__)


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


def compute_agent_metrics(
    agent_state: Dict,
    map_obj,
    ground_truth_map,
    grid_info,
) -> Tuple[float, float, float]:
    """
    Compute metrics for a single agent.

    Args:
        agent_state: Agent state dictionary
        map_obj: Map/environment object
        ground_truth_map: Ground truth for comparison
        grid_info: Grid information

    Returns:
        Tuple of (entropy, mse, coverage)
    """
    belief_map = agent_state["belief_map"]
    observed_ids = agent_state["observed_ids"]

    # Compute metrics
    entropy, mse, coverage = compute_metrics(
        map_obj, belief_map, observed_ids, grid_info, ground_truth_map
    )

    return entropy, mse, coverage


def update_agent_observation(
    agent_state: Dict,
    map_obj,
    conf_dict: Dict,
) -> None:
    """
    Update agent's belief map with new observation.

    Args:
        agent_state: Agent state dictionary
        map_obj: Map/environment object
        conf_dict: Configuration dictionary
    """
    camera = agent_state["camera"]
    uav_pos = agent_state["uav_pos"]
    occupancy_map = agent_state["occupancy_map"]

    # Get observation
    obsd_m_ids = observed_m_ids(camera, map_obj)
    agent_state["observed_ids"].update(obsd_m_ids)

    # Update belief
    new_belief = occupancy_map.updateBelief(
        uav_pos, camera, map_obj, obsd_m_ids, conf_dict
    )
    agent_state["belief_map"] = new_belief


def get_results_folder(
    base_folder: str,
    corr_type: str,
    action_strategy: str,
    e_margin,
    grf_r,
) -> str:
    """
    Generate results folder path.

    Args:
        base_folder: Base results directory
        corr_type: Correlation type
        action_strategy: Planning strategy
        e_margin: Error margin
        grf_r: GRF parameter

    Returns:
        Results folder path
    """
    e_str = f"e{e_margin}" if e_margin is not None else "eNone"
    return f"{base_folder}/{corr_type}_{action_strategy}_{e_str}_r{grf_r}"


def save_step_plot(
    agent_state: Dict,
    step: int,
    results_folder: str,
    iter_idx: int,
    ground_truth_map,
    map_obj,
) -> None:
    """
    Save visualization for a single step.

    Args:
        agent_state: Agent state dictionary
        step: Current step number
        results_folder: Results directory
        iter_idx: Iteration index
        ground_truth_map: Ground truth map
        map_obj: Map object
    """
    from viewer import plot_terrain_2d

    save_path = f"{results_folder}/{iter_idx}/steps/step_{step}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_terrain_2d(
        map_obj,
        agent_state["belief_map"],
        ground_truth_map,
        agent_state["uav_positions"],
        save_path,
    )


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


def compute_multi_agent_fused_metrics(
    agents: List[Dict],
    map_obj,
    ground_truth_map,
    grid_info,
) -> Tuple[float, float, float]:
    """
    Compute fused metrics across all agents.

    Fuses beliefs from all agents and computes global metrics.

    Args:
        agents: List of agent state dictionaries
        map_obj: Map/environment object
        ground_truth_map: Ground truth
        grid_info: Grid information

    Returns:
        Tuple of (fused_entropy, fused_mse, combined_coverage)
    """
    # Fuse beliefs (simple averaging for now)
    fused_belief = np.mean([agent["belief_map"] for agent in agents], axis=0)

    # Combine observed cells
    combined_observed = set()
    for agent in agents:
        combined_observed.update(agent["observed_ids"])

    # Compute fused metrics
    fused_entropy, fused_mse, combined_coverage = compute_metrics(
        map_obj, fused_belief, combined_observed, grid_info, ground_truth_map
    )

    return fused_entropy, fused_mse, combined_coverage


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
        "mh_dec_mcts",
        "hierarchical_dec_mcts",
        "mh_dec_mcts_both",
    ) and hasattr(planner, "_hierarchical_planner"):
        hp = planner._hierarchical_planner
        if hasattr(hp, "current_region_metadata"):
            region_metadata = hp.current_region_metadata
            selected_region_id = getattr(hp, "current_selected_region", None)
            region_scores = getattr(hp, "current_region_scores", None)

    return region_metadata, selected_region_id, region_scores

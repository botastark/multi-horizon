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
    camera = Camera(
        grid_info,
        60,
        seed=seed + agent_id,
        camera_altitude=min_alt,
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


# =============================================================================
# Multi-Agent Experiment Loop Utilities
# =============================================================================


def process_agent_observations(
    agents: List[Dict],
    map_obj,
    conf_dict: Dict,
    coordinator=None,
) -> Dict:
    """
    Phase 1: Process observations for all agents.

    Each agent:
    1. Gets observation from environment
    2. Updates local OG belief
    3. Runs local LBP propagation

    Args:
        agents: List of agent state dictionaries
        map_obj: Map/environment object
        conf_dict: Configuration dictionary
        coordinator: Optional coordinator

    Returns:
        Dict mapping agent_id to observation data
    """
    agent_observations = {}

    for agent in agents:
        agent_id = agent["agent_id"]
        camera = agent["camera"]
        occupancy_map = agent["occupancy_map"]
        uav_pos = agent["uav_pos"]
        belief_map = agent["belief_map"]

        # Process coordination messages
        if coordinator is not None:
            coordinator.process_messages(agent_id)

        # Get observations with sensor model
        sigmas = None
        if conf_dict is not None:
            s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
            sigmas = [s0, s1]

        fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

        # Update local belief with OG (Bayesian update)
        occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)

        # Run local LBP propagation (decentralized per agent)
        occupancy_map.propagate_messages(fp_vertices_ij, submap, max_iterations=1)

        # Update agent's belief map
        belief_map[:, :, 1] = occupancy_map.get_belief().copy()
        belief_map[:, :, 0] = 1 - belief_map[:, :, 1]
        agent["belief_map"] = belief_map

        # Store observation info
        agent_observations[agent_id] = {
            "fp_ij": fp_vertices_ij,
            "submap": submap,
            "sigmas": sigmas,
            "camera": camera,
            "uav_pos": uav_pos,
        }

    return agent_observations


def perform_belief_fusion(
    agents: List[Dict],
    coordinator,
    agent_observations: Dict,
    grid_info,
    step: int = 0,
) -> None:
    """
    Phase 2: Perform synchronous belief fusion across agents.

    Args:
        agents: List of agent state dictionaries
        coordinator: Coordinator object
        agent_observations: Observation data from Phase 1
        grid_info: Grid information
        step: Current step number
    """
    if coordinator.lbp_fusion is not None:
        # Synchronous news belief fusion
        coordinator.update_all_news(agent_observations)
        coordinator.fuse_all_news()

        # Log fusion stats periodically
        if step == 0 or step % 20 == 0:
            fusion_stats = coordinator.get_statistics().get("lbp_fusion", {})
            print(
                f"[Step {step}] Belief Fusion: news_fusions={fusion_stats.get('news_fusions', 0)}, "
                f"mode={fusion_stats.get('news_mode', 'N/A')}"
            )

        # Feed fused beliefs back to agents
        for agent in agents:
            agent_id = agent["agent_id"]
            fused_belief = coordinator.get_agent_belief(agent_id)
            if fused_belief is not None:
                agent["belief_map"][:, :, 1] = fused_belief
                agent["belief_map"][:, :, 0] = 1 - fused_belief
    else:
        # Fallback: simple belief sharing (weighted averaging)
        for agent in agents:
            agent_id = agent["agent_id"]
            camera = agent["camera"]
            uav_pos = agent["uav_pos"]
            belief_map = agent["belief_map"]

            if coordinator.should_coordinate(agent_id):
                observed_mask = np.zeros(grid_info.shape, dtype=bool)
                [[imin, imax], [jmin, jmax]] = camera.get_range(
                    position=uav_pos.position,
                    altitude=uav_pos.altitude,
                    index_form=True,
                )
                observed_mask[imin:imax, jmin:jmax] = True
                coordinator.share_belief(agent_id, belief_map, observed_mask)


def select_agent_actions(
    agents: List[Dict],
    ground_truth_map,
    grid_info,
    coordinator,
    config: Dict,
    step: int = 0,
) -> None:
    """
    Phase 3: Compute metrics, select actions for all agents.

    Args:
        agents: List of agent state dictionaries
        ground_truth_map: Ground truth map
        grid_info: Grid information
        coordinator: Coordinator object
        config: Configuration dictionary
        step: Current step number
    """
    for agent in agents:
        agent_id = agent["agent_id"]
        camera = agent["camera"]
        planner = agent["planner"]
        uav_pos = agent["uav_pos"]
        belief_map = agent["belief_map"]

        # Compute metrics
        agent["observed_ids"].update(observed_m_ids(camera, uav_pos))
        entropy_val, mse_val, coverage_val = compute_metrics(
            ground_truth_map, belief_map, agent["observed_ids"], grid_info
        )
        agent["entropy"].append(entropy_val)
        agent["mse"].append(mse_val)
        agent["coverage"].append(coverage_val)
        agent["height"].append(uav_pos.altitude)

        # Select action
        next_action, info_gain_action = planner.select_action(
            belief_map, agent["uav_positions"]
        )
        agent["info_gain_action"] = info_gain_action

        # Apply collision avoidance if enabled
        if coordinator and coordinator.collision_distance > 0:
            next_action = _apply_collision_avoidance(
                agent, next_action, info_gain_action, coordinator, config, camera
            )

        # Log selected action
        print(
            f"[Agent {agent_id}] Step {step}: action={next_action} | "
            f"pos=({uav_pos.position[0]:.1f}, {uav_pos.position[1]:.1f})"
        )

        # Store action for later position update
        agent["_next_action"] = next_action


def _apply_collision_avoidance(
    agent: Dict,
    next_action,
    info_gain_action: Dict,
    coordinator,
    config: Dict,
    camera,
) -> Any:
    """
    Apply collision avoidance penalty to action selection.

    Args:
        agent: Agent state dictionary
        next_action: Initially selected action
        info_gain_action: Action scores
        coordinator: Coordinator object
        config: Configuration dictionary
        camera: Camera object

    Returns:
        Adjusted action with collision avoidance
    """
    agent_id = agent["agent_id"]
    best_action = next_action
    best_score = float("-inf")

    # Get max IG for normalization
    valid_scores = [s for s in info_gain_action.values() if isinstance(s, (int, float))]
    max_ig = max(max(valid_scores), 1.0) if valid_scores else 1.0

    for action, ig_score in info_gain_action.items():
        if not isinstance(ig_score, (int, float)):
            continue
        proposed_state = camera.x_future(action)
        if proposed_state is None:
            continue
        proposed_pos = uav_position(proposed_state)
        proposed_row, proposed_col = camera.convert_xy_ij(
            proposed_pos.position[0],
            proposed_pos.position[1],
            camera.grid.center,
        )

        # Get collision penalty (0-1 range)
        penalty = coordinator.get_collision_penalty(
            agent_id, (proposed_row, proposed_col)
        )

        # Adjusted score with collision penalty
        collision_weight = config.get("agents", {}).get("collision_penalty_weight", 1.0)
        adjusted_score = ig_score - penalty * max_ig * collision_weight

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_action = action

    return best_action


def update_agent_positions(
    agents: List[Dict],
    coordinator=None,
) -> None:
    """
    Update all agent positions based on selected actions.

    Args:
        agents: List of agent state dictionaries
        coordinator: Optional coordinator object
    """
    for agent in agents:
        agent_id = agent["agent_id"]
        camera = agent["camera"]
        next_action = agent.get("_next_action")

        if next_action is None:
            continue

        # Update UAV position
        uav_pos = uav_position(camera.x_future(next_action))
        agent["actions"].append(next_action)
        agent["uav_positions"].append(uav_pos)
        agent["uav_pos"] = uav_pos
        camera.set_altitude(uav_pos.altitude)
        camera.set_position(uav_pos.position)

        # Update coordinator
        if coordinator:
            current_row, current_col = camera.convert_xy_ij(
                uav_pos.position[0], uav_pos.position[1], camera.grid.center
            )
            coverage_val = agent["coverage"][-1] if agent["coverage"] else 0.0
            coordinator.update_agent_state(
                agent_id=agent_id,
                position=(current_row, current_col),
                altitude=uav_pos.altitude,
                coverage=coverage_val,
            )

        # Clean up temporary action
        agent.pop("_next_action", None)


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

    if action_strategy in ("mh_dec_mcts", "hierarchical_dec_mcts") and hasattr(
        planner, "_hierarchical_planner"
    ):
        hp = planner._hierarchical_planner
        if hasattr(hp, "current_region_metadata"):
            region_metadata = hp.current_region_metadata
            selected_region_id = getattr(hp, "current_selected_region", None)
            region_scores = getattr(hp, "current_region_scores", None)

    return region_metadata, selected_region_id, region_scores

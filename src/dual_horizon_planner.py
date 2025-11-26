# dual_horizon_planner.py
"""
Dual-Horizon MCTS Planner for UAV Coverage Planning

This module implements a dual-horizon planning approach that combines:
- Short-horizon: Information gain (IG) greedy planning for local exploitation
- Long-horizon: Coverage optimization to avoid leaving isolated uncovered patches

The key insight is that pure IG-greedy planning can leave fragmented uncovered regions
that are expensive to revisit later. By considering both short and long horizons,
we achieve better overall coverage quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
import copy
import logging

from helper import uav_position, H, expected_posterior
from mcts import MCTSPlanner, MCTSNode, copy_state

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants for fragmentation analysis and blend weight computation
# -----------------------------------------------------------------------------

# Fragmentation score normalization: patches above this count have maximum penalty
FRAGMENTATION_PATCH_THRESHOLD = 10

# Blend weight adjustment factors for dual-horizon planning
# These control how much each factor affects the short/long horizon balance
COVERAGE_ADJUSTMENT_FACTOR = 0.3    # How much low coverage boosts long-horizon
UNCERTAINTY_ADJUSTMENT_FACTOR = 0.2  # How much high uncertainty boosts short-horizon
FRAGMENTATION_ADJUSTMENT_FACTOR = 0.3  # How much fragmentation boosts long-horizon


# -----------------------------------------------------------------------------
# Coverage Analysis Functions
# -----------------------------------------------------------------------------

def analyze_coverage_fragmentation(covered_mask: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the coverage map to detect isolated uncovered regions.
    
    Uses connected components analysis to identify separate uncovered patches
    and calculate metrics about coverage fragmentation.
    
    Args:
        covered_mask: Boolean 2D array where True indicates covered cells
        
    Returns:
        Dict with fragmentation metrics:
            - num_patches: Number of separate uncovered regions
            - patch_sizes: List of sizes for each patch
            - patch_centroids: List of (row, col) centroids for each patch
            - total_uncovered: Total number of uncovered cells
            - fragmentation_score: Normalized fragmentation metric (0-1)
    """
    if covered_mask is None:
        return {
            'num_patches': 0,
            'patch_sizes': [],
            'patch_centroids': [],
            'total_uncovered': 0,
            'fragmentation_score': 0.0
        }
    
    # Invert mask to get uncovered areas
    uncovered_mask = ~covered_mask
    
    # Label connected components in uncovered regions
    # Using 8-connectivity for diagonal neighbors
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_array, num_patches = ndimage.label(uncovered_mask, structure=structure)
    
    if num_patches == 0:
        return {
            'num_patches': 0,
            'patch_sizes': [],
            'patch_centroids': [],
            'total_uncovered': 0,
            'fragmentation_score': 0.0
        }
    
    # Calculate size and centroid of each patch
    patch_sizes = []
    patch_centroids = []
    
    for label_id in range(1, num_patches + 1):
        patch_mask = (labeled_array == label_id)
        size = np.sum(patch_mask)
        patch_sizes.append(int(size))
        
        # Calculate centroid
        rows, cols = np.where(patch_mask)
        centroid = (float(np.mean(rows)), float(np.mean(cols)))
        patch_centroids.append(centroid)
    
    total_uncovered = np.sum(uncovered_mask)
    total_cells = covered_mask.size
    
    # Fragmentation score: higher when there are many small patches
    # Normalized by field size. Uses FRAGMENTATION_PATCH_THRESHOLD as the
    # normalization factor - patches above this count contribute maximum penalty.
    if total_uncovered > 0 and num_patches > 0:
        avg_patch_size = total_uncovered / num_patches
        # Penalize small patches more heavily
        size_variance = np.var(patch_sizes) if len(patch_sizes) > 1 else 0
        # Score increases with more patches and smaller average size
        fragmentation_score = min(
            1.0, 
            (num_patches / FRAGMENTATION_PATCH_THRESHOLD) * (1.0 - avg_patch_size / total_cells)
        )
    else:
        fragmentation_score = 0.0
    
    return {
        'num_patches': num_patches,
        'patch_sizes': patch_sizes,
        'patch_centroids': patch_centroids,
        'total_uncovered': int(total_uncovered),
        'fragmentation_score': fragmentation_score
    }


def compute_revisit_cost(
    uav_pos: 'uav_position',
    uncovered_patches: Dict[str, Any],
    uav_speed: float = 1.0,
    grid_length: float = 1.0
) -> float:
    """
    Calculate the cost of returning to cover isolated patches later.
    
    For each uncovered patch, estimate travel distance from current position.
    Weight by patch size (small isolated patches are more expensive per cell).
    
    Args:
        uav_pos: Current UAV position with .position attribute
        uncovered_patches: Output from analyze_coverage_fragmentation()
        uav_speed: UAV movement speed in grid units per time step
        grid_length: Length of each grid cell in real units
        
    Returns:
        Total expected revisit cost (higher = more expensive to cover later)
    """
    if uncovered_patches['num_patches'] == 0:
        return 0.0
    
    total_cost = 0.0
    current_pos = np.array(uav_pos.position)
    
    for i, (centroid, size) in enumerate(zip(
        uncovered_patches['patch_centroids'],
        uncovered_patches['patch_sizes']
    )):
        # Convert centroid (row, col) to position coordinates (x, y)
        # Grid convention: centroid[0] = row index, centroid[1] = col index
        # Position convention: position[0] = x (corresponds to col), position[1] = y (corresponds to row)
        patch_pos = np.array([centroid[1] * grid_length, centroid[0] * grid_length])
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(current_pos - patch_pos)
        
        # Travel time to reach patch
        travel_time = distance / max(uav_speed, 1e-6)
        
        # Small isolated patches are more expensive per cell
        # because you still need to travel there for minimal gain
        size_penalty = 1.0 / max(np.sqrt(size), 1.0)
        
        # Cost = travel_time * size_penalty
        patch_cost = travel_time * size_penalty
        total_cost += patch_cost
    
    return total_cost


def partition_field(
    belief_map: np.ndarray,
    tile_size: Tuple[int, int] = (10, 10)
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Divide the field into virtual sub-areas (tiles) for long-horizon planning.
    
    Args:
        belief_map: 3D array of shape (H, W, 2) with belief probabilities
        tile_size: (height, width) of each tile in grid cells
        
    Returns:
        Tuple of:
            - tile_map: 2D array where each cell contains its tile ID
            - tile_metadata: Dict mapping tile_id to metadata:
                - 'bounds': ((row_min, row_max), (col_min, col_max))
                - 'cells': List of (row, col) tuples
                - 'entropy': Total entropy in tile
                - 'center': (row, col) center of tile
    """
    H_dim, W_dim = belief_map.shape[:2]
    tile_h, tile_w = tile_size
    
    tile_map = np.zeros((H_dim, W_dim), dtype=np.int32)
    tile_metadata = {}
    
    tile_id = 0
    for row_start in range(0, H_dim, tile_h):
        for col_start in range(0, W_dim, tile_w):
            row_end = min(row_start + tile_h, H_dim)
            col_end = min(col_start + tile_w, W_dim)
            
            # Mark cells with tile ID
            tile_map[row_start:row_end, col_start:col_end] = tile_id
            
            # Gather cells
            cells = [
                (r, c)
                for r in range(row_start, row_end)
                for c in range(col_start, col_end)
            ]
            
            # Calculate entropy for this tile
            tile_belief = belief_map[row_start:row_end, col_start:col_end, 1]
            tile_entropy = float(np.sum(H(tile_belief)))
            
            # Center of tile
            center = (
                (row_start + row_end - 1) / 2.0,
                (col_start + col_end - 1) / 2.0
            )
            
            tile_metadata[tile_id] = {
                'bounds': ((row_start, row_end), (col_start, col_end)),
                'cells': cells,
                'entropy': tile_entropy,
                'center': center
            }
            
            tile_id += 1
    
    return tile_map, tile_metadata


# -----------------------------------------------------------------------------
# Long-Horizon Reward Function
# -----------------------------------------------------------------------------

def compute_long_horizon_reward(
    state: Dict[str, Any],
    action_sequence: List[str],
    camera: Any,
    weights: Dict[str, float],
    conf_dict: Optional[Dict] = None
) -> float:
    """
    Evaluate action sequences based on coverage quality for long-horizon planning.
    
    Simulates the action sequence and computes reward based on:
    - Coverage area gained (positive)
    - Number of fragmented uncovered patches created (penalty)
    - Total revisit cost for isolated patches (penalty)
    - Leaving small gaps between covered areas (penalty)
    
    Args:
        state: Dict with 'uav_pos', 'belief', 'covered_mask'
        action_sequence: List of action strings to evaluate
        camera: UAV camera object for position updates and footprint
        weights: Dict with reward weights:
            - 'w_coverage': Weight for coverage reward
            - 'w_fragmentation': Penalty weight for fragmentation
            - 'w_revisit_cost': Penalty weight for revisit costs
        conf_dict: Optional sensor confusion matrix dict
        
    Returns:
        Weighted reward score (higher is better)
    """
    # Make copies to avoid modifying original state
    sim_state = copy_state(state)
    
    # Initialize covered_mask if not present
    H_dim, W_dim = sim_state['belief'].shape[:2]
    if 'covered_mask' not in sim_state or sim_state['covered_mask'] is None:
        sim_state['covered_mask'] = np.zeros((H_dim, W_dim), dtype=bool)
    
    initial_coverage = np.sum(sim_state['covered_mask'])
    
    # Simulate action sequence
    current_pos = sim_state['uav_pos']
    
    for action in action_sequence:
        # Get next position
        next_state = camera.x_future(action, x=current_pos)
        if next_state is None:
            continue
        
        current_pos = uav_position(next_state)
        
        # Get footprint range
        try:
            [[imin, imax], [jmin, jmax]] = camera.get_range(
                position=current_pos.position,
                altitude=current_pos.altitude,
                index_form=True
            )
        except (ValueError, IndexError):
            continue
        
        # Mark footprint as covered
        sim_state['covered_mask'][imin:imax, jmin:jmax] = True
    
    # Calculate final metrics
    final_coverage = np.sum(sim_state['covered_mask'])
    coverage_gained = (final_coverage - initial_coverage) / max(H_dim * W_dim, 1)
    
    # Analyze fragmentation of remaining uncovered areas
    fragmentation_info = analyze_coverage_fragmentation(sim_state['covered_mask'])
    
    # Calculate revisit cost
    revisit_cost = compute_revisit_cost(
        current_pos,
        fragmentation_info,
        uav_speed=1.0
    )
    
    # Normalize revisit cost
    max_distance = np.sqrt(H_dim**2 + W_dim**2)
    revisit_cost_normalized = revisit_cost / max(max_distance * 10, 1.0)
    
    # Compute weighted reward
    w_coverage = weights.get('w_coverage', 1.0)
    w_fragmentation = weights.get('w_fragmentation', 0.5)
    w_revisit = weights.get('w_revisit_cost', 0.3)
    
    reward = (
        w_coverage * coverage_gained
        - w_fragmentation * fragmentation_info['fragmentation_score']
        - w_revisit * revisit_cost_normalized
    )
    
    return reward


# -----------------------------------------------------------------------------
# DualHorizonPlanner Class
# -----------------------------------------------------------------------------

class DualHorizonPlanner:
    """
    Dual-horizon MCTS planner combining short and long horizon planning.
    
    The planner balances:
    - Short-horizon: Information gain exploitation (current MCTS approach)
    - Long-horizon: Coverage optimization to avoid fragmentation
    
    The blend between horizons adapts based on mission progress:
    - Early mission: favor long-horizon (global coverage strategy)
    - High uncertainty areas: favor short-horizon (IG exploitation)
    - Late mission: balance to avoid fragmentation
    """
    
    def __init__(
        self,
        uav_camera: Any,
        conf_dict: Optional[Dict] = None,
        mcts_params: Optional[Dict] = None,
        horizon_weights: Optional[Dict] = None
    ):
        """
        Initialize dual-horizon planner.
        
        Args:
            uav_camera: UAV camera object for action space and observations
            conf_dict: Sensor confusion matrix dictionary
            mcts_params: MCTS configuration (depth, iterations, etc.)
            horizon_weights: Dict with keys:
                - 'w_coverage': Weight for coverage reward
                - 'w_fragmentation': Penalty weight for fragmentation
                - 'w_revisit_cost': Penalty weight for revisit costs
                - 'w_ig': Weight for information gain (short horizon)
                - 'short_horizon_depth': Planning depth for short horizon (e.g., 5)
                - 'long_horizon_depth': Planning depth for long horizon (e.g., 15)
                - 'tile_size': Tile size for partitioning [height, width]
        """
        self.camera = uav_camera
        self.conf_dict = conf_dict
        
        # MCTS parameters with defaults
        self.mcts_params = mcts_params or {}
        self.num_iterations = self.mcts_params.get('num_iterations', 100)
        self.timeout = self.mcts_params.get('timeout', 2000)
        self.ucb1_c = self.mcts_params.get('ucb1_c', 0.95)
        self.discount_factor = self.mcts_params.get('discount_factor', 0.99)
        self.parallel = self.mcts_params.get('parallel', 1)
        
        # Horizon-specific parameters
        horizon_weights = horizon_weights or {}
        self.w_coverage = horizon_weights.get('w_coverage', 1.0)
        self.w_fragmentation = horizon_weights.get('w_fragmentation', 0.5)
        self.w_revisit_cost = horizon_weights.get('w_revisit_cost', 0.3)
        self.w_ig = horizon_weights.get('w_ig', 0.8)
        self.short_horizon_depth = horizon_weights.get('short_horizon_depth', 5)
        self.long_horizon_depth = horizon_weights.get('long_horizon_depth', 15)
        self.tile_size = tuple(horizon_weights.get('tile_size', [10, 10]))
        
        # Weights dict for reward computation
        self.weights = {
            'w_coverage': self.w_coverage,
            'w_fragmentation': self.w_fragmentation,
            'w_revisit_cost': self.w_revisit_cost
        }
        
        logger.info(
            f"DualHorizonPlanner initialized: short_depth={self.short_horizon_depth}, "
            f"long_depth={self.long_horizon_depth}, weights={self.weights}"
        )
    
    def select_action(
        self,
        state: Dict[str, Any],
        strategy: str = 'dual'
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select next action using specified strategy.
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            strategy: 'short', 'long', or 'dual' (combined)
        
        Returns:
            Tuple of (action, metrics_dict) where metrics_dict contains
            planning diagnostics and scores
        """
        # Ensure state has covered_mask
        if 'covered_mask' not in state or state['covered_mask'] is None:
            H_dim, W_dim = state['belief'].shape[:2]
            state['covered_mask'] = np.zeros((H_dim, W_dim), dtype=bool)
        
        if strategy == 'short':
            return self.short_horizon_plan(state)
        elif strategy == 'long':
            return self.long_horizon_plan(state)
        elif strategy == 'dual':
            return self.dual_horizon_plan(state)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def short_horizon_plan(
        self,
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run short-horizon IG-greedy MCTS (current approach).
        
        This uses the standard MCTS with information gain reward,
        focusing on immediate exploitation of high-uncertainty areas.
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            
        Returns:
            Tuple of (action, metrics_dict)
        """
        planner = MCTSPlanner(
            initial_state=state,
            uav_camera=self.camera,
            conf_dict=self.conf_dict,
            discount_factor=self.discount_factor,
            max_depth=self.short_horizon_depth,
            parallel=self.parallel,
            ucb1_c=self.ucb1_c,
            plan_cfg={'use_coverage_reward': False}  # Use IG reward
        )
        
        action, action_scores = planner.search(
            num_iterations=self.num_iterations,
            timeout=self.timeout,
            return_action_scores=True
        )
        
        metrics = {
            'strategy': 'short',
            'action_scores': action_scores,
            'planning_depth': self.short_horizon_depth
        }
        
        return action, metrics
    
    def long_horizon_plan(
        self,
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run long-horizon coverage-optimizing MCTS with fragmentation penalties.
        
        This uses MCTS with coverage reward and additional penalties
        for creating fragmented uncovered regions.
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            
        Returns:
            Tuple of (action, metrics_dict)
        """
        # Use coverage reward with fragmentation awareness
        plan_cfg = {
            'use_coverage_reward': True,
            'horizon_weights': self.weights
        }
        
        planner = MCTSPlanner(
            initial_state=state,
            uav_camera=self.camera,
            conf_dict=self.conf_dict,
            discount_factor=self.discount_factor,
            max_depth=self.long_horizon_depth,
            parallel=self.parallel,
            ucb1_c=self.ucb1_c,
            plan_cfg=plan_cfg
        )
        
        action, action_scores = planner.search(
            num_iterations=self.num_iterations,
            timeout=self.timeout,
            return_action_scores=True
        )
        
        # Analyze current coverage state
        fragmentation_info = analyze_coverage_fragmentation(state['covered_mask'])
        
        metrics = {
            'strategy': 'long',
            'action_scores': action_scores,
            'planning_depth': self.long_horizon_depth,
            'fragmentation_info': fragmentation_info
        }
        
        return action, metrics
    
    def dual_horizon_plan(
        self,
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Combine short and long horizon planning.
        
        Approach:
        1. Run short-horizon MCTS to get exploitation plan
        2. Run long-horizon MCTS to get coverage strategy
        3. Blend decisions based on current coverage state:
           - Early mission: favor long-horizon (global coverage)
           - High uncertainty areas: favor short-horizon (IG exploitation)
           - Late mission: balance to avoid fragmentation
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            
        Returns:
            Tuple of (action, metrics_dict)
        """
        # Run both planners
        short_action, short_metrics = self.short_horizon_plan(state)
        long_action, long_metrics = self.long_horizon_plan(state)
        
        # Calculate blending weights based on current state
        blend_weights = self._compute_blend_weights(state)
        
        # Get action scores from both
        short_scores = short_metrics.get('action_scores', {})
        long_scores = long_metrics.get('action_scores', {})
        
        # Combine scores with blending weights
        combined_scores = {}
        all_actions = set(short_scores.keys()) | set(long_scores.keys())
        
        for action in all_actions:
            short_val = short_scores.get(action, 0.0)
            long_val = long_scores.get(action, 0.0)
            
            # Normalize scores to [0, 1] range for fair blending
            combined_scores[action] = (
                blend_weights['w_short'] * short_val +
                blend_weights['w_long'] * long_val
            )
        
        # Select action with highest combined score
        if combined_scores:
            selected_action = max(combined_scores.keys(), key=lambda a: combined_scores[a])
        else:
            # Fallback to short horizon action
            selected_action = short_action
        
        metrics = {
            'strategy': 'dual',
            'selected_action': selected_action,
            'short_action': short_action,
            'long_action': long_action,
            'short_scores': short_scores,
            'long_scores': long_scores,
            'combined_scores': combined_scores,
            'blend_weights': blend_weights,
            'fragmentation_info': long_metrics.get('fragmentation_info', {})
        }
        
        logger.debug(
            f"Dual-horizon: short={short_action}, long={long_action}, "
            f"selected={selected_action}, blend={blend_weights}"
        )
        
        return selected_action, metrics
    
    def _compute_blend_weights(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute blending weights for combining short and long horizon decisions.
        
        The weights adapt based on:
        - Coverage progress (early vs late mission)
        - Current uncertainty level
        - Fragmentation state
        
        Args:
            state: Current planning state
            
        Returns:
            Dict with 'w_short' and 'w_long' weights (sum to 1.0)
        """
        belief = state['belief']
        covered_mask = state.get('covered_mask')
        
        # Calculate coverage progress (0 = no coverage, 1 = full coverage)
        if covered_mask is not None:
            coverage_progress = np.mean(covered_mask)
        else:
            coverage_progress = 0.0
        
        # Calculate average uncertainty (entropy)
        avg_entropy = np.mean(H(belief[:, :, 1]))
        max_entropy = 1.0  # Maximum binary entropy
        uncertainty_ratio = avg_entropy / max_entropy
        
        # Analyze fragmentation
        fragmentation_info = analyze_coverage_fragmentation(covered_mask)
        fragmentation_score = fragmentation_info['fragmentation_score']
        
        # Adaptive weighting logic:
        # Early mission (low coverage): favor long-horizon for global strategy
        # High uncertainty: favor short-horizon for IG exploitation
        # High fragmentation: favor long-horizon to consolidate
        
        # Base weights
        base_short = self.w_ig
        base_long = self.w_coverage
        
        # Adjust based on coverage progress
        # Early mission: boost long-horizon (uses COVERAGE_ADJUSTMENT_FACTOR)
        coverage_adjustment = COVERAGE_ADJUSTMENT_FACTOR * (1.0 - coverage_progress)
        
        # High uncertainty: boost short-horizon (uses UNCERTAINTY_ADJUSTMENT_FACTOR)
        uncertainty_adjustment = UNCERTAINTY_ADJUSTMENT_FACTOR * uncertainty_ratio
        
        # High fragmentation: boost long-horizon (uses FRAGMENTATION_ADJUSTMENT_FACTOR)
        fragmentation_adjustment = FRAGMENTATION_ADJUSTMENT_FACTOR * fragmentation_score
        
        # Compute final weights
        w_short = base_short + uncertainty_adjustment - coverage_adjustment - fragmentation_adjustment
        w_long = base_long - uncertainty_adjustment + coverage_adjustment + fragmentation_adjustment
        
        # Normalize to sum to 1
        total = w_short + w_long
        if total > 0:
            w_short /= total
            w_long /= total
        else:
            w_short = 0.5
            w_long = 0.5
        
        return {
            'w_short': float(w_short),
            'w_long': float(w_long),
            'coverage_progress': float(coverage_progress),
            'uncertainty_ratio': float(uncertainty_ratio),
            'fragmentation_score': float(fragmentation_score)
        }


# -----------------------------------------------------------------------------
# Utility function for creating planner from config
# -----------------------------------------------------------------------------

def create_dual_horizon_planner(
    uav_camera: Any,
    conf_dict: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> DualHorizonPlanner:
    """
    Factory function to create a DualHorizonPlanner from configuration.
    
    Args:
        uav_camera: UAV camera object
        conf_dict: Sensor confusion matrix
        config: Full configuration dict (typically from config.json)
        
    Returns:
        Configured DualHorizonPlanner instance
    """
    config = config or {}
    mcts_params = config.get('mcts_params', {})
    horizon_weights = mcts_params.get('horizon_weights', {})
    
    return DualHorizonPlanner(
        uav_camera=uav_camera,
        conf_dict=conf_dict,
        mcts_params=mcts_params,
        horizon_weights=horizon_weights
    )

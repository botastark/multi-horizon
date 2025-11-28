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
import json
import os
from datetime import datetime

from helper import uav_position, H, expected_posterior
from mcts import MCTSPlanner, MCTSNode, copy_state

# Configure logging
logger = logging.getLogger(__name__)

# Dual-horizon specific logger for detailed analysis
dual_horizon_logger = None
dual_horizon_log_file = None


def setup_dual_horizon_logger(log_dir: str = "logs", experiment_name: str = None) -> str:
    """
    Set up a dedicated logger for dual-horizon planning analysis.
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Optional experiment name for the log file
        
    Returns:
        Path to the log file
    """
    global dual_horizon_logger, dual_horizon_log_file
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_suffix = f"_{experiment_name}" if experiment_name else ""
    log_filename = f"dual_horizon{exp_suffix}_{timestamp}.log"
    dual_horizon_log_file = os.path.join(log_dir, log_filename)
    
    # Create dedicated logger
    dual_horizon_logger = logging.getLogger('dual_horizon_detailed')
    dual_horizon_logger.setLevel(logging.DEBUG)
    dual_horizon_logger.handlers.clear()  # Remove existing handlers
    
    # File handler with detailed format
    file_handler = logging.FileHandler(dual_horizon_log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed formatter (no timestamps)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    dual_horizon_logger.addHandler(file_handler)
    
    # Log initialization
    dual_horizon_logger.info("="*80)
    dual_horizon_logger.info("DUAL-HORIZON PLANNER LOG")
    dual_horizon_logger.info(f"Experiment: {experiment_name if experiment_name else 'default'}")
    dual_horizon_logger.info(f"Timestamp: {timestamp}")
    dual_horizon_logger.info("="*80)
    
    return dual_horizon_log_file


def log_planning_step(
    step_num: int,
    state: Dict[str, Any],
    short_action: str,
    short_scores: Dict[str, float],
    long_action: str,
    long_scores: Dict[str, float],
    blend_weights: Dict[str, float],
    combined_scores: Dict[str, float],
    selected_action: str,
    fragmentation_info: Dict[str, Any],
    additional_info: Optional[Dict] = None
):
    """
    Log detailed information about a planning step.
    
    Args:
        step_num: Current step number
        state: Current planning state
        short_action: Action selected by short-horizon planner
        short_scores: Action scores from short-horizon planner
        long_action: Action selected by long-horizon planner
        long_scores: Action scores from long-horizon planner
        blend_weights: Computed blend weights
        combined_scores: Combined action scores
        selected_action: Final selected action
        fragmentation_info: Fragmentation analysis results
        additional_info: Optional additional metrics
    """
    if dual_horizon_logger is None:
        return
    
    dual_horizon_logger.info("")
    dual_horizon_logger.info(f"{'='*80}")
    dual_horizon_logger.info(f"STEP {step_num}")
    dual_horizon_logger.info(f"{'='*80}")
    
    # State information
    uav_pos = state.get('uav_pos')
    belief = state.get('belief')
    covered_mask = state.get('covered_mask')
    
    if uav_pos:
        dual_horizon_logger.info(f"UAV Position: {uav_pos.position}, Altitude: {uav_pos.altitude:.2f}")
    
    if belief is not None:
        avg_entropy = np.mean(H(belief[:, :, 1]))
        dual_horizon_logger.info(f"Average Entropy: {avg_entropy:.4f}")
    
    if covered_mask is not None:
        coverage = np.mean(covered_mask)
        dual_horizon_logger.info(f"Coverage Progress: {coverage:.4f} ({coverage*100:.2f}%)")
    
    # Blend weights and state metrics
    dual_horizon_logger.info("")
    dual_horizon_logger.info("BLEND WEIGHTS COMPUTATION:")
    dual_horizon_logger.info(f"  Coverage Progress: {blend_weights.get('coverage_progress', 0):.4f}")
    dual_horizon_logger.info(f"  Uncertainty Ratio: {blend_weights.get('uncertainty_ratio', 0):.4f}")
    dual_horizon_logger.info(f"  Fragmentation Score: {blend_weights.get('fragmentation_score', 0):.4f}")
    dual_horizon_logger.info(f"  → w_short: {blend_weights.get('w_short', 0):.4f}")
    dual_horizon_logger.info(f"  → w_long:  {blend_weights.get('w_long', 0):.4f}")
    
    # Fragmentation details
    dual_horizon_logger.info("")
    dual_horizon_logger.info("FRAGMENTATION ANALYSIS:")
    dual_horizon_logger.info(f"  Num Patches: {fragmentation_info.get('num_patches', 0)}")
    dual_horizon_logger.info(f"  Total Uncovered: {fragmentation_info.get('total_uncovered', 0)}")
    dual_horizon_logger.info(f"  Patch Sizes: {fragmentation_info.get('patch_sizes', [])[:10]}")  # Show first 10
    dual_horizon_logger.info(f"  Fragmentation Score: {fragmentation_info.get('fragmentation_score', 0):.4f}")
    
    # Planner decisions
    dual_horizon_logger.info("")
    dual_horizon_logger.info("LLP (Short-Horizon) Decision:")
    dual_horizon_logger.info(f"  Selected Action: {short_action}")
    dual_horizon_logger.info(f"  Action Scores: {json.dumps({k: f'{v:.4f}' for k, v in short_scores.items()}, indent=4)}")
    
    dual_horizon_logger.info("")
    dual_horizon_logger.info("HLP (Long-Horizon) Decision:")
    dual_horizon_logger.info(f"  Selected Action: {long_action}")
    dual_horizon_logger.info(f"  Action Scores: {json.dumps({k: f'{v:.4f}' for k, v in long_scores.items()}, indent=4)}")
    
    # Blending results
    dual_horizon_logger.info("")
    dual_horizon_logger.info("BLENDING COMPUTATION:")
    for action in sorted(combined_scores.keys()):
        short_val = short_scores.get(action, 0.0)
        long_val = long_scores.get(action, 0.0)
        combined_val = combined_scores[action]
        dual_horizon_logger.info(
            f"  {action}: "
            f"LLP={short_val:.4f} * {blend_weights['w_short']:.3f} + "
            f"HLP={long_val:.4f} * {blend_weights['w_long']:.3f} = {combined_val:.4f}"
        )
    
    dual_horizon_logger.info("")
    dual_horizon_logger.info(f"FINAL SELECTED ACTION: {selected_action}")
    dual_horizon_logger.info(f"Agreement: {'YES' if short_action == long_action == selected_action else 'PARTIAL' if selected_action in [short_action, long_action] else 'DIFFERENT'}")
    
    # Additional info
    if additional_info:
        dual_horizon_logger.info("")
        dual_horizon_logger.info("ADDITIONAL METRICS:")
        for key, value in additional_info.items():
            dual_horizon_logger.info(f"  {key}: {value}")
    
    dual_horizon_logger.info("")


def log_summary_statistics(total_steps: int, final_coverage: float, additional_stats: Optional[Dict] = None):
    """
    Log summary statistics at the end of an episode/experiment.
    
    Args:
        total_steps: Total number of steps taken
        final_coverage: Final coverage achieved
        additional_stats: Optional additional statistics
    """
    if dual_horizon_logger is None:
        return
    
    dual_horizon_logger.info("")
    dual_horizon_logger.info("="*80)
    dual_horizon_logger.info("EPISODE SUMMARY")
    dual_horizon_logger.info("="*80)
    dual_horizon_logger.info(f"Total Steps: {total_steps}")
    dual_horizon_logger.info(f"Final Coverage: {final_coverage:.4f} ({final_coverage*100:.2f}%)")
    
    if additional_stats:
        for key, value in additional_stats.items():
            dual_horizon_logger.info(f"{key}: {value}")
    
    dual_horizon_logger.info("="*80)
    dual_horizon_logger.info("")


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
    covered_mask: np.ndarray = None,
    tile_size: Tuple[int, int] = (20, 20)
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Divide the field into virtual sub-areas (regions) for long-horizon planning.
    
    Args:
        belief_map: 3D array of shape (H, W, 2) with belief probabilities
        covered_mask: Boolean array indicating covered areas
        tile_size: (height, width) of each region in grid cells
        
    Returns:
        Tuple of:
            - region_map: 2D array where each cell contains its region ID
            - region_metadata: Dict mapping region_id to metadata:
                - 'bounds': ((row_min, row_max), (col_min, col_max))
                - 'cells': List of (row, col) tuples
                - 'entropy': Total entropy in region
                - 'center': (row, col) center of region
                - 'coverage': Coverage ratio in region
                - 'uncovered_cells': Number of uncovered cells
                - 'value': Combined value score for prioritization
    """
    H_dim, W_dim = belief_map.shape[:2]
    tile_h, tile_w = tile_size
    
    region_map = np.zeros((H_dim, W_dim), dtype=np.int32)
    region_metadata = {}
    
    if covered_mask is None:
        covered_mask = np.zeros((H_dim, W_dim), dtype=bool)
    
    region_id = 0
    for row_start in range(0, H_dim, tile_h):
        for col_start in range(0, W_dim, tile_w):
            row_end = min(row_start + tile_h, H_dim)
            col_end = min(col_start + tile_w, W_dim)
            
            # Mark cells with region ID
            region_map[row_start:row_end, col_start:col_end] = region_id
            
            # Gather cells
            cells = [
                (r, c)
                for r in range(row_start, row_end)
                for c in range(col_start, col_end)
            ]
            
            # Calculate entropy for this region
            region_belief = belief_map[row_start:row_end, col_start:col_end, 1]
            region_entropy = float(np.sum(H(region_belief)))
            
            # Calculate coverage for this region
            region_coverage = covered_mask[row_start:row_end, col_start:col_end]
            coverage_ratio = float(np.mean(region_coverage))
            uncovered_cells = int(np.sum(~region_coverage))
            
            # Center of region (in grid coordinates)
            center_row = (row_start + row_end - 1) / 2.0
            center_col = (col_start + col_end - 1) / 2.0
            
            # Calculate region value (for HLP prioritization)
            # High value = high entropy + low coverage
            avg_entropy = region_entropy / len(cells) if cells else 0
            value = avg_entropy * (1.0 - coverage_ratio)
            
            region_metadata[region_id] = {
                'bounds': ((row_start, row_end), (col_start, col_end)),
                'cells': cells,
                'entropy': region_entropy,
                'center': (center_row, center_col),
                'coverage': coverage_ratio,
                'uncovered_cells': uncovered_cells,
                'value': value,
                'num_cells': len(cells)
            }
            
            region_id += 1
    
    return region_map, region_metadata


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
        self.tile_size = tuple(horizon_weights.get('tile_size', [20, 20]))
        self.hlp_replan_frequency = horizon_weights.get('hlp_replan_frequency', 5)
        
        # Weights dict for reward computation
        self.weights = {
            'w_coverage': self.w_coverage,
            'w_fragmentation': self.w_fragmentation,
            'w_revisit_cost': self.w_revisit_cost
        }
        
        # Step counter for logging
        self.step_counter = 0
        
        # Statistics tracking
        self.agreement_stats = {'YES': 0, 'PARTIAL': 0, 'DIFFERENT': 0}
        self.short_selected_count = 0
        self.long_selected_count = 0
        
        # Cache for HLP target region (to avoid replanning every step)
        self.cached_target_region_id = None
        self.cached_region_metadata = None
        self.steps_since_hlp_replan = 0
        
        logger.info(
            f"DualHorizonPlanner initialized: short_depth={self.short_horizon_depth}, "
            f"long_depth={self.long_horizon_depth}, weights={self.weights}"
        )
    
    def finalize_episode(self, final_state: Dict[str, Any]):
        """
        Log final episode statistics.
        
        Args:
            final_state: Final state of the episode
        """
        covered_mask = final_state.get('covered_mask')
        final_coverage = np.mean(covered_mask) if covered_mask is not None else 0.0
        
        # Calculate agreement percentages
        total_decisions = sum(self.agreement_stats.values())
        agreement_pct = {
            k: (v / total_decisions * 100) if total_decisions > 0 else 0
            for k, v in self.agreement_stats.items()
        }
        
        # Which planner's action was chosen more often
        short_pct = (self.short_selected_count / total_decisions * 100) if total_decisions > 0 else 0
        long_pct = (self.long_selected_count / total_decisions * 100) if total_decisions > 0 else 0
        
        log_summary_statistics(
            total_steps=self.step_counter,
            final_coverage=final_coverage,
            additional_stats={
                'Agreement Stats': self.agreement_stats,
                'Full Agreement %': f"{agreement_pct['YES']:.1f}%",
                'Partial Agreement %': f"{agreement_pct['PARTIAL']:.1f}%",
                'Disagreement %': f"{agreement_pct['DIFFERENT']:.1f}%",
                'Short Action Selected %': f"{short_pct:.1f}%",
                'Long Action Selected %': f"{long_pct:.1f}%",
                'Short Horizon Depth': self.short_horizon_depth,
                'Long Horizon Depth': self.long_horizon_depth,
                'Num Iterations': self.num_iterations,
            }
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
        print("\n[LLP - Short Horizon] Running IG-greedy MCTS...")
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
        
        print(f"[LLP] Selected action: {action}")
        print(f"[LLP] Action scores: {action_scores}")
        
        metrics = {
            'strategy': 'short',
            'action_scores': action_scores,
            'planning_depth': self.short_horizon_depth
        }
        
        return action, metrics
    
    def long_horizon_plan(
        self,
        state: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Run long-horizon region selection (HLP).
        
        Instead of selecting actions, HLP selects which REGION to explore next
        based on coverage optimization and fragmentation avoidance.
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            
        Returns:
            Tuple of (selected_region_id, metrics_dict)
        """
        print("\n[HLP - Long Horizon] Selecting target region...")
        
        # Partition field into regions
        belief = state['belief']
        covered_mask = state.get('covered_mask')
        region_map, region_metadata = partition_field(
            belief, 
            covered_mask, 
            self.tile_size
        )
        
        # Current UAV position
        uav_pos = state['uav_pos']
        
        # Convert spatial coordinates to grid indices using camera's conversion method
        # The camera has the grid info and knows how to convert (x,y) -> (i,j)
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0],  # x
            uav_pos.position[1],  # y
            self.camera.grid.center
        )
        
        # Current region
        H_dim, W_dim = belief.shape[:2]
        if 0 <= current_row < H_dim and 0 <= current_col < W_dim:
            current_region_id = int(region_map[current_row, current_col])
        else:
            current_region_id = -1
        
        # Identify regions containing isolated patches
        isolation_scores = self._identify_regions_with_isolated_patches(
            region_map, covered_mask, region_metadata
        )
        
        # Score each region for HLP selection
        region_scores = {}
        
        for region_id, metadata in region_metadata.items():
            # Base value: entropy * (1 - coverage)
            base_value = metadata['value']
            
            # Distance penalty: farther regions are less attractive
            center_row, center_col = metadata['center']
            distance = np.sqrt(
                (current_row - center_row)**2 + 
                (current_col - center_col)**2
            )
            max_distance = np.sqrt(H_dim**2 + W_dim**2)
            distance_penalty = distance / max_distance
            
            # Coverage bonus: prioritize uncovered regions
            coverage_bonus = (1.0 - metadata['coverage']) * self.w_coverage
            
            # Isolation bonus: STRONGLY prioritize isolated patches
            # Higher w_fragmentation means we care more about covering isolated patches
            isolation_score = isolation_scores.get(region_id, 0.0)
            isolation_bonus = isolation_score * self.w_fragmentation * 2.0  # 2x multiplier for strong effect
            
            # Combined score
            # Note: distance_penalty weight of 0.8 ensures we strongly prefer nearby regions
            # Only switch to distant region if it's significantly more valuable
            score = (
                base_value + 
                coverage_bonus + 
                isolation_bonus -
                0.8 * distance_penalty
            )
            
            region_scores[region_id] = score
        
        # Select region with highest score
        if region_scores:
            selected_region_id = max(region_scores.keys(), key=lambda r: region_scores[r])
            selected_metadata = region_metadata[selected_region_id]
        else:
            selected_region_id = 0
            selected_metadata = region_metadata.get(0, {})
        
        # Store for visualization
        self.current_region_metadata = region_metadata
        self.current_selected_region = selected_region_id
        self.current_region_scores = region_scores
        
        # Analyze current coverage state
        fragmentation_info = analyze_coverage_fragmentation(covered_mask)
        
        print(f"[HLP] Selected region: {selected_region_id}")
        print(f"[HLP] Region center: {selected_metadata.get('center', 'N/A')}")
        print(f"[HLP] Region coverage: {selected_metadata.get('coverage', 0):.3f}")
        print(f"[HLP] Region entropy: {selected_metadata.get('entropy', 0):.3f}")
        print(f"[HLP] Region scores (top 5):")
        for rid in sorted(region_scores.keys(), key=lambda r: region_scores[r], reverse=True)[:5]:
            iso_score = isolation_scores.get(rid, 0.0)
            iso_str = f" [ISOLATED: {iso_score:.3f}]" if iso_score > 0 else ""
            print(f"      Region {rid}: {region_scores[rid]:.4f}{iso_str}")
        print(f"[HLP] Fragmentation - patches: {fragmentation_info['num_patches']}, "
              f"score: {fragmentation_info['fragmentation_score']:.3f}")
        if isolation_scores:
            print(f"[HLP] Isolated regions: {isolation_scores}")
        
        metrics = {
            'strategy': 'long',
            'selected_region_id': selected_region_id,
            'region_scores': region_scores,
            'region_metadata': region_metadata,
            'planning_depth': self.long_horizon_depth,
            'fragmentation_info': fragmentation_info,
            'current_region_id': current_region_id
        }
        
        return selected_region_id, metrics
    
    def _identify_regions_with_isolated_patches(
        self,
        region_map: np.ndarray,
        covered_mask: np.ndarray,
        region_metadata: Dict[int, Dict]
    ) -> Dict[int, float]:
        """
        Identify which regions contain isolated uncovered patches.
        
        Returns a dict mapping region_id -> isolation_score, where
        higher scores indicate the region contains a smaller isolated patch
        that should be prioritized for coverage.
        
        Args:
            region_map: Map of region assignments
            covered_mask: Current coverage mask
            region_metadata: Metadata for all regions
            
        Returns:
            Dict[region_id, isolation_score] where 0 = not isolated, 1 = most isolated
        """
        if covered_mask is None:
            return {}
        
        # Analyze fragmentation to get patches
        fragmentation_info = analyze_coverage_fragmentation(covered_mask)
        num_patches = fragmentation_info['num_patches']
        
        if num_patches <= 1:
            # No fragmentation or fully uncovered
            return {}
        
        # Get uncovered mask and labeled patches
        uncovered_mask = ~covered_mask
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_array, _ = ndimage.label(uncovered_mask, structure=structure)
        
        # For each patch, find which region contains most of it
        patch_sizes = fragmentation_info['patch_sizes']
        total_uncovered = fragmentation_info['total_uncovered']
        
        region_isolation_scores = {}
        
        for patch_id in range(1, num_patches + 1):
            patch_mask = (labeled_array == patch_id)
            patch_size = patch_sizes[patch_id - 1]
            
            # Smaller isolated patches should get higher priority
            # Normalize by total uncovered area
            patch_isolation = 1.0 - (patch_size / total_uncovered)
            
            # Find which regions overlap with this patch
            regions_in_patch = {}
            for region_id in region_metadata.keys():
                region_mask = (region_map == region_id)
                overlap = np.logical_and(patch_mask, region_mask)
                overlap_size = np.sum(overlap)
                
                if overlap_size > 0:
                    regions_in_patch[region_id] = overlap_size
            
            # Assign isolation score to region with most overlap
            if regions_in_patch:
                best_region = max(regions_in_patch.keys(), key=lambda r: regions_in_patch[r])
                # Use the highest isolation score if region overlaps multiple patches
                current_score = region_isolation_scores.get(best_region, 0.0)
                region_isolation_scores[best_region] = max(current_score, patch_isolation)
        
        return region_isolation_scores
    
    def _compute_region_fragmentation_impact(
        self,
        region_id: int,
        region_map: np.ndarray,
        covered_mask: np.ndarray,
        metadata: Dict
    ) -> float:
        """
        Estimate how much covering this region would impact fragmentation.
        
        Regions that connect existing covered areas have lower scores (better).
        Regions that create new isolated patches have higher scores (worse).
        
        Args:
            region_id: ID of region to evaluate
            region_map: Map of region assignments
            covered_mask: Current coverage mask
            metadata: Region metadata
            
        Returns:
            Fragmentation impact score (lower is better)
        """
        if covered_mask is None:
            return 0.0
        
        # Get region bounds
        (row_min, row_max), (col_min, col_max) = metadata['bounds']
        
        # Check coverage of neighboring regions
        H_dim, W_dim = covered_mask.shape
        
        # Expand bounds to check neighbors
        neighbor_row_min = max(0, row_min - 1)
        neighbor_row_max = min(H_dim, row_max + 1)
        neighbor_col_min = max(0, col_min - 1)
        neighbor_col_max = min(W_dim, col_max + 1)
        
        # Get coverage around this region
        neighborhood = covered_mask[
            neighbor_row_min:neighbor_row_max,
            neighbor_col_min:neighbor_col_max
        ]
        
        # If surrounded by covered areas, this region connects them (good)
        # If surrounded by uncovered areas, this creates isolation (bad)
        neighbor_coverage = np.mean(neighborhood) if neighborhood.size > 0 else 0.0
        
        # Higher neighbor coverage = lower fragmentation impact
        fragmentation_impact = 1.0 - neighbor_coverage
        
        return fragmentation_impact
    
    def dual_horizon_plan(
        self,
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Combine short and long horizon planning.
        
        Hierarchical approach:
        1. HLP selects target REGION based on coverage optimization (every N steps)
        2. LLP selects best ACTION to move toward that region using IG
        3. Alignment score measures how well LLP action moves toward HLP region
        
        Args:
            state: Dict with 'uav_pos', 'belief', 'covered_mask'
            
        Returns:
            Tuple of (action, metrics_dict)
        """
        # Run LLP every step
        short_action, short_metrics = self.short_horizon_plan(state)
        
        # Check if we need to replan HLP
        # Reasons to replan:
        # 1. No cached target (first step)
        # 2. Target region is sufficiently covered (>80%)
        # 3. Not making progress - 5 steps passed without reaching target
        need_hlp_replan = False
        replan_reason = ""
        
        if self.cached_target_region_id is None:
            need_hlp_replan = True
            replan_reason = "First step"
        else:
            # Get current coverage of target region
            covered_mask = state.get('covered_mask')
            if covered_mask is not None and self.cached_region_metadata is not None:
                target_metadata = self.cached_region_metadata.get(self.cached_target_region_id, {})
                bounds = target_metadata.get('bounds')
                
                if bounds is not None:
                    (row_min, row_max), (col_min, col_max) = bounds
                    region_coverage_mask = covered_mask[row_min:row_max, col_min:col_max]
                    current_coverage = float(np.mean(region_coverage_mask))
                    
                    # Replan if target region is well covered (90% threshold)
                    # Higher threshold ensures we finish regions before switching
                    if current_coverage > 1.:
                        need_hlp_replan = True
                        replan_reason = f"Target region {self.cached_target_region_id} covered ({current_coverage:.1%})"
        
        if need_hlp_replan:
            # Run HLP and cache the result
            target_region_id, long_metrics = self.long_horizon_plan(state)
            self.cached_target_region_id = target_region_id
            self.cached_region_metadata = long_metrics['region_metadata']
            self.steps_since_hlp_replan = 0
            print(f"[HLP] REPLANNING - Reason: {replan_reason} → New target region: {target_region_id}")
        else:
            # Reuse cached HLP target
            target_region_id = self.cached_target_region_id
            region_metadata = self.cached_region_metadata
            
            # Build a minimal long_metrics dict
            long_metrics = {
                'strategy': 'long',
                'selected_region_id': target_region_id,
                'region_metadata': region_metadata,
                'cached': True
            }
            self.steps_since_hlp_replan += 1
            print(f"[HLP] REUSING cached target region: {target_region_id} (steps since replan: {self.steps_since_hlp_replan})")
        
        # Get region metadata
        region_metadata = long_metrics['region_metadata']
        target_region = region_metadata[target_region_id]
        target_center = target_region['center']
        
        # Current UAV position - convert to grid indices
        uav_pos = state['uav_pos']
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0],  # x
            uav_pos.position[1],  # y
            self.camera.grid.center
        )
        current_pos = np.array([current_row, current_col])  # [row, col] in grid indices
        target_pos = np.array(target_center)  # [row, col] in grid indices
        
        # Get LLP action scores
        short_scores = short_metrics.get('action_scores', {})
        
        # Compute alignment scores: how well each action moves toward target region
        # Only consider HORIZONTAL movement (x, y) - let LLP control altitude freely
        alignment_scores = {}
        raw_alignments = {}
        
        for action, ig_score in short_scores.items():
            # Altitude changes should not be influenced by alignment
            # Let LLP exploit IG freely in vertical dimension
            if action in ['up', 'down']:
                alignment_scores[action] = 0.0
                raw_alignments[action] = 0.0
                continue
            
            # Get future position for this action
            next_state = self.camera.x_future(action, x=uav_pos)
            if next_state is None:
                alignment_scores[action] = 0.0
                raw_alignments[action] = 0.0
                continue
            
            next_uav_pos = uav_position(next_state)
            # Convert next position to grid indices
            next_row, next_col = self.camera.convert_xy_ij(
                next_uav_pos.position[0],  # x
                next_uav_pos.position[1],  # y
                self.camera.grid.center
            )
            next_pos = np.array([next_row, next_col])  # [row, col] in grid indices
            
            # Calculate distance before and after action (both in grid index space)
            dist_before = np.linalg.norm(current_pos - target_pos)
            dist_after = np.linalg.norm(next_pos - target_pos)
            
            # Alignment score: positive if moving closer, negative if moving away
            # Raw alignment in grid cells
            alignment = dist_before - dist_after
            raw_alignments[action] = alignment
            
            # Normalize to be comparable to IG scores (which are typically 0.04-0.12)
            # Typical alignment values are 10-25 grid cells, IG values are 0.04-0.12
            # Scale factor: ~200-300x to make them comparable
            # Use a softer scaling: divide by max distance to normalize to [0, 1] range
            H_dim, W_dim = state['belief'].shape[:2]
            max_possible_distance = np.sqrt(H_dim**2 + W_dim**2)
            normalized_alignment = alignment / max_possible_distance
            
            alignment_scores[action] = normalized_alignment
        
        # Calculate blending weights based on current state
        blend_weights = self._compute_blend_weights(state)
        
        # Combine scores: IG (short) + Alignment (long) weighted by blend factors
        combined_scores = {}
        all_actions = set(short_scores.keys())
        
        print("\n[BLENDING] Combining LLP actions with HLP region guidance:")
        print(f"  HLP target region: {target_region_id} at {target_center}")
        print(f"  Current position (grid indices): {current_pos}")
        print(f"  Current UAV (spatial): ({uav_pos.position[0]:.2f}, {uav_pos.position[1]:.2f}, alt={uav_pos.altitude:.2f})")
        print(f"  Distance to target: {np.linalg.norm(current_pos - target_pos):.2f} grid cells")
        print(f"  Weights: w_short={blend_weights['w_short']:.3f}, w_long={blend_weights['w_long']:.3f}")
        print(f"  Note: Altitude (up/down) controlled by LLP IG only")
        
        for action in all_actions:
            ig_val = short_scores.get(action, 0.0)
            align_val = alignment_scores.get(action, 0.0)
            raw_align = raw_alignments.get(action, 0.0)
            
            # Combined score: IG exploitation + alignment to HLP region
            combined_val = (
                blend_weights['w_short'] * ig_val +
                blend_weights['w_long'] * align_val
            )
            combined_scores[action] = combined_val
            
            # Log details for debugging
            if action in ['up', 'down']:
                print(f"  {action:6s}: IG={ig_val:.4f} × {blend_weights['w_short']:.3f} = {combined_val:.4f} [altitude: IG only]")
            else:
                print(f"  {action:6s}: IG={ig_val:.4f} × {blend_weights['w_short']:.3f} + Align={align_val:.4f} (raw={raw_align:.1f}) × {blend_weights['w_long']:.3f} = {combined_val:.4f}")
        
        # Select action with highest combined score
        if combined_scores:
            selected_action = max(combined_scores.keys(), key=lambda a: combined_scores[a])
        else:
            # Fallback to short horizon action
            selected_action = short_action
        
        # Determine agreement type
        best_aligned_action = max(alignment_scores.keys(), key=lambda a: alignment_scores[a]) if alignment_scores else None
        agreement_type = ('YES' if selected_action == short_action == best_aligned_action
                         else 'PARTIAL' if selected_action in [short_action, best_aligned_action]
                         else 'DIFFERENT')
        
        print("\n" + "="*70)
        print("[DUAL HORIZON] Blending Results:")
        print(f"  LLP (short) suggests: {short_action} (IG-greedy)")
        print(f"  HLP (long) guides to: Region {target_region_id}")
        print(f"  Best aligned action: {best_aligned_action}")
        print(f"  Blend weights: w_short={blend_weights['w_short']:.3f}, w_long={blend_weights['w_long']:.3f}")
        print(f"  Coverage progress: {blend_weights['coverage_progress']:.3f}")
        print(f"  Uncertainty ratio: {blend_weights['uncertainty_ratio']:.3f}")
        print(f"  Fragmentation score: {blend_weights['fragmentation_score']:.3f}")
        print(f"  FINAL SELECTED ACTION: {selected_action}")
        print(f"  Agreement: {agreement_type}")
        print("="*70 + "\n")
        
        # Increment step counter
        self.step_counter += 1
        
        # Track statistics
        self.agreement_stats[agreement_type] += 1
        
        if selected_action == short_action:
            self.short_selected_count += 1
        if selected_action == best_aligned_action:
            self.long_selected_count += 1
        
        # Log to dedicated file
        log_planning_step(
            step_num=self.step_counter,
            state=state,
            short_action=short_action,
            short_scores=short_scores,
            long_action=f"Region_{target_region_id}",
            long_scores=alignment_scores,  # Use alignment scores for action-level blending
            blend_weights=blend_weights,
            combined_scores=combined_scores,
            selected_action=selected_action,
            fragmentation_info=long_metrics.get('fragmentation_info', {}),
            additional_info={
                'short_horizon_depth': self.short_horizon_depth,
                'long_horizon_depth': self.long_horizon_depth,
                'num_iterations': self.num_iterations,
                'agreement': agreement_type,
                'target_region_id': target_region_id,
                'target_region_center': target_center,
                'region_scores': long_metrics.get('region_scores', {}),
                'alignment_scores': alignment_scores,
                'best_aligned_action': best_aligned_action
            }
        )
        
        metrics = {
            'strategy': 'dual',
            'selected_action': selected_action,
            'short_action': short_action,
            'target_region_id': target_region_id,
            'target_region': target_region,
            'short_scores': short_scores,
            'region_scores': long_metrics.get('region_scores', {}),
            'alignment_scores': alignment_scores,
            'combined_scores': combined_scores,
            'blend_weights': blend_weights,
            'fragmentation_info': long_metrics.get('fragmentation_info', {}),
            'agreement': agreement_type
        }
        
        logger.debug(
            f"Dual-horizon: LLP={short_action}, HLP=Region{target_region_id}, "
            f"selected={selected_action}, blend={blend_weights}, agreement={agreement_type}"
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
        # Early mission: boost long-horizon slightly (reduced factor)
        coverage_adjustment = 0.15 * (1.0 - coverage_progress)  # Reduced from 0.3
        
        # High uncertainty: boost short-horizon for IG exploitation
        uncertainty_adjustment = UNCERTAINTY_ADJUSTMENT_FACTOR * uncertainty_ratio
        
        # High fragmentation: boost long-horizon to reach isolated patches
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

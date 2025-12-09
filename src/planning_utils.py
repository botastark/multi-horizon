"""
Shared Planning Utilities for Multi-Agent UAV Coverage Planning

This module contains common functions used across different planning strategies:
- Coverage fragmentation analysis
- Field partitioning for hierarchical planning
- Revisit cost estimation
- Entropy and information gain helpers

These utilities are used by:
- greedy_ig_planner.py (Greedy IG baseline)
- dec_mcts.py (Dec-MCTS planner)
- hierarchical_dec_mcts.py (Multi-Horizon Dec-MCTS)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from helper import H

# =============================================================================
# Constants for fragmentation analysis and blend weight computation
# =============================================================================

# Fragmentation score normalization: patches above this count have maximum penalty
FRAGMENTATION_PATCH_THRESHOLD = 10

# Blend weight adjustment factors for dual-horizon planning
# These control how much each factor affects the short/long horizon balance
COVERAGE_ADJUSTMENT_FACTOR = 0.3  # How much low coverage boosts long-horizon
UNCERTAINTY_ADJUSTMENT_FACTOR = 0.2  # How much high uncertainty boosts short-horizon
FRAGMENTATION_ADJUSTMENT_FACTOR = 0.3  # How much fragmentation boosts long-horizon


# =============================================================================
# Coverage Analysis Functions
# =============================================================================


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
            "num_patches": 0,
            "patch_sizes": [],
            "patch_centroids": [],
            "total_uncovered": 0,
            "fragmentation_score": 0.0,
        }

    # Invert mask to get uncovered areas
    uncovered_mask = ~covered_mask

    # Label connected components in uncovered regions
    # Using 8-connectivity for diagonal neighbors
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_array, num_patches = ndimage.label(uncovered_mask, structure=structure)

    if num_patches == 0:
        return {
            "num_patches": 0,
            "patch_sizes": [],
            "patch_centroids": [],
            "total_uncovered": 0,
            "fragmentation_score": 0.0,
        }

    # Calculate size and centroid of each patch
    patch_sizes = []
    patch_centroids = []

    for label_id in range(1, num_patches + 1):
        patch_mask = labeled_array == label_id
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
            (num_patches / FRAGMENTATION_PATCH_THRESHOLD)
            * (1.0 - avg_patch_size / total_cells),
        )
    else:
        fragmentation_score = 0.0

    return {
        "num_patches": num_patches,
        "patch_sizes": patch_sizes,
        "patch_centroids": patch_centroids,
        "total_uncovered": int(total_uncovered),
        "fragmentation_score": fragmentation_score,
    }


def compute_revisit_cost(
    uav_pos: Any,  # uav_position object
    uncovered_patches: Dict[str, Any],
    uav_speed: float = 1.0,
    grid_length: float = 1.0,
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
    if uncovered_patches["num_patches"] == 0:
        return 0.0

    total_cost = 0.0
    current_pos = np.array(uav_pos.position)

    for i, (centroid, size) in enumerate(
        zip(uncovered_patches["patch_centroids"], uncovered_patches["patch_sizes"])
    ):
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
    tile_size: Tuple[int, int] = (20, 20),
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
            # Balance: high entropy regions are valuable even if partially covered
            # Instead of zeroing out covered regions, use entropy as primary metric
            # and only lightly penalize coverage
            avg_entropy = region_entropy / len(cells) if cells else 0
            coverage_penalty = (
                0.1 * coverage_ratio
            )  # Light penalty for coverage (10% per 100% coverage)
            value = avg_entropy * (1.0 - coverage_penalty)

            region_metadata[region_id] = {
                "bounds": ((row_start, row_end), (col_start, col_end)),
                "cells": cells,
                "entropy": region_entropy,
                "center": (center_row, center_col),
                "coverage": coverage_ratio,
                "uncovered_cells": uncovered_cells,
                "value": value,
                "num_cells": len(cells),
            }

            region_id += 1

    return region_map, region_metadata


# =============================================================================
# Helper functions for entropy and information gain
# =============================================================================


def compute_expected_ig(
    belief: np.ndarray,
    footprint: Tuple[int, int, int, int],
    s0: float,
    s1: float,
) -> float:
    """
    Compute expected information gain for a given footprint.

    Args:
        belief: 2D belief map (probability of occupancy)
        footprint: (imin, imax, jmin, jmax) bounds
        s0: False positive rate
        s1: False negative rate

    Returns:
        Total expected IG for the footprint
    """
    from helper import cH

    imin, imax, jmin, jmax = footprint
    region_belief = belief[imin:imax, jmin:jmax]

    prior_entropy = H(region_belief)
    conditional_entropy = cH(region_belief, s0, s1)

    ig = prior_entropy - conditional_entropy
    return float(np.sum(ig))


def compute_field_entropy(belief_map: np.ndarray) -> float:
    """
    Compute total entropy across the entire field.

    Args:
        belief_map: 3D array of shape (H, W, 2) with belief probabilities

    Returns:
        Total entropy
    """
    return float(np.sum(H(belief_map[:, :, 1])))


def compute_coverage_ratio(covered_mask: np.ndarray) -> float:
    """
    Compute the ratio of covered cells to total cells.

    Args:
        covered_mask: Boolean 2D array where True indicates covered cells

    Returns:
        Coverage ratio in [0, 1]
    """
    if covered_mask is None or covered_mask.size == 0:
        return 0.0
    return float(np.mean(covered_mask))

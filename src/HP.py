import math
import numpy as np
from uav_camera import UAVCamera


# class grid_info:
#     x = 50
#     y = 50
#     length = 0.125
#     shape = (int(y / length), int(x / length))
#     center = True


def build_clusters(
    belief: np.ndarray, grid_info: dict, h_max: float, step_xy: float, camera
):
    """
    belief: (H, W, 2), probs in [:,:,1]
    Returns:
        clusters: {cid: {'cells': list[(i,j)], 'entropy': float, 'center_xy': (x,y)}}
    """
    Hh, Ww = belief.shape[:2]
    fov = 60
    fov = camera.fov  # Use camera's FOV

    cell_length = grid_info["length"]
    # Get field boundaries from camera
    x_range = camera.x_range
    y_range = camera.y_range

    # Calculate footprint radius at max altitude
    footprint_half_size = h_max * math.tan(math.radians(fov / 2))
    footprint_cells = int(footprint_radius / cell_length)

    # Field dimensions in meters
    fx, fy = grid_info["x"], grid_info["y"]

    # Initialize
    cid_of_cell = -np.ones((Hh, Ww), dtype=np.int32)
    clusters: Dict[int, dict] = {}
    cid = 0
    # Generate UAV positions at step_xy intervals
    for uav_x in np.arange(x_range[0], x_range[1] + step_xy, step_xy):
        for uav_y in np.arange(y_range[0], y_range[1] + step_xy, step_xy):
            # Ensure position is within bounds
            if not (
                x_range[0] <= uav_x <= x_range[1] and y_range[0] <= uav_y <= y_range[1]
            ):
                continue

            # Convert UAV position to grid indices using camera's method
            center_i, center_j = camera.convert_xy_ij(uav_x, uav_y, grid_info["center"])

            # Skip if center is out of grid bounds
            if not (0 <= center_i < Hh and 0 <= center_j < Ww):
                continue

            # Collect cells within footprint
            cells = []
            ent_mass = 0.0

            for i in range(
                max(0, center_i - footprint_cells),
                min(Hh, center_i + footprint_cells + 1),
            ):
                for j in range(
                    max(0, center_j - footprint_cells),
                    min(Ww, center_j + footprint_cells + 1),
                ):
                    # Convert cell indices to field coordinates
                    cell_x, cell_y = camera.ij_to_xy(i, j)

                    # Check if cell is within circular footprint
                    dx = cell_x - uav_x
                    dy = cell_y - uav_y
                    if dx <= footprint_half_size and dy <= footprint_half_size:
                        cells.append((i, j))
                        p = belief[i, j, 1]
                        if 0 < p < 1:
                            ent_mass += -(p * math.log(p) + (1 - p) * math.log(1 - p))

            # Store cluster information

            # Only create cluster if it contains cells
            if len(cells) > 0:
                clusters[cid] = {
                    "cells": cells,
                    "entropy": ent_mass,
                    "center_ij": (center_i, center_j),
                    "uav_pos": (uav_x, uav_y),  # UAV position in meters
                    "seen": False,  # Mark if area is covered
                    "cost": 0.0,  # Cost to reach (calculate later)
                }

                # Mark cells with cluster ID
                for i, j in cells:
                    if cid_of_cell[i, j] == -1:  # Only assign if unassigned
                        cid_of_cell[i, j] = cid

                cid += 1

    footprint_size = 2 * footprint_half_size
    print(
        f"Built {cid} clusters with rectangular footprint {footprint_size:.2f}m x {footprint_size:.2f}m "
        f"(~{2*footprint_cells} x {2*footprint_cells} cells) at altitude {h_max}m"
    )

    return cid_of_cell, clusters

from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported

from matplotlib import cm
from matplotlib.colors import Normalize


def plot_terrain_2d(filename, grid, ground_truth):
    """
    Plot a 2D visualization of the terrain (ground truth) and save to file.

    Args:
        save_path (str): File path to save the generated figure.
        grid (object): Grid configuration with attributes 'center', 'x', 'y', and 'length'.
        ground_truth (np.ndarray): 2D binary map representing the ground truth.
    """
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

    # Determine x/y ranges
    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    else:
        x_range = [0, grid.x]
        y_range = [0, grid.y]

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)

    # Create a discrete colormap for ground truth
    cmap = colors.ListedColormap(["lemonchiffon", "darkgreen"])
    ax.imshow(
        ground_truth.T,
        cmap=cmap,
        origin="lower",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
    )
    # Set custom tick labels
    ticks_x = np.linspace(x_range[0], x_range[1], 5)
    ticks_y = np.linspace(y_range[0], y_range[1], 5)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks_x])
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks_y])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_terrain(
    save_path, belief, grid, uav_pos, ground_truth, submap, obs, fp, h_range,
    region_metadata=None, selected_region_id=None, region_scores=None
):
    """
    Plot a comprehensive figure with four subplots:
    1. 3D terrain with UAV path.
    2. 2D last observation overlay.
    3. Belief map.
    4. Ground truth in grid (i,j) coordinates.

    Args:
        save_path (str): Path to save the figure.
        belief (np.ndarray): Belief map (either 2D or 3D with probability channel at index 1).
        grid (object): Grid configuration with attributes 'center', 'x', 'y', and 'length'.
        uav_positions (list): List of UAV state objects with attributes 'position' and 'altitude'.
        ground_truth (np.ndarray): Ground truth binary map.
        submap (np.ndarray): Latest observation submap.
        obs (list): [[x_min, x_max], [y_min, y_max]] bounds of the observation.
        fp (dict): Dictionary with footprint vertices in grid coordinates (keys: 'ul', 'bl', 'br', 'ur').
        region_metadata (dict): Optional region metadata for dual-horizon visualization.
        selected_region_id (int): Optional ID of the region selected by HLP.
    """
    # Create figure with 4 subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    for ax in axes:
        ax.set_axis_off()
    # Unpack observation bounds and create polygon coordinates
    [ox_min, ox_max], [oy_min, oy_max] = obs
    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

    # ---- Subplot 1: 3D Terrain with UAV Path ----
    ax1 = fig.add_subplot(141, projection="3d")
    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    else:
        x_range = [0, grid.x]
        y_range = [0, grid.y]

    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim([0, h_range[1]])

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Altitude (m)")
    ax1.set_title("Truth Terrain and UAV position")
    ax1.xaxis.grid(visible=True)

    # Prepare UAV path data and plot with a color gradient

    uav_x, uav_y, uav_z = zip(
        *[(uav.position[0], uav.position[1], uav.altitude) for uav in uav_pos]
    )

    cmap_uav = plt.get_cmap("cool")
    norm = Normalize(vmin=0, vmax=1)
    colors_list = np.linspace(0, 1, len(uav_pos))

    for i in range(len(uav_x) - 1):
        ax1.plot(
            uav_x[i : i + 2],
            uav_y[i : i + 2],
            uav_z[i : i + 2],
            color=cmap_uav(norm(colors_list[i])),
            linewidth=2,
        )
    z_max = max(35, max(uav_z))
    ax1.set_zlim([0, z_max])

    # Plot the ground truth terrain as a flat surface at z=0
    x_vals = np.arange(x_range[0], x_range[1], grid.length)
    y_vals = np.arange(y_range[0], y_range[1], grid.length)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    terrain_colors = np.where(ground_truth == 0, "yellow", "darkgreen")

    ax1.plot_surface(
        X.T,
        -Y.T,
        np.zeros_like(X.T),
        facecolors=terrain_colors,
        alpha=0.3,
        edgecolor="none",
    )
    # Plot observation polygon slightly above terrain
    o_z = np.zeros_like(o_x) + 0.01  # Slightly above z=0
    ax1.plot(o_x, o_y, o_z, color="red", lw=1)

    # ---- Subplot 2: 2D Last Observation ----
    ax2 = fig.add_subplot(142)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("last observation z_t")
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    cmap = colors.ListedColormap(["lemonchiffon", "darkgreen"])

    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax2.imshow(
        submap,
        cmap=cmap,
        norm=norm,
        extent=[ox_min, ox_max, oy_min, oy_max],
        origin="upper",
    )
    ax2.plot(o_x, o_y, color="red", lw=0.9)

    # ---- Subplot 3: Belief Map ----
    ax3 = fig.add_subplot(143)
    ax3.set_xlabel("j-axis")
    ax3.set_ylabel("i-axis")
    ax3.set_title("Belief sampled map M")

    belief_map = belief[:, :, 1] if belief.ndim == 3 else belief
    # Create a continuous colormap going from dark green (0) to lemonchiffon (1)
    colors_list = ["lemonchiffon", "darkgreen"]

    green_yellow_cmap = colors.LinearSegmentedColormap.from_list(
        "GreenYellow", colors_list
    )

    ax3.imshow(belief_map, cmap=green_yellow_cmap, origin="upper", vmin=0, vmax=1)

    # ---- Subplot 4: Ground Truth in Grid Indices ----
    ax4 = fig.add_subplot(144)
    ax4.set_xlabel("j-axis")
    ax4.set_ylabel("i-axis")
    ax4.set_title("Ground Truth in ij")

    ax4.imshow(
        ground_truth,
        cmap=cmap,
        norm=norm,
        origin="upper",
    )
    ax4.set_xlim(0, ground_truth.shape[1])
    ax4.set_ylim(ground_truth.shape[0], 0)
    I, J = 0, 1
    o_i = [fp["ul"][I], fp["bl"][I], fp["br"][I], fp["ur"][I], fp["ul"][I]]
    o_j = [fp["ul"][J], fp["bl"][J], fp["br"][J], fp["ur"][J], fp["ul"][J]]
    ax4.plot(o_j, o_i, color="red", lw=0.9)
    
    # Add region visualization for dual-horizon planning
    if region_metadata is not None:
        print(f"[DEBUG VIEWER] Plotting {len(region_metadata)} regions on Ground Truth subplot, selected: {selected_region_id}")
        
        # Get top 5 regions by score (if scores available)
        top_5_regions = set()
        if region_scores is not None and len(region_scores) > 0:
            sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
            top_5_regions = {rid for rid, score in sorted_regions[:5]}
            print(f"[DEBUG VIEWER] Top 5 regions: {[rid for rid, _ in sorted_regions[:5]]}")
        
        for region_id, metadata in region_metadata.items():
            center = metadata['center']
            bounds = metadata['bounds']
            
            # Bounds are already in grid (i,j) coordinates
            (row_min, row_max), (col_min, col_max) = bounds
            
            # Rectangle dimensions in grid indices
            rect_width = col_max - col_min
            rect_height = row_max - row_min
            
            # Determine color based on selection and ranking
            if region_id == selected_region_id:
                edge_color = 'red'
                line_width = 2.5
                alpha = 0.9
            elif region_id in top_5_regions:
                edge_color = 'cyan'
                line_width = 2.0
                alpha = 0.75
            else:
                edge_color = 'yellow'
                line_width = 1.5
                alpha = 0.6
            
            # Draw rectangle (note: ax4 uses j for x-axis, i for y-axis)
            rect = patches.Rectangle(
                (col_min, row_min),  # (j, i) coordinates
                rect_width,
                rect_height,
                linewidth=line_width,
                edgecolor=edge_color,
                facecolor='none',
                alpha=alpha
            )
            ax4.add_patch(rect)
            
            # Draw center point (center is in (row, col) format)
            center_j = center[1]  # col
            center_i = center[0]  # row
            
            # Color based on rank
            if region_id == selected_region_id:
                marker_color = 'red'
                marker_size = 10
            elif region_id in top_5_regions:
                marker_color = 'cyan'
                marker_size = 8
            else:
                marker_color = 'yellow'
                marker_size = 6
            
            ax4.plot(center_j, center_i, 'x', 
                    color=marker_color,
                    markersize=marker_size,
                    markeredgewidth=2,
                    alpha=0.9)
            
            # Add region ID label at center
            if region_id == selected_region_id:
                label_facecolor = 'red'
                label_fontsize = 9
                label_weight = 'bold'
            elif region_id in top_5_regions:
                label_facecolor = 'blue'
                label_fontsize = 8
                label_weight = 'bold'
            else:
                label_facecolor = 'gray'
                label_fontsize = 7
                label_weight = 'normal'
            
            ax4.text(center_j, center_i + rect_height * 0.1, str(region_id),
                    color='white',
                    fontsize=label_fontsize,
                    ha='center', va='center',
                    weight=label_weight,
                    bbox=dict(boxstyle='round,pad=0.4', 
                             facecolor=label_facecolor,
                             alpha=0.8,
                             edgecolor='white',
                             linewidth=1))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_metrics(save_path, entropy_list, mse_list, coverage_list, height_list, height_range=None):
    """
    Plot metrics (entropy, MSE, coverage, height) over time and save the resulting figure.

    Args:
        save_dir (str): Directory or filename where the plot will be saved.
        entropy_list (list): List of entropy values.
        mse_list (list): List of mean squared error values.
        coverage_list (list): List of coverage values.
        height_list (list): List of UAV height values.
        height_range (tuple): Optional (min_height, max_height) for y-axis limits.
    """
    # Ensure all metric lists have the same length
    assert len(entropy_list) == len(mse_list) == len(coverage_list) == len(height_list)

    steps = range(len(entropy_list))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax1, ax2), (ax3, ax4) = axes

    # Plot entropy in the first subplot
    ax1.plot(steps, entropy_list, "bo-", label="Entropy", markersize=5)
    ax1.set_xlabel("Number of steps")
    ax1.set_ylabel("Entropy")
    ax1.set_title("Entropy over Steps")
    ax1.grid(True)

    # Plot MSE in the second subplot
    ax2.plot(steps, mse_list, "r*-", label="MSE", markersize=5)
    ax2.set_xlabel("Number of steps")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE over Steps")
    ax2.grid(True)

    # Plot covberage in the second subplot
    ax3.plot(steps, coverage_list, "g*-", label="Coverage", markersize=5)
    ax3.set_xlabel("Number of steps")
    ax3.set_ylabel("Coverage")
    ax3.set_title("Coverage over Steps")
    ax3.grid(True)

    # Plot height in the fourth subplot
    ax4.plot(steps, height_list, "m^-", label="Height", markersize=5)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Height")
    ax4.set_title("Height over Steps")
    ax4.grid(True)
    
    # Set height y-axis limits if provided
    if height_range is not None:
        min_h, max_h = height_range
        # Add small padding (5%) for better visualization
        padding = (max_h - min_h) * 0.05
        ax4.set_ylim(min_h - padding, max_h + padding)
        # Add horizontal lines for min and max limits
        ax4.axhline(y=min_h, color='gray', linestyle='--', alpha=0.7, label=f'Min: {min_h:.1f}')
        ax4.axhline(y=max_h, color='gray', linestyle='--', alpha=0.7, label=f'Max: {max_h:.1f}')
        ax4.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save_path.endswith(".png"):
        plt.savefig(save_path)
    else:
        plt.savefig(f"{save_path}/final.png")
    plt.close(fig)

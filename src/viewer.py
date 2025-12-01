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
    save_path,
    belief,
    grid,
    uav_pos,
    ground_truth,
    submap,
    obs,
    fp,
    h_range,
    region_metadata=None,
    selected_region_id=None,
    region_scores=None,
    multi_agent=False,
    per_agent_data=None,
):
    """
    Plot a comprehensive figure with four subplots:
    1. 3D terrain with UAV path.
    2. 2D last observation overlay.
    3. Belief map.
    4. Ground truth in grid (i,j) coordinates.

    For multi-agent mode, additional rows show per-agent observations and beliefs.

    Args:
        save_path (str): Path to save the figure.
        belief (np.ndarray): Belief map (either 2D or 3D with probability channel at index 1).
        grid (object): Grid configuration with attributes 'center', 'x', 'y', and 'length'.
        uav_pos (list): List of UAV state objects OR list of lists for multi-agent.
        ground_truth (np.ndarray): Ground truth binary map.
        submap (np.ndarray): Latest observation submap.
        obs (list): [[x_min, x_max], [y_min, y_max]] bounds of the observation.
        fp (dict): Dictionary with footprint vertices in grid coordinates (keys: 'ul', 'bl', 'br', 'ur').
        region_metadata (dict): Optional region metadata for dual-horizon visualization.
        selected_region_id (int): Optional ID of the region selected by HLP.
        region_scores (dict): Optional region scores for visualization.
        multi_agent (bool): If True, uav_pos is a list of lists (one per agent).
        per_agent_data (list): List of dicts with per-agent observation and belief data.
    """
    # Agent colors for multi-agent visualization
    AGENT_COLORS = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
    ]

    # Determine number of rows based on multi-agent mode
    num_agents = len(per_agent_data) if per_agent_data else 0
    num_rows = 1 + num_agents if multi_agent and num_agents > 0 else 1

    # Create figure with appropriate size
    fig_height = 6 * num_rows
    fig = plt.figure(figsize=(16, fig_height))

    # =========================================================================
    # ROW 1: Main overview (3D terrain, fused observation, fused belief, ground truth)
    # =========================================================================

    # Unpack observation bounds and create polygon coordinates
    [ox_min, ox_max], [oy_min, oy_max] = obs
    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

    # ---- Subplot 1: 3D Terrain with UAV Path ----
    ax1 = fig.add_subplot(num_rows, 4, 1, projection="3d")
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
    ax1.set_title("3D Terrain & UAV Paths")
    ax1.xaxis.grid(visible=True)

    # Handle multi-agent vs single-agent path visualization
    if (
        multi_agent
        and isinstance(uav_pos, list)
        and len(uav_pos) > 0
        and isinstance(uav_pos[0], list)
    ):
        # Multi-agent: uav_pos is list of lists
        all_z = []
        for agent_idx, agent_positions in enumerate(uav_pos):
            if len(agent_positions) == 0:
                continue

            uav_x, uav_y, uav_z = zip(
                *[
                    (uav.position[0], uav.position[1], uav.altitude)
                    for uav in agent_positions
                ]
            )
            all_z.extend(uav_z)

            # Use distinct color per agent
            agent_color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

            for i in range(len(uav_x) - 1):
                ax1.plot(
                    uav_x[i : i + 2],
                    uav_y[i : i + 2],
                    uav_z[i : i + 2],
                    color=agent_color,
                    linewidth=2,
                    alpha=0.9,
                )

            # Mark start position with a larger marker
            if len(uav_x) > 0:
                ax1.scatter(
                    [uav_x[0]],
                    [uav_y[0]],
                    [uav_z[0]],
                    color=agent_color,
                    s=50,
                    marker="o",
                    edgecolors="white",
                    linewidths=1,
                )
                # Mark current position with a different marker
                ax1.scatter(
                    [uav_x[-1]],
                    [uav_y[-1]],
                    [uav_z[-1]],
                    color=agent_color,
                    s=80,
                    marker="^",
                    edgecolors="white",
                    linewidths=1,
                    label=f"Agent {agent_idx}",
                )

        ax1.legend(loc="upper left", fontsize=8)
        z_max = max(35, max(all_z)) if all_z else 35
    else:
        # Single-agent: original behavior
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

    # ---- Subplot 2: 2D Last Observation (fused/first agent) ----
    ax2 = fig.add_subplot(num_rows, 4, 2)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    title_suffix = " (Fused)" if multi_agent else ""
    ax2.set_title(f"Last Observation z_t{title_suffix}")
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    cmap = colors.ListedColormap(["lemonchiffon", "darkgreen"])

    bounds = [-0.5, 0.5, 1.5]
    norm_binary = colors.BoundaryNorm(bounds, cmap.N)

    if submap is not None and submap.size > 0:
        ax2.imshow(
            submap,
            cmap=cmap,
            norm=norm_binary,
            extent=[ox_min, ox_max, oy_min, oy_max],
            origin="upper",
        )
    ax2.plot(o_x, o_y, color="red", lw=0.9)

    # ---- Subplot 3: Belief Map (fused) ----
    ax3 = fig.add_subplot(num_rows, 4, 3)
    ax3.set_xlabel("j-axis")
    ax3.set_ylabel("i-axis")
    ax3.set_title(f"Belief Map M{title_suffix}")

    belief_map = belief[:, :, 1] if belief.ndim == 3 else belief
    # Create a continuous colormap going from lemonchiffon (0) to dark green (1)
    colors_belief = ["lemonchiffon", "darkgreen"]
    green_yellow_cmap = colors.LinearSegmentedColormap.from_list(
        "GreenYellow", colors_belief
    )

    im3 = ax3.imshow(belief_map, cmap=green_yellow_cmap, origin="upper", vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # ---- Subplot 4: Ground Truth in Grid Indices ----
    ax4 = fig.add_subplot(num_rows, 4, 4)
    ax4.set_xlabel("j-axis")
    ax4.set_ylabel("i-axis")
    ax4.set_title("Ground Truth (i,j)")

    ax4.imshow(
        ground_truth,
        cmap=cmap,
        norm=norm_binary,
        origin="upper",
    )
    ax4.set_xlim(0, ground_truth.shape[1])
    ax4.set_ylim(ground_truth.shape[0], 0)

    # Draw footprint on ground truth
    if isinstance(fp, dict) and "ul" in fp:
        I, J = 0, 1
        o_i = [fp["ul"][I], fp["bl"][I], fp["br"][I], fp["ur"][I], fp["ul"][I]]
        o_j = [fp["ul"][J], fp["bl"][J], fp["br"][J], fp["ur"][J], fp["ul"][J]]
        ax4.plot(o_j, o_i, color="red", lw=0.9)

    # Region visualization moved to per-agent rows for multi-agent mode
    # Only show global regions in single-agent mode
    if region_metadata is not None and not multi_agent:
        _draw_regions_on_axis(ax4, region_metadata, selected_region_id, region_scores)

    # =========================================================================
    # ADDITIONAL ROWS: Per-agent observations and beliefs
    # =========================================================================
    if multi_agent and per_agent_data:
        for agent_idx, agent_data in enumerate(per_agent_data):
            row_offset = (agent_idx + 1) * 4  # Each row has 4 columns
            agent_id = agent_data.get("agent_id", agent_idx)
            agent_color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

            # ---- Agent Label (Column 1) ----
            ax_label = fig.add_subplot(num_rows, 4, row_offset + 1)
            ax_label.set_axis_off()
            ax_label.text(
                0.5,
                0.5,
                f"Agent {agent_id}",
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
                color=agent_color,
                transform=ax_label.transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor=agent_color,
                    linewidth=3,
                ),
            )

            # ---- Agent Observation (Column 2) ----
            ax_obs = fig.add_subplot(num_rows, 4, row_offset + 2)
            ax_obs.set_xlabel("X-axis")
            ax_obs.set_ylabel("Y-axis")
            ax_obs.set_title(
                f"Agent {agent_id} - Observation", color=agent_color, fontweight="bold"
            )
            ax_obs.set_xlim(x_range)
            ax_obs.set_ylim(y_range)

            agent_submap = agent_data.get("submap")
            agent_obs_range = agent_data.get("obs_range", obs)

            if agent_submap is not None and agent_submap.size > 0:
                [aox_min, aox_max], [aoy_min, aoy_max] = agent_obs_range
                ax_obs.imshow(
                    agent_submap,
                    cmap=cmap,
                    norm=norm_binary,
                    extent=[aox_min, aox_max, aoy_min, aoy_max],
                    origin="upper",
                )
                # Draw observation boundary
                ao_x = [aox_min, aox_max, aox_max, aox_min, aox_min]
                ao_y = [aoy_min, aoy_min, aoy_max, aoy_max, aoy_min]
                ax_obs.plot(ao_x, ao_y, color=agent_color, lw=2)
            else:
                ax_obs.text(
                    0.5,
                    0.5,
                    "No observation",
                    ha="center",
                    va="center",
                    transform=ax_obs.transAxes,
                    fontsize=12,
                    color="gray",
                )

            # ---- Agent Belief Map (Column 3) ----
            ax_belief = fig.add_subplot(num_rows, 4, row_offset + 3)
            ax_belief.set_xlabel("j-axis")
            ax_belief.set_ylabel("i-axis")
            ax_belief.set_title(
                f"Agent {agent_id} - Belief Map", color=agent_color, fontweight="bold"
            )

            agent_belief = agent_data.get("belief_map")
            if agent_belief is not None and agent_belief.size > 0:
                agent_belief_2d = (
                    agent_belief[:, :, 1] if agent_belief.ndim == 3 else agent_belief
                )
                im_belief = ax_belief.imshow(
                    agent_belief_2d,
                    cmap=green_yellow_cmap,
                    origin="upper",
                    vmin=0,
                    vmax=1,
                )
                plt.colorbar(im_belief, ax=ax_belief, fraction=0.046, pad=0.04)

                # Draw observation footprint on belief
                agent_fp = agent_data.get("fp_ij")
                if isinstance(agent_fp, dict) and "ul" in agent_fp:
                    I, J = 0, 1
                    fp_i = [
                        agent_fp["ul"][I],
                        agent_fp["bl"][I],
                        agent_fp["br"][I],
                        agent_fp["ur"][I],
                        agent_fp["ul"][I],
                    ]
                    fp_j = [
                        agent_fp["ul"][J],
                        agent_fp["bl"][J],
                        agent_fp["br"][J],
                        agent_fp["ur"][J],
                        agent_fp["ul"][J],
                    ]
                    ax_belief.plot(fp_j, fp_i, color=agent_color, lw=2, linestyle="--")
            else:
                ax_belief.text(
                    0.5,
                    0.5,
                    "No belief data",
                    ha="center",
                    va="center",
                    transform=ax_belief.transAxes,
                    fontsize=12,
                    color="gray",
                )

            # ---- Agent HLP Regions (Column 4) ----
            ax_hlp = fig.add_subplot(num_rows, 4, row_offset + 4)
            ax_hlp.set_xlabel("j-axis")
            ax_hlp.set_ylabel("i-axis")
            ax_hlp.set_title(
                f"Agent {agent_id} - HLP Regions", color=agent_color, fontweight="bold"
            )

            # Show ground truth as background
            ax_hlp.imshow(
                ground_truth,
                cmap=cmap,
                norm=norm_binary,
                origin="upper",
                alpha=0.5,
            )
            ax_hlp.set_xlim(0, ground_truth.shape[1])
            ax_hlp.set_ylim(ground_truth.shape[0], 0)

            # Draw HLP regions for this agent
            agent_region_metadata = agent_data.get("region_metadata")
            agent_selected_region = agent_data.get("selected_region_id")
            agent_region_scores = agent_data.get("region_scores")

            if agent_region_metadata is not None:
                _draw_regions_on_axis(
                    ax_hlp,
                    agent_region_metadata,
                    agent_selected_region,
                    agent_region_scores,
                )
            else:
                ax_hlp.text(
                    0.5,
                    0.5,
                    "No HLP regions",
                    ha="center",
                    va="center",
                    transform=ax_hlp.transAxes,
                    fontsize=12,
                    color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _draw_regions_on_axis(ax, region_metadata, selected_region_id, region_scores):
    """Helper function to draw region visualization on an axis."""
    # Get top 5 regions by score (if scores available)
    top_5_regions = set()
    if region_scores is not None and len(region_scores) > 0:
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        top_5_regions = {rid for rid, score in sorted_regions[:5]}

    for region_id, metadata in region_metadata.items():
        center = metadata["center"]
        bounds = metadata["bounds"]

        # Bounds are already in grid (i,j) coordinates
        (row_min, row_max), (col_min, col_max) = bounds

        # Rectangle dimensions in grid indices
        rect_width = col_max - col_min
        rect_height = row_max - row_min

        # Determine color based on selection and ranking
        if region_id == selected_region_id:
            edge_color = "red"
            line_width = 2.5
            alpha = 0.9
        elif region_id in top_5_regions:
            edge_color = "cyan"
            line_width = 2.0
            alpha = 0.75
        else:
            edge_color = "yellow"
            line_width = 1.5
            alpha = 0.6

        # Draw rectangle (note: ax uses j for x-axis, i for y-axis)
        rect = patches.Rectangle(
            (col_min, row_min),  # (j, i) coordinates
            rect_width,
            rect_height,
            linewidth=line_width,
            edgecolor=edge_color,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Draw center point (center is in (row, col) format)
        center_j = center[1]  # col
        center_i = center[0]  # row

        # Color based on rank
        if region_id == selected_region_id:
            marker_color = "red"
            marker_size = 10
        elif region_id in top_5_regions:
            marker_color = "cyan"
            marker_size = 8
        else:
            marker_color = "yellow"
            marker_size = 6

        ax.plot(
            center_j,
            center_i,
            "x",
            color=marker_color,
            markersize=marker_size,
            markeredgewidth=2,
            alpha=0.9,
        )

        # Add region ID label at center
        if region_id == selected_region_id:
            label_facecolor = "red"
            label_fontsize = 9
            label_weight = "bold"
        elif region_id in top_5_regions:
            label_facecolor = "blue"
            label_fontsize = 8
            label_weight = "bold"
        else:
            label_facecolor = "gray"
            label_fontsize = 7
            label_weight = "normal"

        ax.text(
            center_j,
            center_i + rect_height * 0.1,
            str(region_id),
            color="white",
            fontsize=label_fontsize,
            ha="center",
            va="center",
            weight=label_weight,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=label_facecolor,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
            ),
        )


def plot_metrics(
    save_path, entropy_list, mse_list, coverage_list, height_list, height_range=None
):
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
        ax4.axhline(
            y=min_h, color="gray", linestyle="--", alpha=0.7, label=f"Min: {min_h:.1f}"
        )
        ax4.axhline(
            y=max_h, color="gray", linestyle="--", alpha=0.7, label=f"Max: {max_h:.1f}"
        )
        ax4.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path.endswith(".png"):
        plt.savefig(save_path)
    else:
        plt.savefig(f"{save_path}/final.png")
    plt.close(fig)

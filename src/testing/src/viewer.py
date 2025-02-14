from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported


def plot_terrain(filename, belief, grid, uav_pos, gt, submap, obs):

    # Plot both the 3D and 2D maps in subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    for ax in axes:
        ax.set_axis_off()
    [
        [ox_min, ox_max],
        [oy_min, oy_max],
    ] = obs
    # ox_min = np.min(zx)
    # ox_max = np.max(zx)
    # oy_min = np.min(zy)
    # oy_max = np.max(zy)

    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

    # ---- Plot 1: uav position and ground truth 3D ----
    ax1 = fig.add_subplot(131, projection="3d")
    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    else:
        x_range = [0, grid.x]
        y_range = [0, grid.y]

    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim([0, 35])
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Altitude (m)")
    ax1.set_title("Truth Terrain and UAV position")
    ax1.xaxis.grid(visible=True)
    # uav
    x, y, z = zip(
        *[(uav.position[0], uav.position[1], uav.altitude) for uav in uav_pos]
    )
    ax1.plot(x, y, z, marker="o", color="r", linestyle="-")

    # Truth terrain map
    x = np.arange(x_range[0], x_range[1], grid.length)
    y = np.arange(y_range[0], y_range[1], grid.length)
    x, y = np.meshgrid(x, y, indexing="ij")

    ax1.plot_surface(
        x,
        y,
        np.zeros_like(x),
        facecolors=np.where(gt == 0, "green", "yellow"),
        alpha=0.6,
        edgecolor="none",
    )
    o_z = np.zeros_like(o_x) + 0.01  # Slightly above z=0

    # Plot the observation lines in 3D
    ax1.plot(o_y, o_x, o_z, color="red", lw=1)

    # ---- Plot 2: 2D last observation z_t ----
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("last observation z_t")
    # ax2.set_aspect("equal")
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)

    cmap = colors.ListedColormap(["green", "yellow"])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im1 = ax2.imshow(
        submap.T,
        cmap=cmap,
        norm=norm,
        extent=[oy_min, oy_max, ox_min, ox_max],
        origin="lower",
    )

    # ---- Plot 3: Belief sampled map M----
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("X Axis")
    ax3.set_ylabel("Y Axis")
    ax3.set_title("Belief sampled map M")

    if belief.ndim == 3:
        map = belief[:, :, 1].T
    else:
        map = belief.T

    im2 = ax3.imshow(
        map,
        cmap="Blues",
        extent=[y.min(), y.max(), x.min(), x.max()],
        origin="lower",
        # interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    plt.tight_layout()

    # Show the plots
    plt.savefig(filename)
    plt.close(fig)


def plot_metrics(dir, entropy_list, mse_list, coverage_list, height_list):

    assert len(entropy_list) == len(mse_list)
    assert len(coverage_list) == len(mse_list)
    assert len(coverage_list) == len(height_list)

    steps = range(len(entropy_list))

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))  # 3 rows, 1 column
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
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

    # Adjust layout to avoid overlap
    plt.tight_layout()
    # Show the plot
    if dir[-4:] == ".png":
        plt.savefig(dir)
    else:
        plt.savefig(dir + "/final.png")
    plt.close(fig)

    # plt.show()

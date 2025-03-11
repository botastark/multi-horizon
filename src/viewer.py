from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported

from matplotlib import cm
from matplotlib.colors import Normalize


def plot_terrain_3d_only(filename, grid, uav_pos, gt, obs):
    # Larger fonts for a paper
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig = plt.figure(figsize=(11, 14), dpi=300)
    # fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection="3d")

    # Unpack observation boundaries
    [ox_min, ox_max], [oy_min, oy_max] = obs
    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]
    o_z = np.zeros_like(o_x) + 0.01  # slightly above Z=0

    # Determine x/y ranges
    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    else:
        x_range = [0, grid.x]
        y_range = [0, grid.y]

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Axis labels, no title
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)
    ax.set_zlabel("h (m)", labelpad=4)

    # Adjust the camera angle & aspect ratio
    # ax.view_init(elev=45, azim=-60)  # nice angle from above
    # ax.set_box_aspect((1, 1, 0.3))  # flatten Z for a wide terrain

    # Prepare UAV path color gradient (viridis)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=1)
    colors_list = np.linspace(0, 1, len(uav_pos))

    # Plot UAV path
    uav_x, uav_y, uav_z = zip(
        *[(uav.position[0], uav.position[1], uav.altitude) for uav in uav_pos]
    )
    for i in range(len(uav_x) - 1):
        ax.plot(
            uav_x[i : i + 2],
            uav_y[i : i + 2],
            uav_z[i : i + 2],
            color=cmap(norm(colors_list[i])),
            linewidth=3,
        )

    # Adjust Z-limits if UAV altitude is higher than 35
    z_max = max(32.5, max(uav_z))
    ax.set_zlim([0, z_max])

    # Plot ground-truth terrain as a surface at Z=0
    step = grid.length
    x_vals = np.arange(x_range[0], x_range[1], step)
    y_vals = np.arange(y_range[0], y_range[1], step)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    # Example: binary terrain => silver (0), firebrick (1)
    facecolors = np.where(gt == 0, "silver", "firebrick")
    ax.plot_surface(
        X.T,
        -Y.T,
        np.zeros_like(X.T),
        facecolors=facecolors,
        alpha=0.6,  # slightly more opaque
        edgecolor="none",  # no wireframe
        # antialiased=True,
    )

    # Observation boundary in red
    ax.plot(o_x, o_y, o_z, color="red", lw=3)

    # Make sure labels/ticks aren’t cut off
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_terrain_2d(filename, grid, uav_pos, gt, obs):
    # Larger fonts for a paper
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    # fig = plt.figure(figsize=(11, 14), dpi=300)
    # fig = plt.figure(dpi=200)
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

    # Axis labels, no title
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)

    # Plot ground-truth terrain as a surface at Z=0
    step = grid.length
    x_vals = np.arange(x_range[0], x_range[1], step)
    y_vals = np.arange(y_range[0], y_range[1], step)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    cmap = colors.ListedColormap(["lemonchiffon", "darkgreen"])

    ax.imshow(
        gt.T,
        cmap=cmap,
        origin="lower",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
    )
    ax.set_xticks(np.linspace(x_range[0], x_range[1], 5))
    ax.set_yticks(np.linspace(y_range[0], y_range[1], 5))
    ax.set_xticklabels(
        [f"{tick:.1f}" for tick in np.linspace(x_range[0], x_range[1], 5)]
    )
    ax.set_yticklabels(
        [f"{tick:.1f}" for tick in np.linspace(y_range[0], y_range[1], 5)]
    )

    # Make sure labels/ticks aren’t cut off
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_terrain(filename, belief, grid, uav_pos, gt, submap, obs, fp):
    # Plot both the 3D and 2D maps in subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    for ax in axes:
        ax.set_axis_off()
    [
        [ox_min, ox_max],
        [oy_min, oy_max],
    ] = obs

    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

    # ---- Plot 1: uav position and ground truth 3D ----
    ax1 = fig.add_subplot(141, projection="3d")
    if grid.center:
        x_range = [-grid.x / 2, grid.x / 2]
        y_range = [-grid.y / 2, grid.y / 2]
    else:
        x_range = [0, grid.x]
        y_range = [0, grid.y]

    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Altitude (m)")
    ax1.set_title("Truth Terrain and UAV position")
    ax1.xaxis.grid(visible=True)
    # uav
    colors_list = np.linspace(0, 1, len(uav_pos))

    # pyplot
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=1)

    uav_x, uav_y, uav_z = zip(
        *[(uav.position[0], uav.position[1], uav.altitude) for uav in uav_pos]
    )
    # ax1.plot(uav_x, uav_y, uav_z, marker="o", color="r", linestyle="-")
    # Plot the UAV path with color gradient
    for i in range(len(uav_x) - 1):
        ax1.plot(
            uav_x[i : i + 2],
            uav_y[i : i + 2],
            uav_z[i : i + 2],
            color=cmap(norm(colors_list[i])),
            linewidth=2,
        )
    z_max = 35
    if max(uav_z) > 35:
        z_max = max(uav_z)
    ax1.set_zlim([0, z_max])

    # Truth terrain map
    x = np.arange(x_range[0], x_range[1], grid.length)
    y = np.arange(y_range[0], y_range[1], grid.length)
    x, y = np.meshgrid(x, y, indexing="ij")
    ax1.plot_surface(
        x.T,
        -y.T,
        np.zeros_like(x.T),
        facecolors=np.where(gt == 0, "green", "yellow"),
        alpha=0.6,
        edgecolor="none",
    )
    o_z = np.zeros_like(o_x) + 0.01  # Slightly above z=0

    # Plot the observation lines in 3D
    ax1.plot(o_x, o_y, o_z, color="red", lw=1)

    # ---- Plot 2: 2D last observation z_t ----
    ax2 = fig.add_subplot(142)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("last observation z_t")
    # ax2.set_aspect("equal")
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)

    cmap = colors.ListedColormap(["green", "yellow"])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im0 = ax2.imshow(
        gt,
        cmap=cmap,
        norm=norm,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="upper",
    )
    im1 = ax2.imshow(
        submap,
        cmap=cmap,
        norm=norm,
        extent=[ox_min, ox_max, oy_min, oy_max],
        origin="upper",
    )
    ax2.plot(o_x, o_y, o_z, color="red", lw=3)

    # ---- Plot 3: Belief sampled map M----
    ax3 = fig.add_subplot(143)
    ax3.set_xlabel("j-axis")
    ax3.set_ylabel("i-axis")
    ax3.set_title("Belief sampled map M")

    if belief.ndim == 3:
        map = belief[:, :, 1]
    else:
        map = belief

    im2 = ax3.imshow(
        map,
        cmap="Blues",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    ax4 = fig.add_subplot(144)
    ax4.set_xlabel("j-axis")
    ax4.set_ylabel("i-axis")
    ax4.set_title("GT in ij")

    cmap = colors.ListedColormap(["green", "yellow"])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im4 = ax4.imshow(
        gt,
        cmap=cmap,
        norm=norm,
        origin="upper",
    )
    I, J = 0, 1
    o_i = [fp["ul"][I], fp["bl"][I], fp["br"][I], fp["ur"][I], fp["ul"][I]]
    o_j = [fp["ul"][J], fp["bl"][J], fp["br"][J], fp["ur"][J], fp["ul"][J]]
    ax4.plot(o_j, o_i, np.zeros_like(o_j), color="red", lw=2)
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

from typing import List
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported


def plot_terrain(filename, belief, grid, uav_pos, gt, submap, zx, zy):

    # Plot both the 3D and 2D maps in subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    for ax in axes:
        ax.set_axis_off()

    ox_min = np.min(zx)
    ox_max = np.max(zx)
    oy_min = np.min(zy)
    oy_max = np.max(zy)

    o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
    o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

    # ---- Plot 1: uav position and ground truth 3D ----

    ax1 = fig.add_subplot(131, projection="3d")

    ax1.set_xlim([0, grid.x])
    ax1.set_ylim([0, grid.y])
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
    x = np.arange(0, grid.x, grid.length)
    y = np.arange(0, grid.y, grid.length)
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

    # Plot the line in 3D
    ax1.plot(o_x, o_y, o_z, color="red", lw=1)

    # ---- Plot 2: 2D last observation z_t ----
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("last observation z_t")
    # ax2.set_aspect("equal")
    ax2.set_xlim([0, grid.x])
    ax2.set_ylim([0, grid.y])

    cmap = colors.ListedColormap(["green", "yellow"])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im1 = ax2.imshow(
        submap.T,
        cmap=cmap,
        norm=norm,
        extent=[ox_min, ox_max, oy_min, oy_max],
        origin="lower",
    )

    # ---- Plot 3: Belief sampled map M----
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("X Axis")
    ax3.set_ylabel("Y Axis")
    ax3.set_title("Belief sampled map M")
    # ax3.set_xlim([0, self.x_range[1]])
    # ax3.set_ylim([0, self.y_range[1]])
    im2 = ax3.imshow(
        belief[:, :, 1].T,
        cmap="Blues",
        # norm=norm,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        # interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    plt.tight_layout()

    # Show the plots
    plt.savefig(filename)
    plt.close(fig)

    # def plot_prob(self, filename):

    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 50))

    #     # # Example 2D grid of probabilities (replace with your actual probability values)
    #     probability_map_0 = self.probability[0, :, :]  # First probability map
    #     # print(self.probability[0].shape)
    #     # probability_map_1 = self.probability[1, :, :]  # Second probability map

    #     # # Plot the first probability map in ax[0]
    #     cax0 = ax.imshow(
    #         probability_map_0.T,
    #         cmap="Blues",
    #         interpolation="nearest",
    #         origin="lower",
    #         vmin=0,
    #         vmax=1,
    #         extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
    #     )

    #     # # Add text annotations for the first probability map
    #     for (i, j), prob in np.ndenumerate(probability_map_0.T):
    #         ax.text(
    #             self.y[i, j] + self.grid.length / 4,
    #             self.x[i, j] + self.grid.length / 4,
    #             f"{prob:.2f}",
    #             ha="center",
    #             va="center",
    #             color="black",
    #             fontsize=10,
    #         )

    #     ax.set_title("Probability Map 0")
    #     ax.set_xlabel("X-axis")
    #     ax.set_ylabel("Y-axis")

    #     # # Adjust the layout to make sure everything fits well
    #     plt.tight_layout()

    #     # # Save the figure to a file
    #     plt.savefig(filename)
    #     plt.close(fig)

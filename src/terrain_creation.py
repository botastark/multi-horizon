from typing import List
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported


class terrain:
    def __init__(self, grid):
        x = np.arange(0, grid.x, grid.length)
        y = np.arange(0, grid.y, grid.length)
        self.x, self.y = np.meshgrid(x, y, indexing="ij")
        self.map = np.ones(self.x.shape) * 0.5
        self.probability = np.array(
            [np.ones(self.x.shape) * 0.5, np.ones(self.x.shape) * 0.5]
        )
        self.grid = grid
        self.y_range = (0, grid.y)
        self.x_range = (0, grid.x)
        self.z_range = (0, 1)
        self._current = 0  # internal index for iteration

    def __len__(self):
        return self.map.shape

    def get_grid(self):
        return self.x, self.y

    def get_map(self):
        return self.map

    def get_ranges(self):
        return self.x_range, self.y_range, self.z_range

    def pos2grid(self, pos):
        # Ensure the position is within the defined x_range and y_range
        if not (self.x_range[0] <= pos[0] <= self.x_range[1]) or not (
            self.y_range[0] <= pos[1] <= self.y_range[1]
        ):
            raise ValueError(
                "Position is outside the valid range defined by x_range and y_range."
            )

        # Convert position in meters into grid coordinates
        grid_x = int((pos[0] - self.x_range[0]) / self.grid.length)
        grid_y = int((pos[1] - self.y_range[0]) / self.grid.length)

        # Ensure the grid coordinates are within the bounds of the grid shape

        grid_x = min(max(grid_x, 0), self.map.shape[-2] - 1)
        grid_y = min(max(grid_y, 0), self.map.shape[-1] - 1)

        return grid_x, grid_y

    def grid2pos(self, coords):
        # Convert grid coordinates back to positions in meters using x_range and y_range
        pos_x = self.x_range[0] + coords[0] * self.grid.length
        pos_y = self.y_range[0] + coords[1] * self.grid.length
        return pos_x, pos_y

    def set_map(self, z, x=[], y=[]):
        self.map = np.array([])
        if self.x.shape == z.shape:
            self.map = z
        else:
            if len(x) != 0 and len(y) != 0:
                self.x = x
                self.y = y
                self.map = z

                # Check if x and y shapes match the shape of z
                if self.x.shape != self.y.shape or self.x.shape != z.shape:
                    raise ValueError("Shapes of x, y, and z must match.")

                    # Update x_range and y_range based on the new x and y meshgrid
                self.x_range = (np.min(self.x), np.max(self.x))
                self.y_range = (np.min(self.y), np.max(self.y))

                # if map shape changes, prob must change as well to 0.5 with shape [2, map.shape[0], map.shape[1]]
                self.probability = np.array(
                    [0.5 * np.ones_like(self.map), 0.5 * np.ones_like(self.map)]
                )
            else:
                raise TypeError("Grid and Map sizes don't match and no grid is passed")

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        num_rows, num_cols = self.map.shape[-2], self.map.shape[-1]
        if self._current >= num_rows * num_cols:
            raise StopIteration
        row = self._current // num_cols
        col = self._current % num_cols

        self._current += 1
        return (row, col)

    def copy(self):
        return copy.deepcopy(self)

    def plot_map(self, filename, fit=True):
        # Plot both the 3D and 2D maps in subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        vmin = 0
        vmax = 1

        # Plot 3D map
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Elevation")
        ax1.set_title("3D Terrain Map")
        if fit:
            surf = ax1.plot_surface(
                self.x, self.y, self.map, cmap="viridis", alpha=0.8, vmin=-1, vmax=1.5
            )
            ax1.set_xlim([0, self.x_range[1]])
            ax1.set_ylim([0, self.y_range[1]])
            # ax1.set_zlim([0, 1000])
        else:
            surf = ax1.plot_surface(self.x, self.y, self.map, cmap="viridis", alpha=0.8)

        # Plot 2D map
        ax2 = axes[1]
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_title("2D Terrain Map")
        levels = np.linspace(vmin, vmax, 41)

        if fit:
            contour = ax2.contourf(
                self.x,
                self.y,
                self.map,
                cmap="viridis",
                levels=levels,
                vmin=-1,
                vmax=1.5,
            )
            ax2.set_xlim([0, self.x_range[1]])
            ax2.set_ylim([0, self.y_range[1]])
            # ax2.set_zlim([0, 1000])
        else:
            contour = ax2.contourf(self.x, self.y, self.map, cmap="viridis", levels=40)

        cbar2 = fig.colorbar(contour, ax=ax2, label="Elevation")
        plt.tight_layout()
        # Show the plots
        plt.savefig(filename)
        plt.close(fig)

    def plot_terrain(self, filename, uav_pos, gt, obs_z):

        # Plot both the 3D and 2D maps in subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
        for ax in axes:
            ax.set_axis_off()

        ox_min = np.min(obs_z.x)
        ox_max = np.max(obs_z.x)
        oy_min = np.min(obs_z.y)
        oy_max = np.max(obs_z.y)

        o_x = [ox_min, ox_max, ox_max, ox_min, ox_min]
        o_y = [oy_min, oy_min, oy_max, oy_max, oy_min]

        # ---- Plot 1: uav position and ground truth 3D ----

        ax1 = fig.add_subplot(131, projection="3d")

        ax1.set_xlim([self.x_range[0], self.x_range[1]])
        ax1.set_ylim([self.y_range[0], self.y_range[1]])
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
        ax1.plot_surface(
            self.x,
            self.y,
            np.zeros_like(self.x),
            facecolors=np.where(gt.map == 0, "green", "yellow"),
            alpha=0.6,
            edgecolor="none",
        )
        o_z = np.zeros_like(o_x) + 0.01  # Slightly above z=0

        # Plot the line in 3D
        ax1.plot(o_x, o_y, o_z, color="red", lw=2)

        # ---- Plot 2: 2D last observation z_t ----
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_title("last observation z_t")
        # ax2.set_aspect("equal")
        ax2.set_xlim([0, self.x_range[1]])
        ax2.set_ylim([0, self.y_range[1]])

        cmap = colors.ListedColormap(["green", "yellow"])
        bounds = [-0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im1 = ax2.imshow(
            obs_z.map.T,
            cmap=cmap,
            norm=norm,
            extent=[obs_z.x.min(), obs_z.x.max(), obs_z.y.min(), obs_z.y.max()],
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
            self.map.T,
            cmap=cmap,
            norm=norm,
            extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
            origin="lower",
        )

        plt.tight_layout()

        # Show the plots
        plt.savefig(filename)
        plt.close(fig)

    def plot_prob(self, filename):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 50))

        # # Example 2D grid of probabilities (replace with your actual probability values)
        probability_map_0 = self.probability[0, :, :]  # First probability map
        # print(self.probability[0].shape)
        # probability_map_1 = self.probability[1, :, :]  # Second probability map

        # # Plot the first probability map in ax[0]
        cax0 = ax.imshow(
            probability_map_0.T,
            cmap="Blues",
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=1,
            extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
        )

        # # Add text annotations for the first probability map
        for (i, j), prob in np.ndenumerate(probability_map_0.T):
            ax.text(
                self.y[i, j] + self.grid.length / 4,
                self.x[i, j] + self.grid.length / 4,
                f"{prob:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        ax.set_title("Probability Map 0")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # # Adjust the layout to make sure everything fits well
        plt.tight_layout()

        # # Save the figure to a file
        plt.savefig(filename)
        plt.close(fig)


# import gstools as gs


# def gen_fast(map, radius):
#     xx, yy = map.get_grid()
#     x = xx[:, 0]
#     y = yy[0, :]

#     seed = 20170519
#     if radius == 0.0:
#         radius += 0.001
#     model = gs.Gaussian(
#         dim=2,
#         len_scale=radius,
#     )

#     srf = gs.SRF(model, seed=seed)

#     srf.structured([x, y])
#     z = srf.transform("binary")
#     z = np.where(z == -1.0, 0.0, z)

#     return z
def gaussian_random_field(cluster_radius, n_cell):

    """Generate 2D gaussian random field:
    https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/"""

    def _fft_indices(n) -> List:
        a = list(range(0, int(np.floor(n / 2)) + 1))
        b = reversed(range(1, int(np.floor(n / 2))))
        b = [-i for i in b]
        return a + b

    def _pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        val = np.sqrt(np.sqrt(kx ** 2 + ky ** 2) ** (-cluster_radius))
        return val

    # memoization of amplitude (amplitude depends on n_cell, fft_indices, cluster_radius)
    # there is a one-to-one mapping between cluster_radius and amplitude
    # for each cluster_radius, what makes the maps different is just the noise
    # if cluster_radius not in self.cluster_radius_to_amplitude:
    cluster_radius_to_amplitude = {}
    map_rng = np.random.default_rng(123)
    amplitude = np.zeros((n_cell, n_cell))
    fft_indices = _fft_indices(n_cell)

    for i, kx in enumerate(fft_indices):
        for j, ky in enumerate(fft_indices):
            amplitude[i, j] = _pk2(kx, ky)


        cluster_radius_to_amplitude[cluster_radius] = np.copy(amplitude)

    noise = np.fft.fft2(map_rng.normal(size=(n_cell, n_cell)))
    random_field = np.fft.ifft2(noise * cluster_radius_to_amplitude[cluster_radius]).real
    normalized_random_field = (random_field - np.min(random_field)) / (
            np.max(random_field) - np.min(random_field)
    )

    # Make field binary
    normalized_random_field[normalized_random_field >= 0.5] = 1
    normalized_random_field[normalized_random_field < 0.5] = 0

    return normalized_random_field.astype(np.uint8)

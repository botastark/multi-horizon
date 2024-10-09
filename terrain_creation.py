from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported
from scipy.spatial.distance import cdist


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
        # plt.show(block=False)
        plt.savefig(filename)

    def plot_terrain(self, filename, uav_pos, gt, obs_z, fit=True):

        # Plot both the 3D and 2D maps in subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

        # ---- Plot 1: uav position and ground truth 3D ----

        ax1 = fig.add_subplot(131, projection="3d")
        # Remove axes for the 3D plot

        ax1.set_xlim([0, self.x_range[1]])
        ax1.set_ylim([0, self.y_range[1]])
        ax1.set_zlim([0, 20])
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Altitude (m)")
        ax1.set_title("Truth Terrain and UAV position")
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

        # ---- Plot 2: 2D last observation z_t ----
        ax2 = fig.add_subplot(132, frameon=False)
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_title("last observation z_t")
        ax2.set_aspect("equal")
        ax2.set_xlim([0, self.x_range[1]])
        ax2.set_ylim([0, self.y_range[1]])

        cmap = colors.ListedColormap(["green", "yellow"])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im1 = ax2.imshow(
            obs_z.map.T,
            cmap=cmap,
            norm=norm,
            extent=[obs_z.x.min(), obs_z.x.max(), obs_z.y.min(), obs_z.y.max()],
            origin="lower",
            aspect="auto",
        )

        # ---- Plot 3: Belief sampled map M----
        ax3 = fig.add_subplot(133, frameon=False)
        ax3.set_xlabel("X Axis")
        ax3.set_ylabel("Y Axis")
        ax3.set_title("Belief sampled map M")
        ax3.set_xlim([0, self.x_range[1]])
        ax3.set_ylim([0, self.y_range[1]])

        im2 = ax3.imshow(self.map.T, cmap=cmap, norm=norm)

        # Add a color bar for the 2D plot L
        fig.colorbar(im2, ax=ax3, boundaries=bounds, ticks=[0, 1])

        for ax in [axes[0]]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        plt.tight_layout()

        # Show the plots
        plt.savefig(filename)


def generate_n_peaks(n_peaks, map):
    x, y = map.get_grid()
    x_range, y_range, z_range = map.get_ranges()

    z_combined = np.zeros_like(x)

    # Loop through each peak and generate elevations
    for _ in range(n_peaks):
        x_center = np.random.uniform(x_range[0], x_range[1])
        y_center = np.random.uniform(y_range[0], y_range[1])

        # Random amplitude (within the z range)
        amplitude = np.random.uniform(z_range[0], z_range[1])

        # Random spreads (standard deviations)
        sigma_x = np.random.uniform(
            3, 10
        )  # Control the width of the Gaussian in x-direction
        sigma_y = np.random.uniform(
            3, 10
        )  # Control the width of the Gaussian in y-direction

        z_combined += amplitude * np.exp(
            -(
                ((x - x_center) ** 2) / (2 * sigma_x**2)
                + ((y - y_center) ** 2) / (2 * sigma_y**2)
            )
        )
    z_combined = (z_combined - np.min(z_combined)) / (
        np.max(z_combined) - np.min(z_combined)
    )

    return z_combined


from scipy.spatial.distance import cdist
from scipy.linalg import cholesky


def generate_correlated_gaussian_field(map, r, scale=10.0):
    """
    Generate a correlated Gaussian random field terrain parametrized by a cluster radius r.

    Parameters:
    - grid_size: Tuple (m, n) defining the size of the grid.
    - r: Correlation cluster radius that controls the strength of spatial correlation.
    - scale: Scaling factor for the Gaussian field (default is 1.0).

    Returns:
    - terrain: A (m, n) matrix representing the correlated Gaussian random field.
    """
    # Create grid coordinates

    xx, yy = map.get_grid()
    m, n = xx.shape
    # Flatten the grid points into pairs of (x, y) coordinates
    xy_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
    if r == 0:
        # If r = 0, generate an independent random Gaussian field (no spatial correlation)
        terrain = np.random.normal(0, 1, (m, n))
    else:

        # Compute the pairwise Euclidean distance matrix for all grid points
        distance_matrix = cdist(xy_grid, xy_grid)

        # Define the covariance matrix using an exponential decay function based on distance and radius r
        # Covariance decreases exponentially with distance, with r controlling the decay rate
        covariance_matrix = np.exp(-distance_matrix / r)

        # Perform Cholesky decomposition for the covariance matrix (to ensure it is positive semi-definite)
        L = cholesky(covariance_matrix, lower=True)

        # Generate independent Gaussian random values
        z = np.random.normal(0, 1, xy_grid.shape[0])

        # Apply Cholesky factorization to introduce spatial correlation
        correlated_field = L @ z

        # Reshape the correlated field back into the original (m, n) grid shape
        terrain = correlated_field.reshape(m, n)

        # Scale the field by the provided scale factor
        terrain = scale * terrain

        # Normalize terrain values between 0 and 1
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
    terrain = (terrain > 0.5).astype(int)

    return terrain

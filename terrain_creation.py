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
        grid_x = min(max(grid_x, 0), self.map.shape[0] - 1)
        grid_y = min(max(grid_y, 0), self.map.shape[1] - 1)

        return grid_x, grid_y

    def grid2pos(self, coords):
        # Convert grid coordinates back to positions in meters using x_range and y_range
        pos_x = self.x_range[0] + coords[0] * self.grid.length
        pos_y = self.y_range[0] + coords[1] * self.grid.length
        return pos_x, pos_y

    def set_map(self, z, x=[], y=[]):
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

            else:
                raise TypeError("Grid and Map sizes don't match and no grid is passed")

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        num_rows, num_cols = self.map.shape
        if self._current >= num_rows * num_cols:
            raise StopIteration
        row = self._current // num_cols
        col = self._current % num_cols

        self._current += 1
        return (row, col)

    def copy(self):
        return copy.deepcopy(self)

    def plot_map(self, fit=True):
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
        plt.show()


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

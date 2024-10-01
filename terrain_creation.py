import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is imported

# class grid_info:
#   x = 30
#   y = 30
#   length = 0.75
#   shape = (int(x/length), int(y/length))
# grid = grid_info


class terrain:
    def __init__(self, grid):
      x = np.arange(0, grid.x, grid.length)
      y = np.arange(0, grid.y, grid.length)
      self.x, self.y = np.meshgrid(x, y,indexing='ij')
      self.map = np.ones(self.x.shape)*0.5
      self.y_range = (0, grid.y)
      self.x_range = (0, grid.x)
      self.z_range = (0, 1)

    def __len__(self):
      return self.map.shape

    def get_grid(self):
      return self.x, self.y

    def get_map(self):
      return self.map
    
    def get_ranges(self):
      return self.x_range, self.y_range, self.z_range


    def set_map(self, z, x=[], y=[]):
      if self.x.shape==z.shape:
        self.map = z
      else:
        if (len(x)!=0 and len(y)!=0):
          self.x=x
          self.y=y
          self.map=z
        else:
          raise TypeError("Grid and Map sizes don't match and no grid is passed")

    def plot_map(self, fit = True):
      # Plot both the 3D and 2D maps in subplots
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
      vmin=0
      vmax=1

      # Plot 3D map
      ax1 = fig.add_subplot(121, projection='3d')
      ax1.set_xlabel('X (m)')
      ax1.set_ylabel('Y (m)')
      ax1.set_zlabel('Elevation')
      ax1.set_title('3D Terrain Map')
      if fit:
        surf = ax1.plot_surface(self.x, self.y, self.map, cmap='viridis', alpha=0.8, vmin=-1, vmax=1.5)
        ax1.set_xlim([0, self.x_range[1]])
        ax1.set_ylim([0, self.y_range[1]])
        # ax1.set_zlim([0, 1000])
      else:
        surf = ax1.plot_surface(self.x, self.y, self.map, cmap='viridis', alpha=0.8)


      # Plot 2D map
      ax2 = axes[1]
      ax2.set_xlabel('X-axis')
      ax2.set_ylabel('Y-axis')
      ax2.set_title('2D Terrain Map')
      levels = np.linspace(vmin, vmax, 41)

      if fit:
        contour = ax2.contourf(self.x, self.y, self.map, cmap='viridis', levels=levels, vmin = -1, vmax=1.5)
        ax2.set_xlim([0, self.x_range[1]])
        ax2.set_ylim([0, self.y_range[1]])
        # ax2.set_zlim([0, 1000])
      else:
        contour = ax2.contourf(self.x, self.y, self.map, cmap='viridis', levels=40)

      cbar2 = fig.colorbar(contour, ax=ax2, label='Elevation')
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
        sigma_x = np.random.uniform(3, 10)  # Control the width of the Gaussian in x-direction
        sigma_y = np.random.uniform(3, 10)  # Control the width of the Gaussian in y-direction
    
        z_combined += amplitude * np.exp(-(((x - x_center)**2) / (2 * sigma_x**2) + ((y - y_center)**2) / (2 * sigma_y**2)))
    z_combined = (z_combined - np.min(z_combined)) / (np.max(z_combined) - np.min(z_combined))

    return z_combined
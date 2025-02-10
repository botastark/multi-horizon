# dataset
from PIL import Image
from osgeo import gdal
import numpy as np


import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
from torchvision import transforms


class WheatOthomapDataset(Dataset):
    def __init__(self, ortomap_path, annotation_path, tile_ortomappixel_path):
        dataset = gdal.Open(ortomap_path)
        band1 = dataset.GetRasterBand(1)  # Red channel
        band2 = dataset.GetRasterBand(2)  # Green channel
        band3 = dataset.GetRasterBand(3)  # Blue channel

        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        self.img = np.dstack((b1, b2, b3))
        self.tile_pixel_loc = self._parse_tile_file(tile_ortomappixel_path)
        self.labels = self._read_annotations_to_matrix(annotation_path)
        self.tiles = [(row, col) for row in range(3, 113) for col in range(13, 73)]

        self.transform = transforms.Compose(
            [
                transforms.Resize((180, 180)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _parse_tile_file(self, file_path):
        """
        Parse a text file containing tile information and store it in an np.ndarray format.
        np.ndarray: A matrix where each element M[row][col] = [[X coord, X coord+length], [Y coord, Y coord+height]].
        """
        tile_data = {}
        max_row, max_col = 0, 0

        with open(file_path, "r") as file:
            for line in file:
                # Parse the line to extract row, col, and tile data
                parts = line.strip().split(":")
                tile_id = parts[0].strip()
                coords = list(map(int, parts[1].strip().split(",")))

                # Extract row and col from the tile ID
                row_col = tile_id.replace("tile", "").split("_")
                row = int(row_col[0])
                col = int(row_col[1])

                # Update max row and col values
                max_row = max(max_row, row)
                max_col = max(max_col, col)

                # Store in the dictionary
                tile_data[(row, col)] = coords

        # Create an empty matrix of appropriate size
        matrix = np.empty((max_row + 1, max_col + 1), dtype=object)

        for (row, col), coords in tile_data.items():
            y, x, length, height = coords
            matrix[row][col] = [[x, x + length], [y, y + height]]

        return matrix

    def _read_annotations_to_matrix(self, file_path):
        try:
            matrix = np.loadtxt(file_path, dtype=int)  # Adjust dtype if necessary
            matrix[matrix == 10] = 1  # Adjust the value to match the label
            matrix[matrix == 2] = 1
            assert matrix.all() in [
                0,
                1,
            ], "Invalid label values found in the annotation file"
            return matrix
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")

    def _index_to_tile(self, index):
        return self.tiles[index]

    # Function to convert a tile (row, col) to an index
    def _tile_to_index(self, tile):
        return self.tiles.index(tile)

    def _get_image_range(self, tile):
        if isinstance(tile, tuple):
            r, c = tile
        else:
            r, c = tile[0], tile[1]
        tile_coords = self.tile_pixel_loc[r][c]
        x_range = slice(tile_coords[0][0], tile_coords[0][1])
        y_range = slice(tile_coords[1][0], tile_coords[1][1])
        return x_range, y_range

    def _get_tile_img(self, tile):
        x_range, y_range = self._get_image_range(tile)
        cropped_img = self.img[x_range, y_range, :]
        return Image.fromarray(cropped_img)

    def __getitem__(self, index):
        tile = self._index_to_tile(index)
        pil_img = self._get_tile_img(tile)
        label = self.labels[tile[0], tile[1]]
        data = self.transform(pil_img)
        target = torch.zeros(2)
        target[label] = 1
        return data, target

    def __len__(self):
        return len(self.tiles)

    def get_tile_info(self, tile):
        return self._get_tile_img(tile), self.labels[tile[0], tile[1]]


# train = WheatOthomapDataset(dataset_path, annotation_path, tile_ortomappixel_path)
# train_loader = torch.utils.data.DataLoader(
#     train, batch_size=32, shuffle=True, num_workers=16, pin_memory=True
# )

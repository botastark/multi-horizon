import os
import csv
from matplotlib import pyplot as plt
import numpy as np
from classifier import predict, predict_batch
from utilities import get_image_properties, img_NED, gps_ned
import pickle
from proj import camera


class TileOperations:
    def __init__(self, tiles_dir, gps_csv, row_imgs_dir):
        """
        Initialize TileOperations with paths and load necessary data.

        Args:
            tiles_dir (str): Directory containing tile images.
            gps_csv (str): Path to the CSV file with tile GPS data.
            row_imgs_dir (str): Directory containing row images for NED calculations.
        """
        self.tiles_dir = tiles_dir
        self.row_imgs_dir = row_imgs_dir
        self.tile_to_img_dict = self._init_tiles()
        self.gps_tile_dict = self._collect_tile_gps(gps_csv)
        ref_img_path = "/media/bota/BOTA/wheat/APPEZZAMENTO_PICCOLO/DJI_20240607121133_0006_D_point0.JPG"
        ref_point_info = get_image_properties(ref_img_path)

        self.L2 = camera(ref_point_info)

    def parse_tiles(self, tile_list):
        """Parse tile filenames to extract tile row and column."""
        tiles = []
        for tile in tile_list:
            _, row_col = tile.split("tile")
            row_col, _ = row_col.split("_crop")
            row, col = map(int, row_col.split("_"))
            tiles.append((row, col))
        return tiles

    def _init_tiles(self):
        """Initialize tile-to-image dictionary from the tiles directory."""
        folder = os.listdir(self.tiles_dir)
        tile_to_img_dict = {}
        for file_path in folder:
            tile_loc = self.parse_tiles([file_path])[0]
            if tile_loc in tile_to_img_dict:
                tile_to_img_dict[tile_loc].append(file_path)
            else:
                tile_to_img_dict[tile_loc] = [file_path]
        return tile_to_img_dict

    def groundtruth_tiles(self, grid_info, cache_dir="cache"):
        """
        Predict (TBU soon) from tiles and generate a 2D binary map as a numpy array.

        Parameters:
            grid_info (object): An object with `x` and `y` attributes for grid dimensions.
            cache_dir (str): Directory for caching the generated map.

        Returns:
            numpy.ndarray: 2D binary map.
        """
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Generate cache filename
        cache_file = os.path.join(
            cache_dir, f"field_tiles_{grid_info.x}x{grid_info.y}.pkl"
        )

        # Try loading from cache
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                print(f"Loading cached field from {cache_file}")
                return pickle.load(f)

        # Generate ground truth map
        ground_truth_map = self.observed_submap((0, grid_info.x), (0, grid_info.y))

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(ground_truth_map, f)
        print(f"Field generated and saved to {cache_file}")

        return ground_truth_map

    def get_tile_img_path(self, tile_loc):
        """Get the image paths associated with a specific tile."""
        return self.tile_to_img_dict[tile_loc]

    def _collect_tile_gps(self, csv_file):
        """Collect GPS data for each tile from the CSV file."""
        tile_dict = {}
        with open(csv_file, mode="r") as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            tiles = rows[0]
            gps_data = rows[1]

            for tile, gps_ in zip(tiles, gps_data):
                _, row_col = tile.split("tile")
                row, col = map(int, row_col.split("_"))
                tile = (row, col)
                gps = eval(gps_)
                gps_center = [(gps[3] + gps[2]) / 2, (gps[1] + gps[0]) / 2]
                tile_dict[tile] = gps_center
        return tile_dict

    # def observed_submap(self, xrange, yrange):
    #     """Generate an observed submap for the given range."""
    #     submap = np.zeros((xrange[1] - xrange[0], yrange[1] - yrange[0]))
    #     for c in range(xrange[0], xrange[1]):
    #         for r in range(yrange[0], yrange[1]):
    #             tile_loc = (r + 3, c + 13)  # Offset for 100% coverage rect
    #             tile_img_path, _ = self.find_closest_image(tile_loc)
    #             img_path = os.path.join(self.tiles_dir, tile_img_path)
    #             pred = predict(img_path)[0]
    #             if pred == 2:
    #                 pred = 1
    #             submap[c - xrange[0], r - yrange[0]] = pred
    #         return submap

    # Modify observed_submap to use batch prediction
    def observed_submap(self, xrange, yrange):
        """Generate an observed submap for the given range, with batch prediction."""
        submap = np.zeros((xrange[1] - xrange[0], yrange[1] - yrange[0]))

        # Collect all tile image paths in a batch
        img_paths_batch = []
        tiles_to_predict = []

        for c in range(xrange[0], xrange[1]):
            for r in range(yrange[0], yrange[1]):
                tile_loc = (r + 3, c + 13)  # Offset for 100% coverage rect
                tile_img_path, _ = self.find_closest_image(tile_loc)
                img_path = os.path.join(self.tiles_dir, tile_img_path)
                img_paths_batch.append(img_path)
                tiles_to_predict.append(
                    (c - xrange[0], r - yrange[0])
                )  # Store positions to map predictions later

        # Batch predict for all images
        predictions = predict_batch(img_paths_batch)

        # Map predictions to submap grid
        for pred, (c, r) in zip(predictions, tiles_to_predict):
            # Update the submap with the predicted label
            submap[c, r] = (
                1 if pred == 1 else 0
            )  # Assuming binary classification, adjust if necessary

        return submap

    def gt2map(self, gt_txt_path):
        matrix = np.loadtxt(gt_txt_path, dtype=int)
        print(f"annot map shape: {matrix.shape}")
        map = matrix[3:-3, 13:]

        map[map == 2] = 1
        map[map == 10] = 1
        map[map == 100] = 1

        # Ensure the array is binary
        map = (map > 0).astype(int)
        return map.transpose()

    def get_rawimagepath(self, tile_to_test):
        return self.get_tile_img_path(tile_to_test)

    def find_closest_image(self, tile_to_test, ref_rel_alt=20):
        """
        Find the closest image to a given tile based on NED coordinates.
        """
        gps_coordinates = self.gps_tile_dict[tile_to_test]
        tile_center_ned = [gps_coordinates[0], gps_coordinates[1], ref_rel_alt]
        image_paths_for_tile = self.get_tile_img_path(tile_to_test)

        closest_image_path = None
        minimum_distance = float("inf")

        for image_path in image_paths_for_tile:
            image_name = image_path.split("_tile")[0] + ".JPG"
            image_ned_coordinates = img_NED(os.path.join(self.row_imgs_dir, image_name))
            tile_ned_coordinates = gps_ned(
                tile_center_ned, ref_info=image_ned_coordinates
            )

            distance_to_tile = (
                tile_ned_coordinates[0] ** 2 + tile_ned_coordinates[1] ** 2
            ) ** 0.5

            if distance_to_tile < minimum_distance:
                minimum_distance = distance_to_tile
                closest_image_path = image_path

        return closest_image_path, minimum_distance

    def locate_wrt_tile(self, tile_to_test, ref_rel_alt=20):
        """
        Find the closest image to a given tile based on NED coordinates.
        """
        gps_coordinates = self.gps_tile_dict[tile_to_test]
        tile_center_gps = [gps_coordinates[0], gps_coordinates[1], ref_rel_alt]
        tile_center_ned = gps_ned(tile_center_gps, (tile_center_gps, [0, 0, 0]))
        ref_point = (tile_center_gps, tile_center_ned)
        tile_center_ned = gps_ned(tile_center_gps, ref_point)

        # print(f"tile center gps: {tile_center_gps}")
        # print(f"tile center ned: {tile_center_ned}")
        image_paths_for_tile = self.get_tile_img_path(tile_to_test)

        fov_corners_all = []
        centers = []

        for image_path in image_paths_for_tile:

            image_name = image_path.split("_tile")[0] + ".JPG"
            img_path = os.path.join(self.row_imgs_dir, image_name)
            tile_ned_coordinates = img_NED(img_path, ref_info=ref_point)
            # # tile_ned_coordinates = gps_ned(image_ned_coordinates[0], ref_info=(tile_center_ned, [0,0,0]))
            # print(tile_ned_coordinates)

            tile_ned_coordinates = tile_ned_coordinates[1]
            centers.append(tile_ned_coordinates)
            img_info = get_image_properties(img_path)
            fov_corners = np.array(
                self.L2.imgToWorldCoord(tile_ned_coordinates, img_info)
            )
            # fov_corners = np.array(L2.get_fov_corners_in_ned(T, img_info))
            fov_corners_all.append(fov_corners)
            distance_to_tile = (
                tile_ned_coordinates[0] ** 2 + tile_ned_coordinates[1] ** 2
            ) ** 0.5
        return fov_corners_all, centers

    def get_camera(self):
        return self.L2.camera_characteristics()


"""
Testing
"""

"""
tiles_dir = '/home/bota/Downloads/projtiles1/'
gps_csv = '/home/bota/Desktop/active_sensing/src/gpstiles.csv'
row_imgs_dir = "/media/bota/BOTA/wheat/APPEZZAMENTO_PICCOLO/"

# Initialize TileOperations
tile_ops = TileOperations(tiles_dir, gps_csv, row_imgs_dir)


# Test finding the closest image
tile_to_test = (13, 39)
# closest_image_path, minimum_distance = tile_ops.find_closest_image(tile_to_test)
# print(f"Selected tile image: {closest_image_path}")
# print(f"Distance to tile center: {minimum_distance:.2f}")
tile_ops.locate_wrt_tile(tile_to_test)
tile_ops.get_camera()
"""

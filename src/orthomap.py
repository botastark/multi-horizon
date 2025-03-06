#!/usr/bin/env python3
import pickle
import random
import sys
import os
import math

import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

from helper import gaussian_random_field

sys.path.append(os.path.abspath("/home/bota/Desktop/active_sensing"))
from binary_classifier.classifier import Predicter

# dataset
from PIL import Image, ImageFilter
from osgeo import gdal

gdal.UseExceptions()  # Enable exceptions to avoid the warning

desktop = "/home/bota/Desktop/active_sensing"
annotation_path = desktop + "/src/annotation.txt"
dataset_path = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
tile_ortomappixel_path = desktop + "/data/tomatotiles.txt"
model_path = (
    desktop + "/binary_classifier/models/best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth"
)
cache_dir = desktop + "/data/predictions_cache/"


class img_sampler:
    def __init__(self):
        self.focal_length = 0.01229  # 12.29mm lens in meters
        self.sensor_width = 0.017424  # sensor width in meters
        self.sensor_height = 0.0130548  # sensor height in meters
        self.resolution_x = 5280  # horizontal resolution (number of pixels)
        self.resolution_y = 3956  # vertical resolution (number of pixels)
        self.fov_h, self.fov_v = self._calculate_fov()

    def _calculate_fov(self):
        fov_horizontal = 2 * math.atan(self.sensor_width / (2 * self.focal_length))
        fov_vertical = 2 * math.atan(self.sensor_height / (2 * self.focal_length))
        return fov_horizontal, fov_vertical

    # Function to calculate the size of a 1m x 1m tile on the image in pixels
    def calculate_tile_size_on_image(self, altitude):
        # Calculate ground coverage at altitude
        coverage_horizontal = 2 * math.tan(self.fov_h / 2) * altitude
        coverage_vertical = 2 * math.tan(self.fov_v / 2) * altitude

        # Calculate tile size on image in pixels (1m x 1m tile)
        tile_size_image_horizontal = self.resolution_x / coverage_horizontal
        tile_size_image_vertical = self.resolution_y / coverage_vertical

        return int(tile_size_image_horizontal), int(tile_size_image_vertical)

    # Function to calculate the altitude from FOV and tile size on the image
    def calculate_altitude_from_fov_and_tile_size(self, size):
        tile_size_image_horizontal, tile_size_image_vertical = size
        # Calculate the ground coverage corresponding to the pixel size of the tile on the image
        coverage_horizontal = self.resolution_x / tile_size_image_horizontal
        coverage_vertical = self.resolution_y / tile_size_image_vertical

        # Calculate the altitude based on horizontal and vertical coverage
        altitude_horizontal = coverage_horizontal / (2 * math.tan(self.fov_h / 2))
        altitude_vertical = coverage_vertical / (2 * math.tan(self.fov_v / 2))

        # Average the two altitudes for a more accurate result
        altitude = (altitude_horizontal + altitude_vertical) / 2

        return altitude

    # Function for downsampling with Gaussian blur, with adjustable blur based on size
    def downsample_with_blur(self, image, target_size, original_size):
        # Calculate blur strength based on size difference
        blur_radius = max(0.5, abs(target_size[0] - original_size[0]) / 50)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return blurred.resize(target_size, Image.LANCZOS)

    def simulate_higher_altitude(
        self,
        image,
        target_altitude,
        original_altitude=20,
        base_blur_radius=2,
        base_noise_std=20,
        base_contrast_factor=0.8,
        base_brightness_factor=1.1,
    ):
        altitude_ratio = target_altitude / original_altitude

        # Step 1: Downsample the image (scale factor proportional to altitude ratio)
        downsample_factor = int(np.round(altitude_ratio))
        width, height = image.size
        new_size = (width // downsample_factor, height // downsample_factor)
        downsampled = image.resize(new_size, Image.BILINEAR)

        # Step 2: Apply Gaussian blur (scale blur radius proportionally)
        blur_radius = base_blur_radius * altitude_ratio
        blurred = downsampled.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Step 3: Add noise (scale noise standard deviation proportionally)
        noise_std = base_noise_std * altitude_ratio
        noisy_image = np.array(blurred).astype(np.float32)
        noise = np.random.normal(0, noise_std, noisy_image.shape).astype(np.float32)
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_image)

        # Step 4: Reduce contrast and brightness (scale factors proportionally)
        contrast_factor = base_contrast_factor / altitude_ratio
        brightness_factor = base_brightness_factor * altitude_ratio
        adjusted = Image.eval(
            noisy_image,
            lambda x: np.clip(
                contrast_factor * (x - 128) + 128 * brightness_factor, 0, 255
            ),
        )

        # Step 5: Resize back to original size (optional, if footprint is not enlarged)
        final_image = adjusted.resize((width, height), Image.BILINEAR)

        return final_image

    def img_at_alt(self, image, alt):

        r0, c0 = image.size
        orig_size = (r0, c0)
        # alts = round(self.calculate_altitude_from_fov_and_tile_size(orig_size), 2)
        target_size = self.calculate_tile_size_on_image(alt)

        if target_size[0] < image.size[0] and target_size[1] < image.size[1]:
            return self.downsample_with_blur(image, target_size, orig_size)
            # return self.simulate_higher_altitude(image, alt)
        else:
            return image


class Field:
    def __init__(
        self,
        grid_info,
        field_type,
        cache_dir=cache_dir,
        seed=123,
        model_path=model_path,
        ortomap_path=dataset_path,
        sweep="ig",
        a=1,
        b=0.015,
        h_range=[],
    ):
        self.grid_info = grid_info
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.a = a
        self.b = b
        start = h_range[0]
        num_values = 6
        end = h_range[-1]
        self.altitudes = np.round(np.linspace(start, end, num=num_values), decimals=2)

        if sweep == "sweep":
            self.sweep = True
        else:
            self.sweep = False

        if isinstance(field_type, int):
            # self.field_type = f"Gaussian_r{field_type}"
            self.field_type = "Gaussian"
            self.field_r = field_type
            self.ground_truth_map = gaussian_random_field(self.field_r, grid_info.shape)

        elif field_type == "Ortomap":
            self.field_type = field_type

            self.model_path = model_path
            self.ortomap_path = ortomap_path
            if not self.sweep:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)
                self._init_ortomap()

                self.img_sampler = img_sampler()
                self.predictions_cache = self._load_cache()

            else:
                self.ground_truth_map = self._read_annotations_to_matrix(
                    annotation_path
                )
                self.tiles = [
                    (row, col)
                    for row in range(0, self.grid_info.y)
                    for col in range(0, self.grid_info.x)
                ]

    def _cache_filepath(self):
        return os.path.join(self.cache_dir, "predictions.pkl")

    def _load_cache(self):
        filepath = self._cache_filepath()
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        self.predictions_cache = {}
        self._initialize_predictions()

        # return {}

    def _save_cache(self):
        with open(self._cache_filepath(), "wb") as f:
            pickle.dump(self.predictions_cache, f)

    def _initialize_predictions(self):

        for altitude in self.altitudes:
            z = np.zeros_like(self.ground_truth_map, dtype=int)
            for ind, (r, c) in enumerate(self.tiles):
                tile_pil_img = self._get_tile_img((r, c))
                tile_pil_img = self.img_sampler.img_at_alt(tile_pil_img, altitude)
                z.flat[ind] = int(self.predictor.predict(tile_pil_img))

            self.predictions_cache[altitude] = z

        self._save_cache()

    def reset(self):
        if self.field_type == "Gaussian":
            try:
                self.ground_truth_map = gaussian_random_field(
                    self.field_r, self.grid_info.shape
                )
                self.rng = np.random.default_rng(self.seed)
            except Exception as e:
                raise ValueError(
                    f"Couldn't reset {self.field_type} field with rad {self.field_r}: {e}"
                )

    def _init_ortomap(self):

        self.predictor = Predicter(model_weights_path=self.model_path, num_classes=2)
        dataset = gdal.Open(self.ortomap_path)
        band1 = dataset.GetRasterBand(1)  # Red channel
        band2 = dataset.GetRasterBand(2)  # Green channel
        band3 = dataset.GetRasterBand(3)  # Blue channel

        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        self.img = np.dstack((b1, b2, b3))

        self.tile_pixel_loc = self._parse_tile_file(tile_ortomappixel_path)
        self.ground_truth_map = self._read_annotations_to_matrix(annotation_path)
        # self.tiles = [(row, col) for row in range(3, 113) for col in range(13, 73)]
        self.tiles = [(row, col) for row in range(0, 110) for col in range(0, 60)]

    def _parse_tile_file(self, file_path):
        """
        Parse a text file containing tile information and store it in an np.ndarray format.
        np.ndarray: A matrix where each element M[row][col] = [[X coord, X coord+length], [Y coord, Y coord+height]].
        """
        tile_data = {}
        max_row, max_col = 0, 0

        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(":")
                tile_id = parts[0].strip()
                coords = list(map(int, parts[1].strip().split(",")))

                row_col = tile_id.replace("tile", "").split("_")
                row = int(row_col[0])
                col = int(row_col[1])

                max_row = max(max_row, row)
                max_col = max(max_col, col)
                tile_data[(row, col)] = coords

        matrix = np.empty((max_row + 1, max_col + 1), dtype=object)

        for (row, col), coords in tile_data.items():
            y, x, length, height = coords
            matrix[row][col] = [[x, x + length], [y, y + height]]

        return matrix

    def _read_annotations_to_matrix(self, file_path, cut=True):
        try:
            matrix = np.loadtxt(file_path, dtype=int)  # Adjust dtype if necessary
            matrix[matrix == 10] = 1  # Adjust the value to match the label
            matrix[matrix == 2] = 1
            assert matrix.all() in [
                0,
                1,
            ], "Invalid label values found in the annotation file"
            if cut:

                matrix = matrix[3:113, 13:73]
            assert matrix.shape == self.grid_info.shape, "check grid size and length"
            return matrix
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")

    def _get_orto_bbox(self, tile):
        if isinstance(tile, tuple):
            r, c = tile
        else:
            r, c = tile[0], tile[1]
        r += 3
        c += 13
        tile_coords = self.tile_pixel_loc[r][c]
        x_range = slice(tile_coords[0][0], tile_coords[0][1])
        y_range = slice(tile_coords[1][0], tile_coords[1][1])
        return x_range, y_range

    def _get_tile_img(self, tile):
        x_range, y_range = self._get_orto_bbox(tile)
        cropped_img = self.img[x_range, y_range, :]
        return Image.fromarray(cropped_img)

    def get_tile_info(self, tile):
        return self.get_tile_img(tile), self.ground_truth_map[tile[0], tile[1]]

    def convert_xy_ij(self, x, y):
        if self.grid_info.center:
            center_i, center_j = (dim // 2 for dim in self.grid_info.shape)
            j = x / self.grid_info.length + center_j
            i = -y / self.grid_info.length + center_i
        else:
            j = x / self.grid_info.length
            i = self.grid_info.shape[0] - y / self.grid_info.length
        return int(i), int(j)

    def get_visible_range(self, uav_pos, fov=60, index_form=False):
        """
        calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
        """
        grid_length = self.grid_info.length
        fov_rad = np.deg2rad(fov) / 2

        x_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length

        x_range = [0, self.grid_info.x]
        y_range = [0, self.grid_info.y]

        if self.grid_info.center:
            x_range = [-self.grid_info.x / 2, self.grid_info.x / 2]
            y_range = [-self.grid_info.y / 2, self.grid_info.y / 2]
        # print(f"ranges x{x_range} y{y_range}")
        # print(f"pos: {uav_pos.position} {uav_pos.altitude}")

        x_min, x_max = np.clip(
            [uav_pos.position[0] - x_dist, uav_pos.position[0] + x_dist], *x_range
        )
        y_min, y_max = np.clip(
            [uav_pos.position[1] - y_dist, uav_pos.position[1] + y_dist], *y_range
        )
        if x_max - x_min == 0 or y_max - y_min == 0:
            return [[0, 0], [0, 0]]

        # print(f"visible ranges x:({x_min}:{x_max}) y:({y_min}:{y_max})")
        if not index_form:
            return [[x_min, x_max], [y_min, y_max]]
        i_max, j_min = self.convert_xy_ij(x_min, y_min)
        i_min, j_max = self.convert_xy_ij(x_max, y_max)
        # print(f"visible ranges i:({i_min}:{i_max}) j:({j_min}:{j_max})")
        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, uav_pos, sigmas=None):
        [[i_min, i_max], [j_min, j_max]] = self.get_visible_range(
            uav_pos, index_form=True
        )

        fp_vertices_ij = {
            "ul": np.array([i_min, j_min]),
            "bl": np.array([i_max, j_min]),
            "ur": np.array([i_min, j_max]),
            "br": np.array([i_max, j_max]),
        }

        if self.field_type == "Gaussian" and self.ground_truth_map is not None:
            # get "perfect" observation from ground truth
            submap = self.ground_truth_map[i_min:i_max, j_min:j_max]
            # add sigma noise to observation: if sigma not given, calculate it
            if sigmas is None:
                sigma = self.a * (1 - np.exp(-self.b * uav_pos.altitude))
                sigmas = [sigma, sigma]

            sigma0, sigma1 = sigmas[0], sigmas[1]
            random_values = self.rng.random(submap.shape)
            success0 = random_values <= 1.0 - sigma0
            success1 = random_values <= 1.0 - sigma1
            z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
            z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
            z = np.where(submap == 0, z0, z1)

        elif self.field_type == "Ortomap":
            x = np.arange(i_min, i_max, 1)
            y = np.arange(j_min, j_max, 1)
            x, y = np.meshgrid(x, y, indexing="ij")
            z = np.zeros_like(x, dtype=int)
            if self.sweep:
                z = self.ground_truth_map[i_min:i_max, j_min:j_max]
                return fp_vertices_ij, z
            if self.predictor is not None:
                # label = np.zeros_like(x, dtype=int)
                if self.predictions_cache is not None:
                    approx_alt = round(uav_pos.altitude, 2)
                    if not approx_alt in self.predictions_cache.keys():
                        print(f"uav alt:{uav_pos.altitude} and approx {approx_alt}")
                        print(f"pred cache alts: {list(self.predictions_cache.keys())}")

                    pred_at_alt = self.predictions_cache[approx_alt]
                    assert (
                        pred_at_alt.shape == self.ground_truth_map.shape
                    ), f"check prediction cache shape: its {pred_at_alt.shape} and gt shape {self.ground_truth_map.shape}"
                    z = pred_at_alt[i_min:i_max, j_min:j_max]
                else:

                    for ind, (r, c) in enumerate(zip(x.flatten(), y.flatten())):
                        tile_pil_img = self._get_tile_img((r, c))
                        tile_pil_img = self.img_sampler.img_at_alt(
                            tile_pil_img, uav_pos.altitude
                        )
                        z.flat[ind] = int(self.predictor.predict(tile_pil_img))
                        # label.flat[ind] = self.ground_truth_map[r, c]
        return fp_vertices_ij, z

    def get_ground_truth(self):
        return self.ground_truth_map

    """"
    Sampler upd
    """

    def sample_random_tiles(self):
        """
        Sample a random "free" (label 0) and a random "occupied" (label 1) tile from the ground truth map.

        Returns:
            free_tile (tuple): (row, column) of a random free tile.
            occupied_tile (tuple): (row, column) of a random occupied tile.
        """
        # Indices of free and  occupied tiles
        free_indices = np.argwhere(self.ground_truth_map == 0)
        occupied_indices = np.argwhere(self.ground_truth_map == 1)

        if len(free_indices) == 0:
            raise ValueError("No free tiles (label 0) found in the ground truth map.")
        if len(occupied_indices) == 0:
            raise ValueError(
                "No occupied tiles (label 1) found in the ground truth map."
            )

        # Randomly select one free and one occupied tile  (row, column)
        free_tile = tuple(free_indices[random.randint(0, len(free_indices) - 1)])
        occupied_tile = tuple(
            occupied_indices[random.randint(0, len(occupied_indices) - 1)]
        )

        return free_tile, occupied_tile

    def _pred_model(self, true_matrix, altitude):
        tile_m0, tile_m1 = self.sample_random_tiles()
        img_m0, img_m1 = self._get_tile_img(tile_m0), self._get_tile_img(tile_m1)
        img_m0 = self.img_sampler.img_at_alt(img_m0, altitude)
        img_m1 = self.img_sampler.img_at_alt(img_m1, altitude)

        z_m0 = int(self.predictor.predict(img_m0))
        z_m1 = int(self.predictor.predict(img_m1))

        observation_matrix = [z_m0, z_m1]
        return observation_matrix

    def _sensor_model(self, true_matrix, altitude):
        sig = self.a * (1 - np.exp(-self.b * altitude))
        P_z_equals_m = 1 - sig
        P_z_not_equals_m = sig

        rows, cols = true_matrix.shape
        observation_matrix = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                if true_matrix[i, j] == 1:
                    observation_matrix[i, j] = np.random.choice(
                        [1, 0], p=[P_z_equals_m, P_z_not_equals_m]
                    )
                else:
                    observation_matrix[i, j] = np.random.choice(
                        [0, 1], p=[P_z_equals_m, P_z_not_equals_m]
                    )

        return observation_matrix

    def _sampler(self, true_matrix, altitude, N, sensor=True):
        cumulative_observation = np.empty(true_matrix.shape)
        for i in range(N):
            if sensor:
                observation_matrix = self._sensor_model(true_matrix, altitude)
            else:
                observation_matrix = self._pred_model(true_matrix, altitude)

            if i == 0:
                cumulative_observation = observation_matrix.copy()
                continue
            cumulative_observation = np.vstack(
                [cumulative_observation, observation_matrix]
            )
        return cumulative_observation

    def _calc_n(self, h, e=np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03])):
        p = self.a * (1 - np.exp(-self.b * h))
        p_ = 1 - p
        return np.round(1.96 * 1.96 * p * p_ / e / e, decimals=0)

    def _get_N(self, altitudes):
        errors = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03]
        n_per_e = {}
        for h in altitudes:
            n_values = self._calc_n(h)
            n_per_e[h] = {e: n for e, n in zip(errors, n_values)}
        return n_per_e

    def _get_confusion_matrix(self, altitude, N, sensor=True):
        true_matrix = np.array([0, 1])
        true_matrix = np.expand_dims(true_matrix, axis=0)
        observation = self._sampler(true_matrix, altitude, int(N), sensor=sensor)
        n = int(observation.shape[0] / true_matrix.shape[0])
        true_matrix_ = np.tile(true_matrix, (n, 1))

        c = confusion_matrix(
            true_matrix_.ravel(), observation.ravel(), normalize="true"
        ).T
        c = np.clip(np.nan_to_num(np.round(c, 2) + 1e-3), 1e-3, 1)

        s0, s1 = c[1, 0], c[0, 1]
        return c, (s0, s1)

    def init_s0_s1(self, e=0.3, sensor=True):

        conf_dict = {}
        Ns = self._get_N(self.altitudes)
        for altitude in self.altitudes:
            conf_dict[altitude] = self._get_confusion_matrix(
                altitude, Ns[altitude][e], sensor=sensor
            )[1]
        return conf_dict


# class grid_info:
#     x = 60  # 60
#     y = 110  # 110 for real field
#     length = 1  # 1
#     shape = (int(y / length), int(x / length))
#     center = True


# map = Field(grid_info, "Ortomap", seed=1)
# free, occ = map.sample_random_tiles()
# print(f"free {free}")
# print(f"occ {occ}")

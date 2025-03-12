#!/usr/bin/env python3
import pickle
import random
import sys
import os
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from helper import gaussian_random_field


# Add project path and import the classifier
sys.path.append(os.path.abspath("/home/bota/Desktop/active_sensing"))
from binary_classifier.classifier import Predicter

from PIL import Image, ImageFilter
from osgeo import gdal

gdal.UseExceptions()  # Enable GDAL exceptions


desktop = "/home/bota/Desktop/active_sensing"
annotation_path = desktop + "/src/annotation.txt"
dataset_path = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
tile_ortomappixel_path = desktop + "/data/tomatotiles.txt"
model_path = (
    desktop + "/binary_classifier/models/best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth"
)
cache_dir = desktop + "/data/predictions_cache/"


# Global paths and configuration
DESKTOP_PATH = "/home/bota/Desktop/active_sensing"
ANNOTATION_PATH = os.path.join(DESKTOP_PATH, "src/annotation.txt")
DATASET_PATH = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
TILE_PIXEL_PATH = os.path.join(DESKTOP_PATH, "data/tomatotiles.txt")
MODEL_PATH = os.path.join(
    DESKTOP_PATH,
    "binary_classifier/models/best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth",
)
CACHE_DIR = os.path.join(DESKTOP_PATH, "data/predictions_cache/")


class ImageSampler:
    """Class to perform various image sampling and simulation operations based on altitude."""

    def __init__(self):
        # Camera parameters (in meters) and sensor resolution (in pixels)
        self.focal_length = 0.01229
        self.sensor_width = 0.017424
        self.sensor_height = 0.0130548
        self.resolution_x = 5280
        self.resolution_y = 3956
        self.fov_h, self.fov_v = self._calculate_fov()

    def _calculate_fov(self):
        """Calculate horizontal and vertical field-of-view (FOV) in radians."""
        fov_horizontal = 2 * math.atan(self.sensor_width / (2 * self.focal_length))
        fov_vertical = 2 * math.atan(self.sensor_height / (2 * self.focal_length))
        return fov_horizontal, fov_vertical

    def calculate_tile_size_on_image(self, altitude):
        """
        Calculate the size in pixels of a 1m x 1m ground tile at a given altitude.
        """
        coverage_h = 2 * math.tan(self.fov_h / 2) * altitude
        coverage_v = 2 * math.tan(self.fov_v / 2) * altitude
        tile_pixels_h = self.resolution_x / coverage_h
        tile_pixels_v = self.resolution_y / coverage_v
        return int(tile_pixels_h), int(tile_pixels_v)

    def calculate_altitude_from_tile_size(self, tile_size):
        """
        Estimate the altitude from the size of a ground tile in the image.
        """
        tile_pixels_h, tile_pixels_v = tile_size
        coverage_h = self.resolution_x / tile_pixels_h
        coverage_v = self.resolution_y / tile_pixels_v
        altitude_h = coverage_h / (2 * math.tan(self.fov_h / 2))
        altitude_v = coverage_v / (2 * math.tan(self.fov_v / 2))
        return (altitude_h + altitude_v) / 2

    def downsample_with_blur(self, image, target_size, original_size):
        """
        Downsample the image with a Gaussian blur to avoid aliasing.
        """
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
        """
        Simulate the effect of capturing an image from a higher altitude.
        Applies downsampling, blur, noise, and contrast/brightness adjustments.
        """
        altitude_ratio = target_altitude / original_altitude
        width, height = image.size
        downsample_factor = max(1, int(round(altitude_ratio)))
        new_size = (width // downsample_factor, height // downsample_factor)
        downsampled = image.resize(new_size, Image.BILINEAR)

        # Apply altitude-scaled Gaussian blur
        blur_radius = base_blur_radius * altitude_ratio
        blurred = downsampled.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Add noise scaled with altitude ratio
        noise_std = base_noise_std * altitude_ratio
        noisy_arr = np.array(blurred).astype(np.float32)
        noise = np.random.normal(0, noise_std, noisy_arr.shape).astype(np.float32)
        noisy_arr = np.clip(noisy_arr + noise, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_arr)

        # Adjust contrast and brightness
        contrast_factor = base_contrast_factor / altitude_ratio
        brightness_factor = base_brightness_factor * altitude_ratio
        adjusted = Image.eval(
            noisy_image,
            lambda x: np.clip(
                contrast_factor * (x - 128) + 128 * brightness_factor, 0, 255
            ),
        )
        return adjusted.resize((width, height), Image.BILINEAR)

    def get_image_at_altitude(self, image, altitude):
        """
        Return an image adjusted to simulate capture at the specified altitude.
        """
        original_size = image.size
        target_size = self.calculate_tile_size_on_image(altitude)
        if target_size[0] < original_size[0] and target_size[1] < original_size[1]:
            return self.downsample_with_blur(image, target_size, original_size)
        return image


class Field:
    """
    Class representing a field, either as a Gaussian random field or an orthomap,
    with methods for loading data, obtaining observations, and simulating sensor noise.
    """

    def __init__(
        self,
        grid_info,
        field_type,
        cache_dir=CACHE_DIR,
        seed=123,
        model_path=MODEL_PATH,
        ortomap_path=DATASET_PATH,
        sweep="ig",
        a=1,
        b=0.015,
        h_range=[],
    ):
        """
        Initialize the field.

        Args:
            grid_info: Grid configuration (attributes: shape, x, y, center, length).
            field_type: Integer for Gaussian (interpreted as the radius) or "Ortomap" for orthomap.
            cache_dir: Directory for caching predictions.
            seed: Seed for the random generator.
            model_path: Path to the classifier model.
            ortomap_path: Path to the ortho image.
            sweep: Strategy ("sweep" for sweeping mode; any other value uses prediction mode).
            a: Coefficient for sensor noise.
            b: Decay factor for sensor noise with altitude.
            h_range: List defining the altitude range.
        """

        self.grid_info = grid_info
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.a = a
        self.b = b

        start_alt = h_range[0]
        end_alt = h_range[-1]
        num_altitudes = 6
        self.altitudes = np.round(
            np.linspace(start_alt, end_alt, num=num_altitudes), decimals=2
        )
        self.sweep_mode = True if sweep == "sweep" else False

        if isinstance(field_type, int):
            self.field_type = "Gaussian"
            self.field_radius = field_type
            self.ground_truth_map = gaussian_random_field(
                self.field_radius, grid_info.shape
            )
        elif field_type == "Ortomap":
            self.field_type = field_type

            self.model_path = model_path
            self.ortomap_path = ortomap_path
            if not self.sweep_mode:
                self.cache_dir = cache_dir
                os.makedirs(self.cache_dir, exist_ok=True)
                self._init_ortomap()

                self.img_sampler = ImageSampler()
                self.predictions_cache = self._load_cache()

            else:
                self.ground_truth_map = self._read_annotations_to_matrix(
                    ANNOTATION_PATH
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
        """
        Generate classifier predictions for each altitude and cache the results.
        """
        for alt in self.altitudes:
            pred_map = np.zeros_like(self.ground_truth_map, dtype=int)
            for idx, (row, col) in enumerate(self.tiles):
                tile_img = self.get_tile_img((row, col))
                tile_img = self.img_sampler.get_image_at_altitude(tile_img, alt)
                pred_map.flat[idx] = int(self.predictor.predict(tile_img))
            self.predictions_cache[alt] = pred_map
        self._save_cache()

    def reset(self):
        """Reset the Gaussian field and random generator."""
        if self.field_type == "Gaussian":
            try:
                self.ground_truth_map = gaussian_random_field(
                    self.field_radius, self.grid_info.shape
                )
                self.rng = np.random.default_rng(self.seed)
            except Exception as e:
                raise ValueError(
                    f"Couldn't reset Gaussian field with radius {self.field_radius}: {e}"
                )

    def _init_ortomap(self):
        """
        Load the orthomap image, tile information, and annotations.
        """
        self.predictor = Predicter(model_weights_path=self.model_path, num_classes=2)
        dataset = gdal.Open(self.ortomap_path)
        band1 = dataset.GetRasterBand(1)  # Red channel
        band2 = dataset.GetRasterBand(2)  # Green channel
        band3 = dataset.GetRasterBand(3)  # Blue channel

        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        self.img = np.dstack((b1, b2, b3))

        self.tile_pixel_loc = self._parse_tile_file(TILE_PIXEL_PATH)
        self.ground_truth_map = self._read_annotations_to_matrix(ANNOTATION_PATH)
        # self.tiles = [(row, col) for row in range(3, 113) for col in range(13, 73)]
        self.tiles = [(row, col) for row in range(0, 110) for col in range(0, 60)]

    def _parse_tile_file(self, file_path):
        """
        Parse a tile file into a matrix where each element is [[x_start, x_end], [y_start, y_end]].
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
        """
        Read the annotation file and convert it to a binary matrix.
        """
        try:
            matrix = np.loadtxt(file_path, dtype=int)
            matrix[matrix == 10] = 1
            matrix[matrix == 2] = 1
            assert matrix.all() in [
                0,
                1,
            ], "Invalid label values found in the annotation file"
            if cut:

                matrix = matrix[3:113, 13:73]
            assert (
                matrix.shape == self.grid_info.shape
            ), "Grid size mismatch with annotation matrix"
            return matrix
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")

    def _get_orto_bbox(self, tile):
        """
        Get the bounding box (pixel ranges) for the given tile.
        """
        if isinstance(tile, tuple):
            row, col = tile
        else:
            row, col = tile[0], tile[1]
        row += 3
        col += 13
        tile_coords = self.tile_pixel_loc[row][col]
        x_range = slice(tile_coords[0][0], tile_coords[0][1])
        y_range = slice(tile_coords[1][0], tile_coords[1][1])
        return x_range, y_range

    def _get_tile_img(self, tile):
        """
        Return the tile image as a PIL Image.
        """
        x_range, y_range = self._get_orto_bbox(tile)
        cropped_img = self.img[x_range, y_range, :]
        return Image.fromarray(cropped_img)

    def get_tile_info(self, tile):
        """
        Return the tile image and its ground truth label.
        """
        return self._get_tile_img(tile), self.ground_truth_map[tile[0], tile[1]]

    def convert_xy_ij(self, x, y):
        """
        Convert (x, y) coordinates to grid indices (i, j).
        """
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
        Calculate the visible grid range (or its indices) based on the UAV position and field-of-view.
        """
        grid_length = self.grid_info.length
        fov_rad = np.deg2rad(fov) / 2

        x_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length

        if self.grid_info.center:
            x_range = [-self.grid_info.x / 2, self.grid_info.x / 2]
            y_range = [-self.grid_info.y / 2, self.grid_info.y / 2]
        else:
            x_range = [0, self.grid_info.x]
            y_range = [0, self.grid_info.y]

        x_min, x_max = np.clip(
            [uav_pos.position[0] - x_dist, uav_pos.position[0] + x_dist], *x_range
        )
        y_min, y_max = np.clip(
            [uav_pos.position[1] - y_dist, uav_pos.position[1] + y_dist], *y_range
        )
        if x_max - x_min == 0 or y_max - y_min == 0:
            return [[0, 0], [0, 0]]

        if not index_form:
            return [[x_min, x_max], [y_min, y_max]]
        i_max, j_min = self.convert_xy_ij(x_min, y_min)
        i_min, j_max = self.convert_xy_ij(x_max, y_max)
        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, uav_pos, sigmas=None):
        """
        Get observations from the field based on UAV position.

        Returns:
            A tuple of (footprint vertices, observation matrix).
        """
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
            if self.sweep_mode:
                z = self.ground_truth_map[i_min:i_max, j_min:j_max]
                return fp_vertices_ij, z
            if self.predictor is not None:
                # label = np.zeros_like(x, dtype=int)
                if self.predictions_cache is not None:
                    approx_alt = round(uav_pos.altitude, 2)
                    if not approx_alt in self.predictions_cache.keys():
                        print(
                            f"UAV altitude: {uav_pos.altitude} (approx {approx_alt}) not in prediction cache."
                        )
                        print(
                            f"Available altitudes: {list(self.predictions_cache.keys())}"
                        )
                        closest_alt = min(
                            self.predictions_cache.keys(),
                            key=lambda k: abs(k - approx_alt),
                        )
                        pred_at_alt = self.predictions_cache[closest_alt]
                    else:
                        pred_at_alt = self.predictions_cache[approx_alt]

                    # pred_at_alt = self.predictions_cache[approx_alt]
                    assert (
                        pred_at_alt.shape == self.ground_truth_map.shape
                    ), f"Prediction cache shape mismatch: {pred_at_alt.shape} vs {self.ground_truth_map.shape}"
                    observations = pred_at_alt[i_min:i_max, j_min:j_max]
                else:

                    for ind, (r, c) in enumerate(zip(x.flatten(), y.flatten())):
                        tile_pil_img = self._get_tile_img((r, c))
                        tile_pil_img = self.img_sampler.img_at_alt(
                            tile_pil_img, uav_pos.altitude
                        )
                        observations.flat[ind] = int(
                            self.predictor.predict(tile_pil_img)
                        )
                        # label.flat[ind] = self.ground_truth_map[r, c]
        return fp_vertices_ij, observations

    def get_ground_truth(self):
        """Return the ground truth map of the field."""
        return self.ground_truth_map

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
        """
        Generate classifier predictions for a randomly selected free and occupied tile.
        """
        tile_m0, tile_m1 = self.sample_random_tiles()
        img_m0, img_m1 = self._get_tile_img(tile_m0), self._get_tile_img(tile_m1)
        img_m0 = self.img_sampler.get_image_at_altitude(img_m0, altitude)
        img_m1 = self.img_sampler.get_image_at_altitude(img_m1, altitude)

        z_m0 = int(self.predictor.predict(img_m0))
        z_m1 = int(self.predictor.predict(img_m1))

        return [z_m0, z_m1]

    def _sensor_model(self, true_matrix, altitude):
        """
        Simulate sensor observations on the true matrix with altitude-dependent noise.
        """
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

    def _sampler(self, true_matrix, altitude, N, use_sensor=True):
        """
        Generate N observations using either the sensor or prediction model.
        """
        observations = [
            (
                self._sensor_model(true_matrix, altitude)
                if use_sensor
                else self._pred_model(true_matrix, altitude)
            )
            for _ in range(N)
        ]
        return np.vstack(observations)

    def _calc_sample_size(
        self, altitude, error_levels=np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03])
    ):
        """
        Calculate required sample sizes for given error thresholds based on sensor noise.
        """
        p = self.a * (1 - np.exp(-self.b * altitude))
        p_comp = 1 - p
        return np.round((1.96**2) * p * p_comp / (error_levels**2), decimals=0)

    def _get_sample_sizes(self, altitudes):
        errors = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03]
        sample_sizes = {}
        for alt in altitudes:
            n_vals = self._calc_sample_size(alt)
            sample_sizes[alt] = {e: n for e, n in zip(errors, n_vals)}
        return sample_sizes

    def _get_confusion_matrix(self, altitude, N, use_sensor=True):
        """
        Compute the confusion matrix for a binary true matrix using N observations.
        """
        true_matrix = np.array([0, 1]).reshape(1, -1)
        observations = self._sampler(
            true_matrix, altitude, int(N), use_sensor=use_sensor
        )
        n = observations.shape[0] // true_matrix.shape[0]
        true_repeated = np.tile(true_matrix, (n, 1))
        conf = confusion_matrix(
            true_repeated.ravel(), observations.ravel(), normalize="true"
        ).T
        conf = np.clip(np.nan_to_num(np.round(conf, 2) + 1e-3), 1e-3, 1)
        s0, s1 = conf[1, 0], conf[0, 1]
        return conf, (s0, s1)

    def init_s0_s1(self, e=0.3, sensor=True):
        """
        Initialize confusion matrix parameters (s0, s1) for each altitude.
        """
        conf_dict = {}
        sample_sizes = self._get_sample_sizes(self.altitudes)
        for alt in self.altitudes:
            conf_dict[alt] = self._get_confusion_matrix(
                alt, sample_sizes[alt][e], use_sensor=sensor
            )[1]
        return conf_dict

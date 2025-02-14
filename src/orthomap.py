#!/usr/bin/env python3
import sys
import os
import math
from turtle import position
import numpy as np

from helper import gaussian_random_field

sys.path.append(os.path.abspath("/home/bota/Desktop/active_sensing"))
from binary_classifier.classifier import Predicter

import pickle
from proj import camera

# dataset
from PIL import Image
from osgeo import gdal


import torch
import torch.nn.functional
import random


# def __init__(self, ortomap_path, annotation_path, tile_ortomappixel_path):

annotation_path = "/home/bota/Desktop/active_sensing/src/annotation.txt"
dataset_path = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
tile_ortomappixel_path = "/home/bota/Desktop/active_sensing/data/tomatotiles.txt"
model_path = "/home/bota/Desktop/active_sensing/binary_classifier/models/best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth"


class Field:
    def __init__(
        self,
        grid_info,
        field_type,
        model_path=model_path,
        ortomap_path=dataset_path,
    ):
        self.grid_info = grid_info
        if isinstance(field_type, int):
            # self.field_type = f"Gaussian_r{field_type}"
            self.field_type = "Gaussian"
            self.ground_truth_map = gaussian_random_field(field_type, grid_info.shape)
            self.a = 1
            self.b = 0.015

        elif field_type == "Ortomap":
            self.field_type = field_type
            self.model_path = model_path
            self.ortomap_path = ortomap_path
            self._init_ortomap()
        print("field is ready!")

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
        self.tiles = [(row, col) for row in range(3, 113) for col in range(13, 73)]

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
            matrix = matrix.T
            assert matrix.shape == self.grid_info.shape
            return matrix
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")

    def _get_orto_bbox(self, tile):
        if isinstance(tile, tuple):
            r, c = tile
        else:
            r, c = tile[0], tile[1]
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
            center_j, center_i = (dim // 2 for dim in self.grid_info.shape)
            j = x / self.grid_info.length + center_j
            i = -y / self.grid_info.length + center_i
        else:
            j = x / self.grid_info.length
            i = self.grid_info.shape[1] - y / self.grid_info.length
        return int(i), int(j)

    def get_visible_range(self, uav_pos, fov=60, index_form=False):
        """
        calculates indices of camera footprints (part of terrain (therefore terrain indices) seen by camera at a given UAV pos and alt)
        """
        grid_length = self.grid_info.length
        fov_rad = np.deg2rad(fov) / 2

        x_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length
        y_dist = round(uav_pos.altitude * math.tan(fov_rad) / grid_length) * grid_length

        # print(f"dist x:{x_dist} y:{y_dist}")
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
        i_min, j_min = self.convert_xy_ij(x_min, y_min)
        i_max, j_max = self.convert_xy_ij(x_max, y_max)
        if not self.grid_info.center:
            i_min, i_max = i_max, i_min

        # print(f"visible ranges i:({i_min}:{i_max}) j:({j_min}:{j_max})")
        return [[i_min, i_max], [j_min, j_max]]

    def get_observations(self, uav_pos, sigmas=None):
        [[x_min_id, x_max_id], [y_min_id, y_max_id]] = self.get_visible_range(
            uav_pos, index_form=True
        )
        #     print(f"range x:{x_min_id} - {x_max_id}")
        #     print(f"range y:{y_min_id} - {y_max_id}")

        x = np.arange(x_min_id, x_max_id, 1)
        y = np.arange(y_min_id, y_max_id, 1)
        x, y = np.meshgrid(x, y, indexing="ij")

        if self.field_type == "Gaussian" and self.ground_truth_map is not None:
            # get "perfect" observation from ground truth
            submap = self.ground_truth_map[x_min_id:x_max_id, y_min_id:y_max_id]
            # add sigma noise to observation: if sigma not given, calculate it
            if sigmas is None:
                sigma = self.a * (1 - np.exp(-self.b * uav_pos.altitude))
                sigmas = [sigma, sigma]

            sigma0, sigma1 = sigmas[0], sigmas[1]
            rng = np.random.default_rng()
            random_values = rng.random(submap.shape)
            success0 = random_values <= 1.0 - sigma0
            success1 = random_values <= 1.0 - sigma1
            z0 = np.where(np.logical_and(success0, submap == 0), 0, 1)
            z1 = np.where(np.logical_and(success1, submap == 1), 1, 0)
            z = np.where(submap == 0, z0, z1)
        elif self.field_type == "Ortomap" and self.predictor is not None:
            z = np.zeros_like(x, dtype=int)
            # label = np.zeros_like(x, dtype=int)
            for ind, (r, c) in enumerate(zip(x.flatten(), y.flatten())):
                tile_pil_img = self._get_tile_img((r, c))
                z.flat[ind] = int(self.predictor.predict(tile_pil_img))
                # label.flat[ind] = self.ground_truth_map[r, c]
        return x, y, z

    def get_ground_truth(self):
        return self.ground_truth_map

    # def __len__(self):
    # return len(self.tiles)


# tester for non center:

# """


# class grid_info:
#     x = 50  # 60
#     y = 50  # 110 for real field
#     length = 0.125  # 1
#     shape = (int(x / length), int(y / length))
#     center = True


class grid_info:
    x = 50  # 60
    y = 50  # 110 for real field
    length = 0.125
    shape = (int(x / length), int(y / length))
    center = False


class uav_pos:
    altitude = 5
    position = (0, 0)


print(f"grid info:{grid_info.shape}")

field_type = 4
field_type = "Ortomap"
map = Field(
    grid_info,
    field_type,
    # model_path=model_path,
    # ortomap_path=dataset_path,
)
# xy = [
#     (-30, 55),
#     (0, 55),
#     (30, 55),
#     (30, 0),
#     (30, -55),
#     (-30, -55),
#     (-30, 0),
# ]
# ij = [
#     (0, 0),
#     (0, 30),
#     (0, 60),
#     (55, 60),
#     (110, 60),
#     (110, 0),
#     (55, 0),
# ]
xy_not_center = [
    (0, 0),
    #   (60, 0), (60, 110), (0, 110)
]

ij_not_center = [
    (50, 0),
    #   (110, 60), (0, 60), (0, 0)
]


for position, pixel in zip(xy_not_center, ij_not_center):
    # print(f"point {point} pixel  {pixel} ij {map.convert_xy_ij(point[0], point[1])}")
    assert pixel == map.convert_xy_ij(position[0], position[1])
    # uav_pos.position = position
    # visible_ij = map.get_visible_range(uav_pos, index_form=True)

print("passed!")
# x, y, z = map.get_observations(uav_pos)
# print(f"{z.shape}")
# gt = map.get_ground_truth()
# print(f"gt shape:{gt.shape}")

"""
print(f"center: {center} uav pos:{uav_pos.position}")
print(f"id: {get_range(uav_pos, grid_info, center=center, index_form=True)}")
print(f"pos: {get_range(uav_pos, grid_info, center=center, index_form=False)}")
uav_pos.position = (10, 0)
print(f"center: {center} uav pos:{uav_pos.position}")
print(f"id: {get_range(uav_pos, grid_info, center=center, index_form=True)}")
print(f"pos: {get_range(uav_pos, grid_info, center=center, index_form=False)}")
"""

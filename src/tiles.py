# tiles to map
import os
import numpy as np
from classifier import predict

def parse_tiles(tile_list):
    tiles = []
    for tile in tile_list:
        _, row_col = tile.split("tile")
        row_col,_  = row_col.split("_crop")
        row, col = map(int, row_col.split("_"))
        tiles.append((row, col))
    return tiles

def init_tiles(dir_path):
    folder = os.listdir(dir_path)
    tile_to_img_dict={}
    for file_path in folder:
        tile_loc = parse_tiles([file_path])[0]
        if tile_loc in tile_to_img_dict:
            temp = tile_to_img_dict[tile_loc]
        else:
            temp = []
        temp.append(file_path)
        tile_to_img_dict[tile_loc] = temp
    return tile_to_img_dict

def get_tile_img_path(tile_loc, tile_to_img_dict):
    if tile_loc in tile_to_img_dict:
        return tile_to_img_dict[tile_loc]
    else:
        return None

def observed_submap(xrange, yrange, tile_to_img_dict):
    submap = np.zeros(shape=(xrange[1]-xrange[0],yrange[1]-yrange[0]))
    # xrange = (xrange[0]+13, xrange[1]+13)
    for c in range(xrange[0], xrange[1]):
        for r in range(yrange[0], yrange[1]):
            tile_loc = (r+3,c+13)#offset to make sure to operate only in 100% coverage rect
            img_path = dir_all_tiles+get_tile_img_path(tile_loc, tile_to_img_dict)[0]
            pred = predict(img_path)[0]
            if pred == 2:
                pred = 1
            submap[c-xrange[0],r-yrange[0]]=pred
    return submap


dir_all_tiles = '/home/bota/Downloads/projtiles1/'

# tile_to_img_dict = init_tiles(dir_all_tiles)
# print(get_tile_img_path((1,1), tile_to_img_dict))
# print(get_tile_img_path((15,15), tile_to_img_dict))
# xrange=(0,5)
# yrange=(0,5)
# submap = observed_submap(xrange, yrange, tile_to_img_dict)
# print(submap)
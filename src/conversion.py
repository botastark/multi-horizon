a = 6378137.0
b = 6356752.314245
inv_f = 298.257223563
e_sq = 6.69437999014 * 1e-3
import math
import numpy as np


def radius_n(lat):
    cot = math.cos(lat) / math.sin(lat)
    return a / (1 - e_sq / (1 + cot * cot)) ^ 0.5


def geodetic2ecef(lat, lon, h):
    N = radius_n(lat)
    X = (N + h) * math.cos(lat) * math.cos(lon)
    Y = (N + h) * math.cos(lat) * math.sin(lon)
    Z = ((1 - e_sq) * N + h) * math.sin(lat)
    return (X, Y, Z)


def ecef2ned(p, p_ref, lon_ref, lat_ref):
    sin_lat = math.sin(lat_ref)
    cos_lat = math.cos(lat_ref)

    sin_lon = math.sin(lon_ref)
    cos_lon = math.cos(lon_ref)

    R = np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )
    assert p.shape == p_ref.shape

    return R * (p - p_ref)


# testing
"""
Given:
    Ref point: lon, lat, h(ASML)
    Point p: lon, lat, h(ASML)
Get:
    Point p: x, y, z (NED), wrt Ref point is (0,0,0) NED
"""

ref_point = np.array([])

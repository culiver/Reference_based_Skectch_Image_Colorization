import numpy as np
import thinplate as tps
import cv2
import random
import math

# Reference : https://github.com/cheind/py-thin-plate-spline

def tps_transform(img, num=4, dshape=None):

    while True:
        # point1 = round(random.uniform(0.3, 0.7), 2)
        # point2 = round(random.uniform(0.3, 0.7), 2)
        # range_1 = round(random.uniform(-0.25, 0.25), 2)
        # range_2 = round(random.uniform(-0.25, 0.25), 2)
        points = np.random.uniform(0.1, 0.9, (num, 2))
        ranges = np.random.uniform(-0.05, 0.05, (num, 2))
        dests = points + ranges
        diffs = np.sqrt(np.square(dests[:, None, :2] - dests[None, :, :2]).sum(-1))
        upper_diffs = np.triu(diffs, 1)

        # if math.isclose(point1 + range_1, point2 + range_2):
        if (upper_diffs > 0).sum() < num:
            continue
        else:
            break

    # c_src = np.array([
    #     [0.0, 0.0],
    #     [1., 0],
    #     [1, 1],
    #     [0, 1],
    # ])
    # c_src = np.concatenate([c_src, points], axis=0)
    c_src = points

    # c_dst = np.array([
    #     [0., 0],
    #     [1., 0],
    #     [1, 1],
    #     [0, 1],
    # ])
    # c_dst = np.concatenate([c_dst, dests], axis=0)
    c_dst = dests

    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)


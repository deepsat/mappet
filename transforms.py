import math
import typing

import numpy as np


def warp_augmented(mat: np.array, pt: typing.Tuple[float, float]):
    pt = mat @ [[pt[0]], [pt[1]], [1]]
    return pt[0, 0], pt[1, 0]


def warp_perspective(mat: np.array, pt: typing.Tuple[float, float]):
    pt = mat @ [[pt[0]], [pt[1]], [1]]
    return pt[0, 0] / pt[2, 0], pt[1, 0] / pt[2, 0]


def rotation_matrix(x: float):
    return np.array([
        [math.cos(x), -math.sin(x), 0],
        [math.sin(x), math.cos(x), 0],
        [0, 0, 1]
    ])

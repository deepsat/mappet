import math
import typing

import numpy as np
import cv2


def warp_augmented(mat: np.array, pt: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
    pt = mat @ [[pt[0]], [pt[1]], [1]]
    return pt[0, 0], pt[1, 0]


def warp_perspective(mat: np.array, pt: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
    pt = mat @ [[pt[0]], [pt[1]], [1]]
    return pt[0, 0] / pt[2, 0], pt[1, 0] / pt[2, 0]


def rotation_matrix(x: float) -> np.array:
    return np.array([
        [math.cos(x), -math.sin(x), 0],
        [math.sin(x), math.cos(x), 0],
        [0, 0, 1]
    ])


def fit_homography(src, dst):
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask = mask.reshape(-1).astype(bool)
    src, dst = src[mask], dst[mask]
    base = homography.copy()
    return base, src, dst


def fit_even_similarity(src0, dst0):
    src = np.array([complex(pt[0], pt[1]) for pt in src0])
    dst = np.array([complex(pt[0], pt[1]) for pt in dst0])
    raise NotImplementedError

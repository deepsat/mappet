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


def complex_to_augmented_transform(m: complex, c: complex) -> np.array:
    return np.array([
        [m.real, -m.imag, c.real],
        [m.imag, m.real, c.imag],
        [0, 0, 1]
    ])


def fit_homography_robust(x: np.array, y: np.array, threshold: float = 5.0) -> typing.Tuple[np.array, np.array]:
    homography, mask = cv2.findHomography(x.reshape(-1, 1, 2), y.reshape(-1, 1, 2), cv2.RANSAC, threshold)
    mask = mask.reshape(-1).astype(bool)
    return homography.copy(), mask


def fit_even_similarity_c(x0: np.array, y0: np.array):
    assert x0.shape[0] == y0.shape[0]
    x = x0[:, 0] + 1j * x0[:, 1]
    y = y0[:, 0] + 1j * y0[:, 1]
    x1 = np.column_stack((x, np.ones(x.shape[0])))
    return np.linalg.lstsq(x1, y, rcond=None)[0]


def fit_even_similarity(x0: np.array, y0: np.array):
    return complex_to_augmented_transform(*fit_even_similarity_c(x0, y0))


def fit_even_similarity_robust(x0: np.array, y0: np.array, threshold: float = 5.0,
                               episodes: int = 100, sample_size: int = 3) -> typing.Tuple[np.array, np.array]:
    x = x0[:, 0] + 1j * x0[:, 1]
    y = y0[:, 0] + 1j * y0[:, 1]
    n = x.shape[0]
    assert n == y.shape[0]
    result = (0, np.identity(3), -1, np.array(x.shape[0]))
    for episode in range(episodes):
        sample = np.random.choice(n, sample_size, replace=False)
        s_m, s_c = fit_even_similarity_c(x0[sample], y0[sample])
        s_mask = np.abs((s_m * x + s_c) - y) < threshold
        m, c = fit_even_similarity_c(x0[s_mask], y0[s_mask])
        mask = np.abs((m * x + c) - y) < threshold
        result = max(result, (mask.sum() / n, complex_to_augmented_transform(m, c), mask), key=lambda v: v[0])
    return result[1], result[2]

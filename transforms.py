import logging
import math
import typing

import numpy as np
import cv2

log = logging.getLogger('mappet.transforms')


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


def real2_to_complex(arr: np.array) -> np.array:
    return arr[:, 0] + 1j * arr[:, 1]


def complex_to_augmented_transform(m: complex, c: complex) -> np.array:
    return np.array([
        [m.real, -m.imag, c.real],
        [m.imag, m.real, c.imag],
        [0, 0, 1]
    ])


def fit_homography_robust(x: np.array, y: np.array, threshold: float = 5.0) -> typing.Tuple[np.array, np.array]:
    homography, mask = cv2.findHomography(x.reshape(-1, 1, 2), y.reshape(-1, 1, 2), cv2.RANSAC, threshold)
    mask = mask.reshape(-1).astype(bool)
    log.debug(f"fit_homography_robust: {100 * mask.sum() / x.shape[0]:.2f}% inliers ({mask.sum()}/{x.shape[0]})")
    if mask.sum() / x.shape[0] < 0.4 or x.shape[0] < 100:
        log.warning(f"fit_homography_robust: low-quality fit!")
    return homography.copy(), mask


def fit_even_similarity_c(x0: np.array, y0: np.array):
    x, y = real2_to_complex(x0), real2_to_complex(y0)
    x1 = np.column_stack((x, np.ones(x.shape[0])))
    return np.linalg.lstsq(x1, y, rcond=None)[0]


def fit_even_similarity(x0: np.array, y0: np.array):
    return complex_to_augmented_transform(*fit_even_similarity_c(x0, y0))


def fit_even_similarity_robust(x0: np.array, y0: np.array, threshold: float = 5.0,
                               episodes: int = 100, sample_size: int = 3) -> typing.Tuple[np.array, np.array]:
    x, y = real2_to_complex(x0), real2_to_complex(y0)
    n = x.shape[0]
    result = (0, np.identity(3), -1, np.array(x.shape[0]))
    for episode in range(episodes):
        sample = np.random.choice(n, sample_size, replace=False)
        s_m, s_c = fit_even_similarity_c(x0[sample], y0[sample])
        s_mask = np.abs((s_m * x + s_c) - y) < threshold
        m, c = fit_even_similarity_c(x0[s_mask], y0[s_mask])
        mask = np.abs((m * x + c) - y) < threshold
        result = max(result, (mask.sum() / n, complex_to_augmented_transform(m, c), mask), key=lambda v: v[0])
    _, transform, mask = result
    log.debug(f"fit_even_similarity_robust: {100 * mask.sum() / x.shape[0]:.2f}% inliers ({mask.sum()}/{x.shape[0]})")
    if mask.sum() / x.shape[0] < 0.4 or x.shape[0] < 100:
        log.warning(f"fit_even_similarity_robust: low-quality fit! {mask.sum()}/{x.shape[0]}")
    return transform, mask

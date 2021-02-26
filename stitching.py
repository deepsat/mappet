import typing

import numpy as np
import cv2

T = typing.TypeVar('T')


def blurriness(image: np.array) -> float:
    return cv2.Laplacian(image, cv2.CV_64F).var()


def solve_min_cover_ordering(covers: typing.Iterable[typing.Iterable[T]]) -> typing.List[T]:
    covers = list(covers)
    n = len(covers)
    items = set(sum(covers, []))
    covers = [set(v) for v in covers]
    covered = set()
    result = []
    left = set(range(n))
    while left:
        candy = []
        for i in left:
            candy.append((len(covers[i]), i))
        c, i = max(candy)
        result.append((i, len(covers[i])))
        covered |= covers[i]
        left.discard(i)
        for j in left:
            covers[j] -= covers[i]
    return result


def clip_black_border(image: np.array) -> np.array:
    mask = (image != 0).any(axis=-1).astype(bool)
    x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    while True:
        if not mask[y1, :].any():
            y1 += 1
        else:
            break
    while True:
        if not mask[y2-1, :].any():
            y2 -= 1
        else:
            break

    while True:
        if not mask[:, x1].any():
            x1 += 1
        else:
            break
    while True:
        if not mask[:, x2-1].any():
            x2 -= 1
        else:
            break

    return image[y1:y2, x1:x2]


def last_come(images: typing.Iterable[np.array]) -> np.array:
    it = iter(images)
    out = next(it)
    for curr in images:
        mask = occ_mask(curr)
        out[mask] = curr[mask]
    return out


def occ_weighted_average(images: typing.List[np.array]) -> np.array:
    return weighted_average(images, lambda i: occ_mask(images[i]))


def weighted_average(images: typing.List[np.array], mask: typing.Callable[[int], np.array]) -> np.array:
    out = sum(image.astype(np.float32)/255 * mask(i)[:, :, np.newaxis] for i, image in enumerate(images))
    sum_mask = sum(mask(i) for i in range(len(images))).clip(min=1e-9)
    out /= sum_mask[:, :, np.newaxis]
    out = (out * 255).astype(np.uint8)
    return out


def occ_mask(image: np.array) -> np.array:
    m = (image != 0).any(axis=-1).astype(np.uint8)
    m &= np.pad(m, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    m &= np.pad(m, ((0, 0), (0, 1)), mode='constant')[:, +1:]
    m &= np.pad(m, ((1, 0), (0, 0)), mode='constant')[:-1, :]
    m &= np.pad(m, ((0, 1), (0, 0)), mode='constant')[+1:, :]
    return m.astype(np.bool)


def lay_over(image: np.array, x1: int, y1: int, dest: np.array):
    # Zero pixels are transparent, according to occ_mask rules
    x2, y2 = x1 + image.shape[1], y1 + image.shape[0]
    w, h = dest.shape[1], dest.shape[0]
    xa, xb, ya, yb = max(-x1, 0), max(x2 - w, 0), max(-y1, 0), max(y2 - h, 0)
    x1, x2, y1, y2 = x1 + xa, x2 - xb, y1 + ya, y2 - yb
    image = image[ya:image.shape[0] - yb, xa:image.shape[1] - xb]
    mask = occ_mask(image)
    dest[y1:y2, x1:x2][mask] = image[mask]


def lay_on_canvas(bounds: typing.Tuple[int, int, int, int], warp_gen) -> np.array:
    x_min, x_max, y_min, y_max = bounds
    w, h = x_max - x_min + 1, y_max - y_min + 1
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for (x1, y1), image in warp_gen:
        lay_over(image, x1 - x_min, y1 - y_min, result)
    return result

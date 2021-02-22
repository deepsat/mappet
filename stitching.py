import numpy as np
import cv2


def blurriness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def solve_min_cover_ordering(covers, weights=None):
    n = len(covers)
    if weights is None:
        weights = [1 for _ in range(n)]
    print(weights)
    items = set(sum(covers, []))
    covers = [set(v) for v in covers]
    covered = set()
    result = []
    left = set(range(n))
    while left:
        candy = []
        for i in left:
            candy.append((len(covers[i]) * weights[i], i))
        c, i = max(candy)
        result.append((i, len(covers[i])))
        covered |= covers[i]
        left.discard(i)
        for j in left:
            covers[j] -= covers[i]
    print(result)
    return result


def clip_black_border(image):
    mask = np.any(image != 0, axis=-1).astype(bool)
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


def last_come(images):
    it = iter(images)
    out = next(it)
    for curr in images:
        mask = occ_mask(curr).astype(bool)
        out[mask] = curr[mask]
    return out


def occ_weighted_average(images):
    return weighted_average(images, lambda i: occ_mask(images[i]))


def weighted_average(images, mask):
    out = sum(image.astype(np.float32)/255 * mask(i)[:, :, np.newaxis] for i, image in enumerate(images))
    sum_mask = sum(mask(i) for i in range(len(images))).clip(min=1e-9)
    out /= sum_mask[:, :, np.newaxis]
    out = (out * 255).astype(np.uint8)
    return out


def occ_mask(image):
    m = np.all(image != 0, axis=-1).astype(np.uint8)
    m &= np.pad(m, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    m &= np.pad(m, ((0, 0), (0, 1)), mode='constant')[:, +1:]
    m &= np.pad(m, ((1, 0), (0, 0)), mode='constant')[:-1, :]
    m &= np.pad(m, ((0, 1), (0, 0)), mode='constant')[+1:, :]
    return m

import cv2
import numpy as np
from matplotlib import pyplot as plt

imshow = None

def close_zero(image, p, k=5):
    x, y = int(p[0]), int(p[1])
    x1, x2 = max(0, x - k), min(image.shape[0]-1, x + k)
    y1, y2 = max(0, y - k), min(image.shape[1]-1, y + k)
    return np.all(image[x1:x2,y1:y2] == 0, axis=-1).any()

def homography_for_pair(image1, image2, *, bad_zeroes=True):
    image1, image2 = image1.copy(), image2.copy()
    sift = cv2.SIFT.create()
    key1, desc1 = sift.detectAndCompute(image1, None)
    key2, desc2 = sift.detectAndCompute(image2, None)

    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {'checks': 125})
    matches = matcher.knnMatch(desc1, desc2, k=2)
    matches = [m for m, n in matches if m.distance < 0.8*n.distance]

    if imshow is not None:
        img = cv2.drawMatches(image1, key1, image2, key2, matches, None, matchColor=(0,255,0), singlePointColor=None, flags=2)
        imshow(img)

    src = np.float32([key1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst = np.float32([key2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    print('RANSAC on ', len(src), '-', len(dst))
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    return H


def weighted_average(images, mask):
    out = sum(image.astype(np.float32)/255 * mask(i)[:,:,np.newaxis] for i, image in enumerate(images))
    smask = sum(mask(i) for i in range(len(images))).clip(min=1e-9)
    out /= smask[:,:,np.newaxis]
    out = (out * 255).astype(np.uint8)
    return out


def occ_mask(i, images):
    m = np.all(images[i] != 0, axis=-1).astype(np.uint8)
    m &= np.pad(m, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    m &= np.pad(m, ((0, 0), (0, 1)), mode='constant')[:, +1:]
    m &= np.pad(m, ((1, 0), (0, 0)), mode='constant')[:-1, :]
    m &= np.pad(m, ((0, 1), (0, 0)), mode='constant')[+1:, :]
    return m


def occ_weighted_average(images):
    return weighted_average(images, lambda i: occ_mask(i, images))


def stitch(images):
    images = [image for image in images]
    n = len(images)
    minX, maxX, minY, maxY = 0, images[0].shape[1], 0, images[0].shape[0]
    def shift_matrix():
        return np.array([
            [0, 0, -minX],
            [0, 0, -minY],
            [0, 0, 0]
        ])

    Hs = []
    M = np.identity(3)
    for image1, image2 in zip(images, images[1:]):
        H = homography_for_pair(image2, image1)
        Hs.append(H)
        M = M @ H

        w, h, _ = image2.shape
        print(image1.shape, image2.shape)
        proj_pt = np.float32(((0, 0), (0, w), (h, w), (h, 0))).reshape(-1, 1, 2)
        proj = np.int32(cv2.perspectiveTransform(proj_pt, M))
        x1, x2, y1, y2 = proj[:,:,0].min(), proj[:,:,0].max(), proj[:,:,1].min(), proj[:,:,1].max()
        minX, maxX, minY, maxY = min(x1, x2, minX), max(x1, x2, maxX), min(y1, y2, minY), max(y1, y2, maxY)
    Hs.append(np.identity(3))

    M = np.identity(3)
    warped_images = [None for _ in range(n)]
    for i in range(n):
        warped_images[i] = cv2.warpPerspective(images[i], shift_matrix() + M, (maxX-minX+1, maxY-minY+1))
        M = M @ Hs[i]

    print(warped_images[0].shape)
    return occ_weighted_average(warped_images)


def succ_stitch(images):
    if len(images) > 1:
        curr = stitch((images[0], images[1]))
        if imshow is not None:
            imshow(curr)
        return succ_stitch([curr] + list(images[2:]))
    else:
        return images[0]

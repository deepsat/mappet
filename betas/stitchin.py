import itertools
import functools
import operator
import time
import math
import random
import subprocess
from contextlib import contextmanager

import cv2
import numpy as np
# import torch
import tqdm

import blending
import drone_test_data
import geometry as geo

TILING_EXEC = None # "./geotiling"

def log_stuff(*args, **kwargs):
    # print(*args, **kwargs)
    pass

@contextmanager
def timed(log=None):
    class TimerSpan:
        def __init__(self):
            self.value = None
    obj = TimerSpan()
    start = time.time()
    try:
        yield obj
    finally:
        end = time.time()
        obj.value = end - start
        if log is not None:
            pass
            # print(f"[{log}] {obj.value:.3f}s")

def blur_mask(image, mask, kernel=(3, 3)):
    if mask.shape == image.shape[:-1]:
        mask = np.expand_dims(mask, -1).repeat(3, axis=-1)
    else:
        assert mask.shape == image.shape, f"Incorrect mask shape {mask.shape} for image {image.shape} (nor {image.shape[:-1]})"
    blurred_image = np.full(image.shape, [0, 255, 0], np.uint8)
    # blurred_image = cv2.GaussianBlur(image, kernel, 0)
    return np.where(mask, blurred_image, image)

def blur_segments(image, polys):
    # blurs = np.array([cv2.GaussianBlur(image, (2*(levels-k)+1, 2*(levels-k)+1), 0) for k in range(levels)])
    # blurs = np.array([np.full(image.shape, [0, (1-(k+1)/levels)*255, 0], np.uint8) for k in range(levels)])
    mask = np.zeros(image.shape, np.uint8)
    for poly in polys:
        poly = poly.astype(np.int32)
        cv2.drawContours(mask, poly.reshape(1, -1, 1, 2), -1, (255, 255, 255), 15)
    return blur_mask(image, (mask == 255).all(axis=-1))

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

class Photo:
    key_detect = cv2.ORB_create(nfeatures=2**14)
    # desc_detect = cv2.ORB_create()
    desc_detect = cv2.SIFT_create()

    def __init__(self, image):
        self.image = image.copy()
        self._key, self._desc = None, None

    def _get_features(self):
        if self._key is None or self._desc is None:
            with timed("keypoints"):
                self._key = self.key_detect.detect(self.image, None)
            with timed("descriptors"):
                self._key, self._desc = self.desc_detect.compute(self.image, self._key)
            # self._desc = self._desc.astype(np.float32)

    @property
    def key(self):
        self._get_features()
        return self._key

    @property
    def desc(self):
        self._get_features()
        return self._desc

    def release_features(self):
        self._key, self._desc = None, None

    def draw_image_with_keys(self, color=(0, 255, 0)):
        return cv2.drawKeypoints(self.image, self.key, self.image.copy(), color=color)


def find_keypoint_matches(first, second, *, lowe_coeff=0.8):
    second.desc, first.desc
    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 4}, {'checks': 64})
    with timed("matches"):
       matches = matcher.knnMatch(second.desc, first.desc, k=2)
    matches = [m for m, n in matches if m.distance < lowe_coeff*n.distance]
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # binary descriptors (BRIEF-like)
    # with timed("matches"):
    #     matches = matcher.match(second.desc, first.desc)
    log_stuff("Got", len(matches), "matches")

    src = np.array([second.key[match.queryIdx].pt for match in matches]).reshape((-1, 1, 2))
    dst = np.array([ first.key[match.trainIdx].pt for match in matches]).reshape((-1, 1, 2))
    return src, dst


def prepare_initial_homography(src, dst):
    log_stuff("Homography for", len(src), "pairs")
    with timed("homography"):
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask = mask.reshape(-1).astype(bool)
    src, dst = src[mask], dst[mask]
    log_stuff(" ", len(src), "left")
    return homography, src, dst


class PhotoSeries:
    def __init__(self, images, k=1):
        print("Preparing series")
        self.photos = [Photo(image) for image in images]
        self.homography = []
        print("Homographies")
        def sav(i):
            cv2.imwrite(f'frames/{i}.png', self.photos[i].image)
            cv2.imwrite(f'frames/{i}k.png', self.photos[i].draw_image_with_keys())
        sav(0)
        for i in tqdm.tqdm(range(1, len(self.photos))):
            src, dst = find_keypoint_matches(self.photos[i-1], self.photos[i])
            sav(i)
            M = np.identity(3)
            for j in range(2, k+1):
                if i >= j:
                    M = np.linalg.inv(self.homography[-j+1]) @ M
                    src1, dst1 = find_keypoint_matches(self.photos[i-j], self.photos[i])
                    src = np.concatenate((src, src1))
                    dst = np.concatenate((dst, cv2.perspectiveTransform(dst1, M)))
                else:
                    break
            if i >= k:
                self.photos[i-k].release_features()
            H, s, _ = prepare_initial_homography(src.astype(np.float64), dst.astype(np.float64))
            # print(f"{100*len(s) / len(src):.2f}% | {len(s)} inliers")
            self.homography.append(H)
        for i in range(1, k):
            if len(self.photos) >= i:
                self.photos[-i].release_features()
        self.recalculate_pref_homography()

    def recalculate_pref_homography(self):
        self.pref_homography = [np.identity(3)] + \
            list(itertools.accumulate(self.homography, operator.matmul))

    def homography_between(self, i, j):
        return functools.reduce(operator.matmul, self.homography[i:j])

    def calc_bounds_quads(self):
        q = []
        for i, photo in enumerate(self.photos):
            h, w = photo.image.shape[:2]
            pts = np.array([
                ((0, 0), (w, 0), (w, h), (0, h))
            ], dtype=np.float32).reshape((-1, 1, 2))
            q.append(cv2.perspectiveTransform(pts, self.pref_homography[i]).reshape(-1, 2))
        return np.array(q)

    def calc_bounds(self):
        q = self.calc_bounds_quads()
        return q[:,:,0].min()-2, q[:,:,0].max()+2, q[:,:,1].min()-2, q[:,:,1].max()+2

    def stitch2(self, method='first', blur_edges=True, keys=False):
        y_min, y_max, x_min, x_max = self.calc_bounds()
        print((y_min, y_max), (x_min, x_max))
        shift = np.array([
            [1, 0, -y_min],
            [0, 1, -x_min],
            [0, 0, 1]
        ], dtype=np.float64)
        def warped_images_gen(order=None, *, reverse=False, contour=0):
            it = range(len(self)) if order is None else order
            if not reverse:
                it = reversed(it) # Since we want first-come and get last-come
            it = list(it)
            for i in tqdm.tqdm(it):
                homography, image = self[i]
                matrix = shift @ homography
                if contour:
                    image = np.ones((image.shape[0]+2*contour, image.shape[1]+2*contour, image.shape[2]))
                    image[:, :+2*contour] = [255, 255, 255]
                    image[:, -2*contour:] = [255, 255, 255]
                    image[:+2*contour, :] = [255, 255, 255]
                    image[-2*contour:, :] = [255, 255, 255]
                    matrix = matrix @ [[1, 0, -contour], [0, 1, -contour], [0, 0, 1]]
                elif keys:
                    image = self.photos[i].draw_image_with_keys(color=[random.randint(0, 255) for _ in range(3)])
                yield cv2.warpPerspective(
                    image, matrix, (round(y_max-y_min+1), round(x_max-x_min+1))
                )
        order = None
        def contour_mask(contour):
            ctr = blending.last_come(warped_images_gen(order, contour=contour))
            return (ctr == 255).all(axis=-1)
        print("Initial stitching")
        if method == 'first':
            result = blending.last_come(warped_images_gen())
        elif method == 'avg':
            result = blending.occ_weighted_average(list(warped_images_gen()))
        elif method == 'mincover':
            order = self.photo_min_cover_ordering()
            result = blending.last_come(warped_images_gen(order))
        else:
            raise ValueError("Unknown method")
        if blur_edges:
            assert method == 'first' or method == 'mincover'
            # blurs = [(10, (3, 3)), (4, (5, 5))]
            blurs = [(4, (5, 5))]
            for i, (contour, kernel) in enumerate(blurs):
                print(f"Blur {i+1}/{len(blurs)}: {kernel}@{contour}")
                result = blur_mask(result, contour_mask(contour), kernel)
        return result

    def stitch(self):
        return Photo(self.stitch2())

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, index):
        return self.pref_homography[index], self.photos[index].image

    def photo_min_cover_ordering(self, b=50, thresh=4):
        print("Computing min cover")
        n = len(self.photos)
        polys = [geo.PolygonI(quad) for quad in self.calc_bounds_quads()]
        x_min, x_max, y_min, y_max = self.calc_bounds() # axis swap for clarity
        x_min, y_min = math.floor(x_min - b), math.floor(y_min - b)
        x_max, y_max = math.ceil(x_max + b), math.ceil(y_max + b)
        i_shift, j_shift = (x_min + b - 1)//b, (y_min + b - 1)//b
        i_count, j_count = x_max//b - i_shift, y_max//b - j_shift
        print(f"{x_min=}..{x_max=} {y_min=}..{y_max=}")
        print(f"{i_shift=} {i_count=} {j_shift=} {j_count=}")
        def get_square(i, j):
            v = geo.vec2di((i + i_shift) * b, (j + j_shift) * b)
            return geo.PolygonI((v, v + [0, b], v + [b, b], v + [b, 0]))
        print(" Getting grid")
        grid = [[get_square(i, j) for j in range(j_count)] for i in range(i_count)]
        covers = [[] for _ in range(n)]
        # TODO: consider later with blurry photos
        weights = None
        # weights = [((blurriness(photo.image))/500-1)/10+1 for photo in self.photos]

        print(" Getting covers")
        if TILING_EXEC is None:
            for k, poly in tqdm.tqdm(enumerate(polys), total=n):
                vert = np.array(poly.vertices)
                cx_min, cx_max = vert[:,0].min(), vert[:,0].max()
                cy_min, cy_max = vert[:,1].min(), vert[:,1].max()
                i_min, i_max = cx_min//b - i_shift - 3, cx_max//b - i_shift + 3
                j_min, j_max = cy_min//b - j_shift - 3, cy_max//b - j_shift + 3
                for i in range(max(i_min, 0), min(i_max+1, i_count)):
                    for j in range(max(j_min, 0), min(j_max+1, j_count)):
                        if poly.contains(grid[i][j]):
                            covers[k].append((i, j))
        else:
            string = ""
            output = subprocess.run(
                TILING_EXEC, input=string, capture_output=True, shell=False
            ).stdout.decode()
            for line in output.splitlines():
                k, i, j = [int(x) for x in line.split()]
                covers[k].append((i, j))

        for k in range(n):
            image = np.zeros((y_max-y_min+1, x_max-x_min+1, 3), np.uint8)
            cv2.fillPoly(image, (np.array(polys[k].vertices, dtype=np.int32) - [x_min, y_min]).reshape((1, -1, 2)), (255, 123, 123))
            for i, j in covers[k]:
                cv2.fillPoly(image, (np.array(grid[i][j].vertices, dtype=np.int32) - [x_min, y_min]).reshape((1, -1, 2)), (255, 255, 255))
            cv2.imwrite(f'ghosts/{k}.png', image)

        return [i for i, c in solve_min_cover_ordering(covers, weights) if c >= thresh]

def annotate_good_matches(image, series):
    for i in range(len(series.photos) - 1):
        _, src, dst = prepare_initial_homography(*find_keypoint_matches(series.photos[i], series.photos[i+1]))
        # src, dst = find_keypoint_matches(series.photos[i], series.photos[i+1])
        y_min, y_max, x_min, x_max = series.calc_bounds()
        dst = cv2.perspectiveTransform(dst, series.pref_homography[i])[:,0,:] - [y_min, x_min]
        src = cv2.perspectiveTransform(src, series.pref_homography[i+1])[:,0,:] - [y_min, x_min]
        for a, b in list(zip(src, dst)):
            x1, y1, x2, y2 = round(a[0]), round(a[1]), round(b[0]), round(b[1])
            if math.hypot(x1-x2, y1-y2) < 100:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return image

def main():
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'

    n = 50
    f = -5
    # ind = tuple(range(14700, 14700+n*12, 12))
    # ind = tuple(range(3000, 3000+n*25, 25))
    # ind = [3000+11*50, 3050+12*50+10]
    ind = tuple(range(4525 + f*25, 4525 + f*25 + n*25, 25))
    frames = drone_test_data.frames_at(ind, FILENAME, SUB_FILENAME, silent=False)
    # for frame in frames:
    #      frame.image = frame.image[310:-310,120:-120]

    mtx = np.array([[2.32417540e+04, 0.00000000e+00, 1.13441576e+03],
       [0.00000000e+00, 2.29400020e+04, 6.49951483e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[ 2.86731319e+00, -1.35891191e+03, -1.30516613e-01,
        -4.64144148e-02, -2.77072578e+00]])

    def undistort(img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    images = [undistort(cv2.resize(frame.image, (1920, 1080)).clip(1)) for frame in frames]

    series = PhotoSeries(images)
    image = series.stitch2(method='mincover', blur_edges=True)
    # annotate_good_matches(image, series)
    cv2.imwrite('stitch.png', image)

if __name__ == '__main__':
    main()

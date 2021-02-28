import logging
import functools
import operator
import math
import typing

import numpy as np
import cv2

import stitching
import feature_matching
import transforms
import geometry as geo
from geodesy import LocalTangentPlane
from photo import DronePhoto
from map_photo import MapPhoto
from logging_util import timed_log

log = logging.getLogger('mappet.series')
rel_fit_methods = {
    'homography': transforms.fit_homography_robust,
    'even_similarity': transforms.fit_even_similarity_robust
}

def poly_bounds(points):
    return (
        math.floor(points[:, 0].min()), math.ceil(points[:, 0].max()),
        math.floor(points[:, 1].min()), math.ceil(points[:, 1].max())
    )


class RelativeSeries:
    local_latitude: float
    local_longitude: float
    ref_height: float
    plane: LocalTangentPlane
    photos: typing.List[MapPhoto]
    next_transform: typing.List[np.array]

    def __init__(self, local_latitude: float, local_longitude: float, ref_height: float):
        self.local_latitude, self.local_longitude, self.ref_height = local_latitude, local_longitude, ref_height
        self.plane = LocalTangentPlane(self.local_latitude, self.local_longitude, 0)
        self.photos = []
        self.next_transform = []

    def append(self, photo: DronePhoto, rel_method: str = 'homography'):
        with timed_log(log.info, f'RelativeSeries.append: Photo {len(self.photos)} took {{time:.3f}}s'):
            self.photos.append(MapPhoto.from_drone_photo(
                photo, self.plane, self.ref_height
            ))
            if len(self.photos) >= 2:
                src, dst = feature_matching.find_keypoint_matches(self.photos[-2].image, self.photos[-1].image)
                self.next_transform.append(rel_fit_methods[rel_method](src, dst)[0])

    def transform(self, i: int) -> np.array:
        return functools.reduce(operator.matmul, self.next_transform[:i], np.identity(3))

    def quad(self, i: int, margin: int = 0) -> np.array:
        h, w = self.photos[i].image.shape[:2]
        pts = ((-margin, -margin), (w + margin, -margin), (w + margin, h + margin), (-margin, h + margin))
        return np.array([transforms.warp_perspective(self.transform(i), pt) for pt in pts])

    def warped_center(self, i: int) -> typing.Tuple[float, float]:
        h, w = self.photos[i].image.shape[:2]
        return transforms.warp_perspective(self.transform(i), (w//2, h//2))

    def bounds(self, i: typing.Optional[int] = None, margin: int = 0) -> typing.Tuple[int, int, int, int]:
        if i is None:
            q = np.array([self.quad(i, margin) for i in range(len(self.photos))]).reshape(-1, 2)
        else:
            q = self.quad(i, margin)
        return poly_bounds(q)

    def tile_covering(self, b: int = 128) \
            -> typing.Tuple[typing.List[typing.List[typing.Tuple[int, int]]],
                            typing.Callable[[typing.Tuple[int, int]], np.array]]:
        n = len(self.photos)
        polys = [geo.PolygonI(self.quad(i)) for i in range(n)]
        x_min, x_max, y_min, y_max = self.bounds()
        x_min, y_min = x_min - b, y_min - b
        x_max, y_max = x_max + b, y_max + b
        i_shift, j_shift = (x_min + b - 1)//b, (y_min + b - 1)//b
        i_count, j_count = x_max//b - i_shift, y_max//b - j_shift
        log.debug(f"RelativeSeries.tile_covering: {x_min=}..{x_max=} {y_min=}..{y_max=}")
        log.debug(f"RelativeSeries.tile_covering: {i_shift=} {i_count=} {j_shift=} {j_count=}")

        def get_square(i, j):
            v = np.array(((i + i_shift) * b, (j + j_shift) * b))
            return np.array((v, v + [0, b], v + [b, b], v + [b, 0]))

        import tqdm
        grid = [[get_square(i, j) for j in range(j_count)] for i in range(i_count)]
        covers = [[] for _ in range(n)]
        for k, poly in tqdm.tqdm(enumerate(polys), total=n):
            cx_min, cx_max, cy_min, cy_max = poly_bounds(np.array(poly.vertices))
            i_min, i_max = cx_min // b - i_shift - 3, cx_max // b - i_shift + 3
            j_min, j_max = cy_min // b - j_shift - 3, cy_max // b - j_shift + 3
            c0, c1 = 0, 0
            for i in range(max(i_min, 0), min(i_max + 1, i_count)):
                for j in range(max(j_min, 0), min(j_max + 1, j_count)):
                    if poly.contains(geo.PolygonI(tuple(grid[i][j]))):
                        covers[k].append([i, j])
                        c1 += 1
                    c0 += 1
            log.info(f"RelativeSeries.tile_covering: Photo polygon {k} covers {c1} tiles ({c0} candidates)")

        return covers, get_square

    def warped(self, i: int, image: np.array, margin: int = 0) -> typing.Tuple[typing.Tuple[int, int], np.array]:
        x_min, x_max, y_min, y_max = self.bounds(i, margin)
        shift = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        margin_shift = np.array([
            [1, 0, -margin],
            [0, 1, -margin],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max-x_min+1, y_max-y_min+1
        return (x_min, y_min), cv2.warpPerspective(image, shift @ self.transform(i) @ margin_shift, (w, h))

    def warped_photo(self, i: int) -> typing.Tuple[typing.Tuple[int, int], np.array]:
        return self.warped(i, self.photos[i].image)

    def warped_contour(self, i: int, thickness: int, color: typing.Tuple[int, int, int] = (255, 255, 255),
                       fill_value: int = 0) -> typing.Tuple[typing.Tuple[int, int], np.array]:
        h0, w0, c0 = self.photos[i].image.shape
        ctr = np.full((h0 + 2 * thickness, w0 + 2 * thickness, c0), fill_value, dtype=np.uint8)
        ctr[:, :+2*thickness] = color
        ctr[:, -2*thickness:] = color
        ctr[:+2*thickness, :] = color
        ctr[-2*thickness:, :] = color
        return self.warped(i, ctr, thickness)

    def stitch(self) -> np.array:
        return stitching.lay_on_canvas(self.bounds(), (self.warped_photo(i) for i in range(len(self.photos))))

    def stitch_contours(self, thickness: int, color: typing.Tuple[int, int, int] = (255, 255, 255),
                        fill_in: bool = False, extend_bounds=False) -> np.array:
        result = stitching.lay_on_canvas(
            self.bounds(None, thickness if extend_bounds else 0),
            (self.warped_contour(i, thickness, color, 1 if fill_in else 0) for i in range(len(self.photos)))
        )
        if fill_in:
            result[(result == 1).all(axis=-1)] = (0, 0, 0)
        return result

    def fit_im_to_enu(self) -> typing.Optional[typing.Tuple[complex, complex]]:
        if len(self.photos) < 2:
            return None
        xp = np.array([self.warped_center(i) for i in range(len(self.photos))])
        yp = np.array([(photo.metadata.x, photo.metadata.y) for photo in self.photos])
        m, c = transforms.fit_even_similarity_c(xp, yp)
        x, y = transforms.real2_to_complex(xp), transforms.real2_to_complex(yp)
        err = np.abs((m * x + c) - y)
        log.info(f"RelativeSeries.fit_im_to_enu: {err.mean()=}, {err.max()=}, {np.median(err)=}, {np.std(err)=}")
        return m, c

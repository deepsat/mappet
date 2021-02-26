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
from geodesy import LocalTangentPlane
from photo import DronePhoto
from map_photo import MapPhoto
from logging_util import timed_log

log = logging.getLogger('mappet.series')
rel_fit_methods = {
    'homography': transforms.fit_homography_robust,
    'even_similarity': transforms.fit_even_similarity_robust
}


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
        with timed_log(log.info, f'Appending photo {len(self.photos)} to RelativeSeries took {{time:.3f}}s'):
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
        return (
            math.floor(q[:, 0].min()), math.ceil(q[:, 0].max()),
            math.floor(q[:, 1].min()), math.ceil(q[:, 1].max())
        )

    def warped(self, i: int) -> typing.Tuple[typing.Tuple[int, int], np.array]:
        x_min, x_max, y_min, y_max = self.bounds(i)
        shift = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max - x_min + 1, y_max - y_min + 1
        return (x_min, y_min), cv2.warpPerspective(self.photos[i].image, shift @ self.transform(i), (w, h))

    def warped_contour(self, i: int, thickness: int, color: typing.Tuple[int, int, int] = (255, 255, 255),
                       fill_value: int = 0) -> typing.Tuple[typing.Tuple[int, int], np.array]:
        x_min0, x_max0, y_min0, y_max0 = self.bounds(i)
        x_min, x_max, y_min, y_max = self.bounds(i, thickness)
        shift = np.array([
            [1, 0, -x_min0],
            [0, 1, -y_min0],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max-x_min+1, y_max-y_min+1
        h0, w0, c0 = self.photos[i].image.shape
        ctr = np.full((h0 + 2 * thickness, w0 + 2 * thickness, c0), fill_value, dtype=np.uint8)
        ctr[:, :+2*thickness] = color
        ctr[:, -2*thickness:] = color
        ctr[:+2*thickness, :] = color
        ctr[-2*thickness:, :] = color
        return (x_min, y_min), cv2.warpPerspective(ctr, shift @ self.transform(i), (w, h))

    def stitch(self) -> np.array:
        return stitching.lay_on_canvas(self.bounds(), (self.warped(i) for i in range(len(self.photos))))

    def stitch_contours(self, thickness: int, color: typing.Tuple[int, int, int] = (255, 255, 255),
                        fill_in: bool = False, extend_bounds=False) -> np.array:
        result = stitching.lay_on_canvas(
            self.bounds(None, thickness if extend_bounds else 0),
            (self.warped_contour(i, thickness, color, 1 if fill_in else 0) for i in range(len(self.photos)))
        )
        if fill_in:
            result[(result == 1).all(axis=-1)] = (0, 0, 0)
        return result

    def fit_im_to_enu(self) -> typing.Optional[np.array]:
        if len(self.photos) < 2:
            return None
        xp = np.array([self.warped_center(i) for i in range(len(self.photos))])
        yp = np.array([(photo.metadata.x, photo.metadata.y) for photo in self.photos])
        return transforms.fit_even_similarity_c(xp, yp)

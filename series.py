import functools
import operator
import math
import typing

import numpy as np
import cv2

from transforms import warp_perspective
from geodesy import LocalTangentPlane
from stitching import occ_mask
from photo import DronePhoto
from map_photo import MapPhoto
from feature_matching import find_keypoint_matches, compute_homography, compute_even_similarity

class RelativeSeries:
    local_latitude: float
    local_longitude: float
    ref_height: float
    plane: LocalTangentPlane
    photos: typing.List[MapPhoto]
    next_transform: typing.List[np.array]

    def __init__(self, local_latitude, local_longitude, ref_height):
        self.local_latitude, self.local_longitude, self.ref_height = local_latitude, local_longitude, ref_height
        self.plane = LocalTangentPlane(self.local_latitude, self.local_latitude, 0)
        self.photos = []
        self.next_transform = []

    def append(self, photo: DronePhoto, rel_method: str = 'homography'):
        self.photos.append(MapPhoto.from_drone_photo(
            photo, self.plane, self.ref_height
        ))
        if len(self.photos) >= 2:
            src, dst = find_keypoint_matches(self.photos[-2].image, self.photos[-1].image)
            if rel_method == 'homography':
                def get(): return compute_homography(src, dst)[0]
            elif rel_method == 'even_similarity':
                def get(): return compute_even_similarity(src, dst)[0]
            else:
                raise ValueError("No such relativity method")
            self.next_transform.append(get())

    def transform(self, i):
        return functools.reduce(operator.matmul, self.next_transform[:i], np.identity(3))

    def quad(self, i):
        h, w = self.photos[i].image.shape[:2]
        pts = ((0, 0), (w, 0), (w, h), (0, h))
        return np.array([warp_perspective(self.transform(i), pt) for pt in pts])

    def bounds(self, i=None):
        q = np.array([self.quad(i) for i in range(len(self.photos))]).reshape(-1, 2) if i is None else self.quad(i)
        return (
            math.floor(q[:, 0].min()), math.ceil(q[:, 0].max()),
            math.floor(q[:, 1].min()), math.ceil(q[:, 1].max())
        )

    def warped(self, i):
        x_min, x_max, y_min, y_max = self.bounds(i)
        shift = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max-x_min+1, y_max-y_min+1
        return (x_min, y_min, w, h), cv2.warpPerspective(self.photos[i].image, shift @ self.transform(i), (w, h))

    def stitch(self):
        x_min, x_max, y_min, y_max = self.bounds()
        result = np.zeros((round(y_max-y_min+1), round(x_max-x_min+1), 3), dtype=np.uint8)
        for i in range(len(self.photos)):
            (x, y, w, h), image = self.warped(i)
            cv2.imwrite(f'test/warp{i}.png', image)
            x -= x_min
            y -= y_min
            mask = occ_mask(image).astype(bool)
            result[y:y+h, x:x+w][mask] = image[mask]
        return result


if __name__ == '__main__':
    import tqdm
    from drone_test_data import get_drone_photos
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'

    n = 8
    photos = get_drone_photos([4525, 4625, 4700, 4800, 4850, 4900, 4950, 5000][:n], FILENAME, SUB_FILENAME, silent=False)
    lat, lng, lh = photos[0].metadata.latitude, photos[0].metadata.longitude, photos[0].metadata.height

    series = RelativeSeries(lat, lng, lh)
    for p in tqdm.tqdm(photos):
        series.append(p)

    cv2.imwrite('test/stitch.png', series.stitch())

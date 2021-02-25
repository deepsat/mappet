import functools
import operator
import math
import typing

import numpy as np
import cv2

from transforms import warp_perspective, fit_homography, fit_even_similarity
from geodesy import LocalTangentPlane
from stitching import occ_mask
from photo import DronePhoto
from map_photo import MapPhoto
from feature_matching import find_keypoint_matches


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
        self.photos.append(MapPhoto.from_drone_photo(
            photo, self.plane, self.ref_height
        ))
        if len(self.photos) >= 2:
            src, dst = find_keypoint_matches(self.photos[-2].image, self.photos[-1].image)
            if rel_method == 'homography':
                def get(): return fit_homography(src, dst)[0]
            elif rel_method == 'even_similarity':
                def get(): return fit_even_similarity(src, dst)[0]
            else:
                raise ValueError("No such relativity method")
            self.next_transform.append(get())

    def transform(self, i: int) -> np.array:
        return functools.reduce(operator.matmul, self.next_transform[:i], np.identity(3))

    def quad(self, i: int, margin: int = 0) -> np.array:
        h, w = self.photos[i].image.shape[:2]
        pts = ((-margin, -margin), (w + margin, -margin), (w + margin, h + margin), (-margin, h + margin))
        return np.array([warp_perspective(self.transform(i), pt) for pt in pts])

    def warped_center(self, i: int) -> typing.Tuple[float, float]:
        h, w = self.photos[i].image.shape[:2]
        return warp_perspective(self.transform(i), (w//2, h//2))

    def bounds(self, i: typing.Optional[int] = None, margin: int = 0) -> typing.Tuple[int, int, int, int]:
        if i is None:
            q = np.array([self.quad(i, margin) for i in range(len(self.photos))]).reshape(-1, 2)
        else:
            q = self.quad(i, margin)
        return (
            math.floor(q[:, 0].min()), math.ceil(q[:, 0].max()),
            math.floor(q[:, 1].min()), math.ceil(q[:, 1].max())
        )

    def warped(self, i: int) -> typing.Tuple[typing.Tuple[int, int, int, int], np.array]:
        x_min, x_max, y_min, y_max = self.bounds(i)
        shift = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max-x_min+1, y_max-y_min+1
        return (x_min, y_min, w, h), cv2.warpPerspective(self.photos[i].image, shift @ self.transform(i), (w, h))

    def warped_contour(self, i: int, thickness: int):
        x_min, x_max, y_min, y_max = self.bounds(i, thickness)
        shift = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        w, h = x_max-x_min+1, y_max-y_min+1
        h0, w0, c0 = self.photos[i].image.shape
        ctr = np.zeros((h0 + 2 * thickness, w0 + 2 * thickness, c0))
        ctr[:, :+2*thickness] = [255, 255, 255]
        ctr[:, -2*thickness:] = [255, 255, 255]
        ctr[:+2*thickness, :] = [255, 255, 255]
        ctr[-2*thickness:, :] = [255, 255, 255]
        return (x_min, y_min, w, h), cv2.warpPerspective(ctr, shift @ self.transform(i), (w, h))

    def stitch(self) -> np.array:
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

    def fit_im_to_enu(self) -> np.array:
        x = np.array([complex(*self.warped_center(i)) for i in range(len(self.photos))])
        x1 = np.column_stack((x, np.ones(len(self.photos))))
        y = np.array([complex(photo.metadata.x, photo.metadata.y) for photo in self.photos])
        m, c = np.linalg.lstsq(x1, y, rcond=None)[0]
        return np.array([
            m.real, -m.imag, c.real,
            m.imag,  m.real, c.imag,
            0, 0, 1
        ])


if __name__ == '__main__':
    import tqdm
    from drone_test_data import get_drone_photos
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'

    n = 25
    photos = get_drone_photos(([4525, 4625, 4700, 4800, 4850, 4900, 4950, 5000] + list(range(5000, 7000, 25)))[:n], FILENAME, SUB_FILENAME, silent=False)
    lat, lng, lh = photos[0].metadata.latitude, photos[0].metadata.longitude, photos[0].metadata.height

    series = RelativeSeries(lat, lng, lh)
    for ph in tqdm.tqdm(photos):
        series.append(ph)

    cv2.imwrite('test/stitch.png', series.stitch())
    im_to_enu = series.fit_im_to_enu()

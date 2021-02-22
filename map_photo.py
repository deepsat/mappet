import math
import typing

import numpy as np
import cv2

import geodesy
from photo import DronePhoto, DroneCameraMetadata
import feature_matching

OFloat = typing.Optional[float]


class MapPhotoMetadata:
    x: OFloat
    y: OFloat
    z: OFloat
    roll: OFloat
    pitch: OFloat
    yaw: OFloat

    def __init__(self, x: OFloat, y: OFloat, z: OFloat, roll: OFloat, pitch: OFloat, yaw: OFloat):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def __repr__(self):
        return f"DroneCameraMetadata(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}" \
               + f", roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"


class MapPhoto:
    image: np.array
    metadata: MapPhotoMetadata
    used_metadata: typing.Set[str]
    base_transform: np.array

    def __init__(self, image: np.array, metadata: MapPhotoMetadata,
                 used_metadata: typing.Iterable[str] = ('xy', 'z', 'roll', 'pitch', 'yaw'),
                 base_transform: typing.Optional[np.array] = None):
        self.image, self.metadata = image, metadata
        self.used_metadata = set(used_metadata)
        self.base_transform = np.identity(3) if base_transform is None else base_transform

    @classmethod
    def from_drone_photo(cls, photo: DronePhoto, local_plane: geodesy.LocalTangentPlane):
        x, y, z = local_plane.enu(photo.metadata.latitude, photo.metadata.longitude, photo.metadata.height)
        metadata = MapPhotoMetadata(
            x, y, z, photo.metadata.roll, photo.metadata.pitch, photo.metadata.yaw
        )
        return cls(photo.image, metadata)

    @property
    def transform(self):
        def rot_ok(x):
            return x is not None and abs(x) > 1e-3
        transform = self.base_transform

        if 'xy' in self.used_metadata:
            transform = np.array([
                [1, 0, self.metadata.x],
                [0, 1, self.metadata.y],
                [0, 0, 1]
            ]) @ transform
        if 'z' in self.used_metadata:
            z = self.metadata.z
            transform = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1/z]
            ]) @ transform
        if 'roll' in self.used_metadata and rot_ok(self.metadata.roll):
            raise NotImplementedError
        if 'pitch' in self.used_metadata and rot_ok(self.metadata.pitch):
            raise NotImplementedError
        if 'yaw' in self.used_metadata and rot_ok(self.metadata.yaw):
            a = self.metadata.yaw
            transform = np.array([
                [math.cos(a), -math.sin(a), 0],
                [math.sin(a), math.cos(a), 0],
                [0, 0, 1]
            ]) @ transform

        return transform

    def homography_enhancement(self, prev):
        # self.used_metadata -= {'roll', 'pitch', 'yaw'}
        self.used_metadata.clear()
        src, dst = feature_matching.find_keypoint_matches(prev.image, self.image)
        data, src1, dst1 = feature_matching.compute_homography(src, dst)
        self.metadata.x = self.metadata.y = self.metadata.z = None
        self.metadata.roll = self.metadata.pitch = self.metadata.yaw = None
        self.base_transform = data['base'] @ prev.transform

    def even_similarity_enhancement(self, prev):
        src, dst = feature_matching.find_keypoint_matches(prev.image, self.image)
        data, src1, dst1 = feature_matching.compute_even_similarity(src, dst)
        raise NotImplementedError

    @property
    def bounds_quad(self):
        h, w = self.image.shape[:2]
        pts = np.array([
            ((0, 0), (w, 0), (w, h), (0, h))
        ], dtype=np.float32).reshape((-1, 1, 2))
        return cv2.perspectiveTransform(pts, self.transform).reshape(-1, 2)

    @property
    def bounds(self):
        q = self.bounds_quad
        return q[:, 0].min()-1, q[:, 0].max()+1, q[:, 1].min()-1, q[:, 1].max()+1

    def warp(self, h=100):
        y_min, y_max, x_min, x_max = [x/h for x in self.bounds]
        print(f"{y_min}..{y_max} {x_min}..{x_max}")
        shift = np.array([
            [1, 0, -y_min],
            [0, 1, -x_min],
            [0, 0, h]
        ])
        return cv2.warpPerspective(self.image, shift @ self.transform, (round(y_max-y_min)+1, round(x_max-x_min)+1))

    def __repr__(self):
        return f"MapPhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata}, " + \
               f"used_metadata={self.used_metadata}, base_transform=[ ... ])"


if __name__ == '__main__':
    from drone_test_data import frames_at
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'
    frames = frames_at([4525, 4550, 4600], FILENAME, SUB_FILENAME, silent=False)
    for frame in frames:
        frame.image = cv2.resize(frame.image, (1920//2, 1080//2))

    def rads(i):
        return (geodesy.math.radians(x) for x in frames[i].position[:2][::-1])

    lat0, lng0 = rads(0)
    print(frames[0].position[:2][::-1], (lat0, lng0))
    plane = geodesy.LocalTangentPlane(lat0, lng0, 0)
    photos = [
        MapPhoto.from_drone_photo(DronePhoto(
            frame.image, DroneCameraMetadata(*rads(i), frames[0].altitude, 0, 0, None, None)
        ), plane) for i, frame in enumerate(frames)
    ]
    print(photos)
    photos[0].metadata.yaw = 0
    photos[1].homography_enhancement(photos[0])
    # photos[2].homography_enhancement(photos[1])
    for k in range(2):
        print(photos[k].transform)
        cv2.imwrite(f'test/im{k}.png', photos[k].warp())
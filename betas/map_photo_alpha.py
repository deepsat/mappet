import math
import cmath
import typing

import numpy as np
import cv2

import geodesy
from transforms import warp_perspective, rotation_matrix
from photo import DronePhoto, DroneCameraMetadata
from camera import DroneCamera
import feature_matching

OFloat = typing.Optional[float]


class MapPhotoMetadata:
    x: OFloat
    y: OFloat
    z: OFloat
    roll: OFloat
    pitch: OFloat
    yaw: OFloat
    camera: typing.Optional[DroneCamera]

    def __init__(self, x: OFloat, y: OFloat, z: OFloat, roll: OFloat, pitch: OFloat, yaw: OFloat,
                 camera: typing.Optional[DroneCamera] = None):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.camera = camera

    def __repr__(self):
        return f"DroneCameraMetadata(x={self.x}, y={self.y}, z={self.z}" \
               + f", roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}, camera={self.camera})"

    def get_im_to_enu(self, shape):
        w, h = shape
        if any(v is None for v in (self.x, self.y, self.z, self.yaw, self.camera)):
            return None
        if self.camera.view_angles is None:
            return None
        raise NotImplementedError


class MapPhoto:
    image: np.array
    metadata: MapPhotoMetadata
    used_metadata: typing.Set[str]
    z0: float
    base_transform: np.array
    im_to_enu: np.array

    def __init__(self, image: np.array, metadata: MapPhotoMetadata, z0: float = 100,
                 used_metadata: typing.Iterable[str] = ('xy', 'z', 'roll', 'pitch', 'yaw'),
                 base_transform: typing.Optional[np.array] = None, im_to_enu: typing.Optional[np.array] = None):
        self.image, self.metadata = image, metadata
        self.used_metadata = set(used_metadata)
        self.z0 = z0
        self.base_transform = np.identity(3) if base_transform is None else base_transform
        self.im_to_enu = self.metadata.get_im_to_enu(self.image.shape[:2][::-1]) if im_to_enu is None else im_to_enu

    def __repr__(self):
        return f"MapPhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata}, " + \
               f"z0={self.z0}, used_metadata={self.used_metadata}, base_transform=[ ... ], im_to_enu=[ ... ])"

    @classmethod
    def from_drone_photo(cls, photo: DronePhoto, local_plane: geodesy.LocalTangentPlane, base_height: float = 100):
        x, y, z = local_plane.enu(photo.metadata.latitude, photo.metadata.longitude, photo.metadata.height)
        metadata = MapPhotoMetadata(x, y, z, photo.metadata.roll, photo.metadata.pitch, photo.metadata.yaw)
        return cls(photo.image, metadata, base_height)

    @property
    def transform(self):
        def rot_ok(x):
            return x is not None and abs(x) > 1e-3
        transform = self.base_transform

        if 'z' in self.used_metadata:
            transform = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, self.z0 / self.metadata.z]
            ]) @ transform
        if 'roll' in self.used_metadata and rot_ok(self.metadata.roll):
            raise NotImplementedError
        if 'pitch' in self.used_metadata and rot_ok(self.metadata.pitch):
            raise NotImplementedError
        if 'yaw' in self.used_metadata and rot_ok(self.metadata.yaw):
            transform = rotation_matrix(self.metadata.yaw) @ transform

        return transform

    def homography_relative(self, prev):
        self.used_metadata -= {'z', 'roll', 'pitch', 'yaw'}
        src, dst = feature_matching.find_keypoint_matches(prev.image, self.image)
        transform, src1, dst1 = feature_matching.compute_homography(src, dst)
        transform = prev.transform @ transform
        # self.metadata.x = self.metadata.y = self.metadata.z = None
        if self.im_to_enu is not None:
            x, y = warp_perspective(self.im_to_enu @ transform, (self.image.shape[1]//2, self.image.shape[0]//2))
            print(self.metadata.x, '->', x, ' ', self.metadata.y, '->', y)
        else:
            self.used_metadata -= {'xy'}
        self.metadata.z = self.metadata.roll = self.metadata.pitch = self.metadata.yaw = None
        self.base_transform = transform

    def even_similarity_relative(self, prev):
        src, dst = feature_matching.find_keypoint_matches(prev.image, self.image)
        (a, b), src1, dst1 = feature_matching.compute_even_similarity(src, dst)
        transform = np.identity(3)
        p = cmath.phase(a)
        if prev.metadata.yaw is not None:
            self.metadata.yaw = (prev.metadata.yaw + cmath.phase(a)) % math.tau
        else:
            self.used_metadata -= {'yaw'}
            transform = rotation_matrix(cmath.phase(a)) @ transform
        if self.im_to_enu is not None:
            x, y = warp_perspective(self.im_to_enu @ transform, (self.image.shape[1]//2, self.image.shape[0]//2))

        self.base_transform = transform

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

    def warp(self):
        y_min, y_max, x_min, x_max = self.bounds
        shift = np.array([
            [1, 0, -y_min],
            [0, 1, -x_min],
            [0, 0, 1]
        ])
        return cv2.warpPerspective(self.image, shift @ self.transform, (round(y_max-y_min)+1, round(x_max-x_min)+1))


if __name__ == '__main__':
    from drone_test_data import frames_at
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'
    n = 3
    frames = frames_at([4525, 4625, 4700, 4800, 4900][:n], FILENAME, SUB_FILENAME, silent=False)
    for frame in frames:
        frame.image = cv2.resize(frame.image, (960, 540))
    frames[2].image = cv2.resize(frames[2].image[120:-120, 75:-75], (960, 540))

    def rads(i):
        return (geodesy.math.radians(x) for x in frames[i].position[:2][::-1])

    lat0, lng0 = rads(0)
    print(frames[0].position[:2][::-1], (lat0, lng0))
    plane = geodesy.LocalTangentPlane(lat0, lng0, 0)
    photos = [
        MapPhoto.from_drone_photo(DronePhoto(
            frame.image, DroneCameraMetadata(*rads(i), frames[0].altitude, 0, 0, None, None)
        ), plane, 130) for i, frame in enumerate(frames)
    ]
    print(photos)
    photos[0].metadata.yaw = 0

    for k in range(1, n):
        photos[k].homography_relative(photos[k-1])

    print(photos)
    for k in range(n):
        print(photos[k].transform)
        cv2.imwrite(f'test/im{k}.png', photos[k].image)
        cv2.imwrite(f'test/imw{k}.png', photos[k].warp())

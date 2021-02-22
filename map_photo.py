import typing

import numpy as np
import cv2

import geodesy
from photo import DronePhoto, DroneCameraMetadata

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

    def __init__(self, image, metadata, used_metadata=('xy', 'z', 'roll', 'pitch', 'yaw'), base_transform=None):
        self.image, self.metadata = image, metadata
        self.used_metadata = set(used_metadata)
        self.base_transform = np.identity(3) if base_transform is None else base_transform

    @classmethod
    def from_drone_photo(cls, photo: DronePhoto, local_plane: geodesy.LocalTangentPlane ):
        x, y, z = local_plane.enu(photo.metadata.latitude, photo.metadata.longitude, photo.metadata.height)
        metadata = MapPhotoMetadata(
            x, y, z, photo.metadata.roll, photo.metadata.pitch, photo.metadata.yaw
        )
        return cls(photo.image, metadata)

    def get_transform(self):
        transform = self.base_transform

        if 'xy' in self.used_metadata:
            raise NotImplementedError
        if 'z' in self.used_metadata:
            raise NotImplementedError
        if 'roll' in self.used_metadata:
            raise NotImplementedError
        if 'pitch' in self.used_metadata:
            raise NotImplementedError
        if 'yaw' in self.used_metadata:
            raise NotImplementedError

        return transform

    def homography_enhancement(self, previous_photo):
        self.used_metadata -= {'roll', 'pitch', 'yaw'}
        raise NotImplementedError

    def positive_similarity_enhancement(self, previous_photo):
        raise NotImplementedError

    def __repr__(self):
        return f"MapPhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata}, " + \
               f"used_metadata={self.used_metadata}, base_transform=[ ... ])"


if __name__ == '__main__':
    from drone_test_data import frames_at
    FILENAME = '/run/media/kubin/Common/deepsat/drone4.MP4'
    SUB_FILENAME = '/run/media/kubin/Common/deepsat/drone4.SRT'
    frames = frames_at([4525, 4550, 4600], FILENAME, SUB_FILENAME, silent=False)

    def rads(i):
        return (geodesy.math.radians(x) for x in frames[i].position[:2][::-1])

    lat0, lng0 = rads(0)
    print(frames[0].position[:2][::-1], lat0, lng0)
    plane = geodesy.LocalTangentPlane(lat0, lng0, 0)
    photos = [
        MapPhoto.from_drone_photo(DronePhoto(
            frame.image, DroneCameraMetadata(*rads(i), frames[0].altitude, 0, 0, None, None)
        ), plane) for i, frame in enumerate(frames)
    ]
    print(photos)

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
    """
    Metadata of a photo that is place-able on a map, acquired from measurements or TODO: inferred from other sources.
    Some of the metadata may be missing or be insufficient to place the photo precisely.
    """
    e: OFloat
    n: OFloat
    u: OFloat
    roll: OFloat
    pitch: OFloat
    yaw: OFloat
    camera: typing.Optional[DroneCamera]

    def __init__(self, n: OFloat, e: OFloat, u: OFloat, roll: OFloat, pitch: OFloat, yaw: OFloat,
                 camera: typing.Optional[DroneCamera] = None):
        self.e, self.n, self.u = e, n, u
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.camera = camera

    def __repr__(self) -> str:
        return f"DroneCameraMetadata(x={self.x}, y={self.y}, z={self.z}" \
               + f", roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}, camera={self.camera})"

    def get_im_to_enu(self, shape: typing.Tuple[int, int]) -> typing.Optional[typing.Tuple[float, float]]:
        w, h = shape
        if any(v is None for v in (self.x, self.y, self.z, self.yaw, self.camera)):
            return None
        if self.camera.view_angles is None:
            return None
        raise NotImplementedError


class MapPhoto:
    """
    Photo with data necessary to place it on the map. TODO: absolutize existing constructs (replace RelativeSeries)
    """
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
               f"z0={self.z0}, used_metadata={self.used_metadata}, base_transform=[ ... ], " + \
               f"im_to_enu={'[ ... ]' if self.im_to_enu is not None else 'None'})"

    @classmethod
    def from_drone_photo(cls, photo: DronePhoto, local_plane: geodesy.LocalTangentPlane, base_height: float = 100):
        if not any(v is None for v in (photo.metadata.latitude, photo.metadata.longitude, photo.metadata.height)):
            e, n, u = local_plane.enu(photo.metadata.latitude, photo.metadata.longitude, photo.metadata.height)
        else:
            e = n = u = None
        metadata = MapPhotoMetadata(e, n, u, photo.metadata.roll, photo.metadata.pitch, photo.metadata.yaw)
        return cls(photo.image, metadata, base_height)

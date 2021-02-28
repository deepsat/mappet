import datetime
import typing

import numpy as np

from camera import DroneCamera

OFloat = typing.Optional[float]


class DroneCameraMetadata:
    """
    Full metadata of an aerial photo.
    """
    latitude: OFloat
    longitude: OFloat
    height: OFloat
    roll: OFloat
    pitch: OFloat
    yaw: OFloat
    curr_time: datetime.datetime
    camera: DroneCamera

    def __init__(self, latitude: OFloat, longitude: OFloat, height: OFloat,
                 roll: OFloat, pitch: OFloat, yaw: OFloat, curr_time: datetime.datetime,
                 camera: typing.Optional[DroneCamera]):
        self.latitude, self.longitude, self.height = latitude, longitude, height
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.curr_time, self.camera = curr_time, camera

    def __repr__(self) -> str:
        return f"DroneCameraMetadata(latitude={self.latitude}, longitude={self.longitude}, height={self.height}, " + \
               f"roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}, curr_time={self.curr_time}, " + \
               f"camera={self.camera})"


class DronePhoto:
    """
    Aggregate type for aerial photo data.
    """
    image: np.array
    metadata: DroneCameraMetadata

    def __init__(self, image: np.array, metadata: DroneCameraMetadata):
        self.image, self.metadata = image, metadata

    def __repr__(self) -> str:
        return f"DronePhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata})"

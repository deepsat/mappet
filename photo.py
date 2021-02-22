import datetime
import typing

import numpy as np

OFloat = typing.Optional[float]


class DroneCameraMetadata:
    latitude: OFloat
    longitude: OFloat
    height: OFloat
    roll: OFloat
    pitch: OFloat
    yaw: OFloat
    time: datetime.datetime

    def __init__(self, latitude: OFloat, longitude: OFloat, height: OFloat,
                 roll: OFloat, pitch: OFloat, yaw: OFloat, curr_time: datetime.datetime):
        self.latitude, self.longitude, self.height = latitude, longitude, height
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.curr_time = curr_time

    def __repr__(self):
        return f"DroneCameraMetadata(latitude={self.latitude}, longitude={self.longitude}, height={self.height}" \
                    + f", roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}, curr_time={self.curr_time})"


class DronePhoto:
    image: np.array
    metadata: DroneCameraMetadata

    def __init__(self, image: np.array, metadata: DroneCameraMetadata):
        self.image, self.metadata = image, metadata

    def __repr__(self):
        return f"DronePhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata})"

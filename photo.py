import datetime

import numpy as np


class DroneCameraMetadata:
    latitude: float
    longitude: float
    elevation: float
    roll: float
    pitch: float
    yaw: float
    time: datetime.datetime

    def __init__(self, latitude: float, longitude: float, elevation: float,
                 roll: float, pitch: float, yaw: float, curr_time: datetime.datetime):
        self.latitude, self.longitude, self.elevation = latitude, longitude, elevation
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.curr_time = curr_time

    def __repr__(self):
        return f"DroneCameraMetadata(latitude={self.latitude}, longitude={self.longitude}, elevation={self.elevation}" \
                    + f", roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}, curr_time={self.curr_time})"


class DronePhoto:
    image: np.array
    metadata: DroneCameraMetadata

    def __init__(self, image: np.array, metadata: DroneCameraMetadata):
        self.image, self.metadata = image, metadata

    def __repr__(self):
        return f"DronePhoto(image=<{' x '.join(map(str, self.image.shape))}>, metadata={self.metadata})"

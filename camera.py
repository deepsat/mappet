import typing

import numpy as np
import cv2


class DroneCamera:
    camera_calibration: typing.Tuple[np.array, np.array]
    view_angles: typing.Tuple[float, float]

    def __init__(self, calibration: typing.Tuple[np.array, np.array], view_angles: typing.Tuple[float, float]):
        self.calibration = calibration
        self.view_angles = view_angles

    def __repr__(self) -> str:
        return f"DroneCamera(calibration={'[ ... ]' if self.calibration is not None else 'None'}, view_angles={self.view_angles})"

    def undistort(self, image: np.array) -> np.array:
        mtx, dist = self.camera_calibration
        h, w = image.shape[:2]
        mtx1, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(image, mtx, dist, None, mtx1)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

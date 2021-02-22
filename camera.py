import typing

import numpy as np
import cv2


class DroneCamera:
    camera_calibration: typing.Tuple[np.array, np.array]

    def __init__(self, camera_calibration):
        self.camera_calibration = camera_calibration

    def undistort(self, image):
        mtx, dist = self.camera_calibration
        h, w = image.shape[:2]
        mtx1, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(image, mtx, dist, None, mtx1)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

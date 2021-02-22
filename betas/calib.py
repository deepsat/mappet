import glob

import numpy as np
import cv2
import tqdm

from drone_records import frames_at

SIZE = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, SIZE[0] * SIZE[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# images = [frame.image for frame in frames_at(range(54, 1750, 15), '/run/media/kubin/Common/deepsat/calib.MP4', silent=False)]
images = [cv2.resize(cv2.imread(name), (1920, 1080)) for name in tqdm.tqdm(glob.glob('/run/media/kubin/Common/deepsat/calib-photos/*.*'))]
print(len(images), "frames loaded")
for img in tqdm.tqdm(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, SIZE)
    if ret:
        # print("pass")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, SIZE, corners2, ret)

h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix :")
print(repr(mtx))
print("dist :")
print(repr(dist))
print("rvecs :")
# print(repr(rvecs))
print("tvecs :")
# print(repr(tvecs))


def undistort(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

cv2.imwrite("calib-test0.png", images[2])
cv2.imwrite("calib-test1.png", undistort(images[2]))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


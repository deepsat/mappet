import glob

import numpy as np
import cv2
import tqdm

SIZE = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, SIZE[0] * SIZE[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = [cv2.imread(name) for name in tqdm.tqdm(sorted(glob.glob('/run/media/kubin/Common/deepsat/calib-photos-pi/*.*')))]
print(len(images), "frames loaded")
for img in tqdm.tqdm(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, SIZE)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # cv2.imshow('corners', cv2.drawChessboardCorners(img, SIZE, corners2, ret))
        # cv2.waitKey(0)

h, w = img.shape[:2]

print("engage calibration... ", end="", flush=True)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("done")

print("mtx:")
print(repr(mtx))
print("dist:")
print(repr(dist))
print("rvecs: [snip]")
# print(repr(rvecs))
print("tvecs: [snip]")
# print(repr(tvecs))

with open("../test/calib.txt", "w") as f:
    print("mtx =", repr(mtx), file=f)
    print("dist =", repr(dist), file=f)
    print("rvecs =", repr(rvecs), file=f)
    print("tvecs =", repr(tvecs), file=f)

def old_undistort(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def undistort(img):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    return dst, dst[y:y+h, x:x+w]

tessa = input("image to test: ")
testimg = cv2.imread(tessa)
cv2.imwrite("../test/calib-test0.png", testimg)
cv2.imwrite("../test/calib-test1.png", undistort(testimg)[0])
cv2.imwrite("../test/calib-test2.png", undistort(testimg)[1])

err = []
for i in tqdm.tqdm(range(len(objpoints))):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    err.append(cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2))
err = np.array(err)
print(f"error mean: {err.mean()}")
print(f"error std: {np.std(err)}")
print(f"error median: {np.median(err)}")
print(f"error interval: {err.min()}..{err.max()}")


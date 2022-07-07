import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('../images/stereoLeft/*.png')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('../images/stereoLeft/stereoLeft1.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
#while True:
#    cv.imshow("frame left", dst)
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cap_left = cv.VideoCapture(0)
while cap_left.isOpened():
    _, frame_left = cap_left.read()
    frame_left = cv.flip(frame_left, 0)
    # undistortRectify
    h, w = frame_left.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    frame_left = cv.remap(frame_left, mapx, mapy, cv.INTER_LINEAR)
    # undistort
    # frame_left = cv.undistort(frame_left, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # frame_left = frame_left[y:y + h, x:x + w]
    cv.imshow("frame left", frame_left)
    if cv.waitKey(1) & 0xFF == ord('q'):
       break

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)))
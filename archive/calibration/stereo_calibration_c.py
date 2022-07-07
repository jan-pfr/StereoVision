import numpy as np
import cv2
from tqdm import tqdm
from pygame import mixer

# Set the path to the images captured by the left and right cameras
pathL = "images/stereoLeft/"
pathR = "images/stereoRight/"

print("Extracting image coordinates of respective 3D pattern ....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []
chessboardSize = (7, 9)
count = 0

for i in tqdm(range(0, 51)):

    imgL = cv2.imread(pathL + "stereoLeft%d.png" % i)
    imgR = cv2.imread(pathR + "stereoRight%d.png" % i)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        #count += 1
        #print("Found Points, " + str(count))

        obj_pts.append(objp)

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        img_ptsL.append(cornersL)

        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        img_ptsR.append(cornersR)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv2.imshow('imgLeft', imgL)
        cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv2.imshow('imgRight', imgR)
        cv2.waitKey(1000)

print("Calculating left camera parameters ... ")
# Calibrating left camera

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, grayL.shape[::-1], None, None)
hL, wL = grayL.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, grayR.shape[::-1], None, None)
hR, wR = grayR.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                                                    img_ptsL,
                                                                                    img_ptsR,
                                                                                    new_mtxL,
                                                                                    distL,
                                                                                    new_mtxR,
                                                                                    distR,
                                                                                    grayL.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale = 1  # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                          grayL.shape[::-1], Rot, Trns,
                                                                          rectify_scale, (0, 0))

# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              grayL.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               grayR.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
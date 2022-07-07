import numpy as np
import cv2
from tqdm import tqdm
from pygame import mixer

# Set the path to the images captured by the left and right cameras
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"
chessboardSize = (8,8)
print("Extracting image coordinates of respective 3D pattern ....\n")
mixer.init()  # you must initialize the mixer
alert = mixer.Sound('/home/ec/opencv/learnopencv/stereo-camera/sounds/beep.wav')
alert.play()

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1, 51)):
    imgL = cv2.imread(pathL + "img%d.png" % i)
    imgR = cv2.imread(pathR + "img%d.png" % i)
    imgL_gray = cv2.imread(pathL + "img%d.png" % i, 0)
    imgR_gray = cv2.imread(pathR + "img%d.png" % i, 0)

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(outputL, (9, 6), None)

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, (9, 6), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (9, 6), cornersL, retL)
        cv2.imshow('cornersR', outputR)
        cv2.imshow('cornersL', outputL)
        cv2.waitKey(0)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)

print("Calculating left camera parameters ... ")
# Calibrating left camera
alert.play()
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
alert.play()
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
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
                                                                                    imgL_gray.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale = 1  # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                          imgL_gray.shape[::-1], Rot, Trns,
                                                                          rectify_scale, (0, 0))

# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()

# disparity map from L/R-image
#### #!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )

see https://raw.githubusercontent.com/abidrahmank/opencv/master/samples/python/stereo_match.py

'''

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.pyrDown(cv2.imread('./images/im0.png'))  # downscale images for faster processing
    imgR = cv2.pyrDown(cv2.imread('./images/im1.png'))

    imgL = cv2.resize(imgL, (640, 480))
    imgR = cv2.resize(imgR, (640, 480))

    # disparity range is tuned for 'aloe' image pair
    window_size = 1
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

    ''' print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')
    '''
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
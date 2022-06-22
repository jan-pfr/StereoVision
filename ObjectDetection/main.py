import cv2 as cv
import imutils
import numpy as np
import time
from matplotlib import pyplot as plt

# Functions for stereo vision and depth estimation
import triangulation as tri
import filtering as filter
import objectDetection as od

# Open both cameras

cap_right = cv.VideoCapture(1)
cap_left = cv.VideoCapture(0)

# Stereo vision setup parameters
frame_rate = 30  # Camera frame rate (maximum at 120 fps)
B = 18.1  # Distance between the cameras [cm]
f = 1  # Camera lense's focal length [mm]
alpha = 60  # Camera field of view in the horizontal plane [degrees]

# HSV Range for a Tennis Ball -> Denpends on light-conditions
loverHSVRange = (21, 121, 0)
higherHSVRange = (30, 255, 193)

# allow Camera warm up
time.sleep(2.0)

while (cap_right.isOpened() and cap_left.isOpened()):
    # grab frame
    succes_left, leftFrame = cap_left.read()
    succes_right, rightFrame = cap_right.read()
    # flip frame because of mounting
    leftFrame = cv.flip(leftFrame, 0)
    rightFrame = cv.flip(rightFrame, 0)

    # If cannot catch any frame, break
    if not succes_right or not succes_left:
        break

    # apply HSV Filter and return x,y for the left cam
    leftFrameMasked = filter.applyFilter(leftFrame, loverHSVRange, higherHSVRange)
    leftFound, center_left, leftFrameMasked = od.detectObject(leftFrame,leftFrameMasked)

    # apply HSV Filter and return x,y for the left cam
    rightFrameMasked = filter.applyFilter(rightFrame, loverHSVRange, higherHSVRange)
    rightFound, center_right, rightFrameMasked = od.detectObject(rightFrame,rightFrameMasked)

    if not leftFound and not rightFound:
        cv.putText(rightFrameMasked, "No traceable object found", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 0, 255), 3)
        cv.putText(leftFrameMasked, "No traceable object found", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 0, 255), 3)
    else:
        depth = tri.find_depth(right_point=center_right,
                               left_point=center_left,
                               frame_right=rightFrameMasked,
                               frame_left=leftFrameMasked,
                               baseline=B,
                               f=f,
                               alpha=alpha)
        cv.putText(rightFrameMasked, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 255, 0), 3)
        cv.putText(leftFrameMasked, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 255, 0), 3)
    cv.imshow("right frame", rightFrame)
    cv.imshow("left frame", leftFrame)
    # Hit "q" to close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()

cv.destroyAllWindows()

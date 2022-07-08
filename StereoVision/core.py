import logging

import cv2 as cv
import numpy as np
import time
from triangulation import Triangulation
from filter import Filter
from objectDetection import ObjectDetection
from counter import CountsPerSec
from cameraCapture import CameraCapture
from trajectoryPrediction import TrajectoryPrediction

# Open both cameras in extra Threads
capLeft = CameraCapture(0).start()
capRight = CameraCapture(1).start()

# toDo: Parameter that have to be moved out of the Code into a config-File

# Amount of minimum Samples, before a trajectroy is tried to be predicted
min_samples = 2
min_time_diff = float(0.04)

# Stereo vision setup parameters
frame_rate = 30  # Camera frame rate (maximum at 120 fps)
B = 18.1  # Distance between the cameras [cm]
f = 1  # Camera lense's focal length [mm]
alpha = 60  # Camera field of view in the horizontal plane [degrees]

# HSV Range for a Tennis Ball -> Denpends on light-conditions
lowerHSVRange = (26, 134, 28)
higherHSVRange = (43, 255, 255)

# Array for coordinates of the object, if one is detected in the picture
positions = []

# Empty Arrays for old frames
left_frame_old = np.array([])
right_frame_old = np.array([])

# counter for keeping track of the amount of coordinates so far.
count = 0

# Logger configuration
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Create Objects from the TrajectoryPredictor and others
# toDo: better Object Names
tp = TrajectoryPrediction(min_samples)
tr = Triangulation()
fltr = Filter(lowerHSVRange, higherHSVRange)
od = ObjectDetection()

# allow Camera warm up
time.sleep(2.0)


### Functions that are used inside this application

def put_iterations_per_sec(frame, iterations_per_sec):
    """
    Put the Iterations per Second Number on to a frame.

    :param frame: as np.array
    :param iterations_per_sec: number
    :return: frame with number
    """

    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
               (10, 650), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


def process_frame(frame):
    """
    Apply filter and object detection on frame.
    If no Object is found, the boolean found is False and the center is None.
    The frame is returned in any case.

    :param frame: as np.array
    :return: center found or not as boolean, coordinates as array , maskedframe as np.array
    """

    frameMasked = fltr.apply_filter(frame)
    found, center, frameMasked = od.detect_object(frame, frameMasked)
    return found, center, frameMasked


def draw_points_to_frame(points: np.array, frame) -> np.array:
    """
    Get points and draws them onto the frame.

    :param points: Array of 3D Points, ony x and y are used
    :param frame: as np.array
    :return: the frame.
    """
    for pt in points:
        x, y = pt[0], pt[1]
        if 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0]:
            cv.circle(frame, (int(x), int(y)), 2, (255, 0, 0), 10)
    return frame


# optional counter for iterations per second
cps = CountsPerSec().start()

# Main loop
while True:

    # Collect frames from the camera threads.
    leftTimestamp, leftFrame = capLeft.getFrame()
    rightTimestamp, rightFrame = capRight.getFrame()

    # Flip the frames.
    leftFrame = cv.flip(leftFrame, 0)
    rightFrame = cv.flip(rightFrame, 0)

    # Check if the images are too far apart in time.
    diff = abs(leftTimestamp - rightTimestamp)
    if diff > min_time_diff:
        logging.info(f'Diff too big: {round(diff, 4)} seconds')
        continue

    # Check, if the new and the old images from the camera are the same.
    if left_frame_old.shape[0] == 0 or right_frame_old.shape[0] == 0:
        left_frame_old = leftFrame
        right_frame_old = rightFrame
    elif np.array_equal(leftFrame, left_frame_old) or np.array_equal(rightFrame, right_frame_old):
        logging.info('duplicate in one of the cams')
        continue

    # the pictures getting processed
    leftFound, leftCenter, leftFrameMasked = process_frame(leftFrame)
    rightFound, rightCenter, rightFrameMasked = process_frame(rightFrame)

    # If there is no object in left or right frame, this text is getting put onto the frame.
    if not leftFound or not rightFound:
        cv.putText(rightFrameMasked, "No traceable object found", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 0, 255), 3)
        cv.putText(leftFrameMasked, "No traceable object found", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 0, 255), 3)

        # in the case that the tracking is lost, the counter and the array of coordinates is getting reset
        count = 0
        positions = []

    else:
        count = count + 1
        # calculate the distance between the object and the baseline of the cameras [cm]
        depth = tr.find_depth(right_point=rightCenter,
                               left_point=leftCenter,
                               frame_right=rightFrameMasked,
                               frame_left=leftFrameMasked,
                               baseline=B,
                               f=f,
                               alpha=alpha)
        # the coordinates are put into an array
        position = (leftCenter[0], leftCenter[1], int(depth))

        # if the array is empty, the current coordinate is put as the first entry.
        positions.append(position)

        # determine if already enough coordinates were collected
        if count > min_samples:
            # return an array, containing predicted  n (20) coordinates.
            predicted_path = tp.predict_path(np.asarray(positions), 20)
            leftFrame = draw_points_to_frame(predicted_path, leftFrame)

        # since a depth has been calculated, it will be shown in the frame
        cv.putText(rightFrameMasked, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 255, 0), 3)
        cv.putText(leftFrameMasked, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (0, 255, 0), 3)
        logging.info(positions)

    # the counter of iterations per second is put on to the frame as well
    cps.increment()
    leftFrame = put_iterations_per_sec(leftFrame, cps.countsPerSec())

    # the left and right frame is stacked together for better view.
    frames = np.hstack((leftFrame, rightFrame))
    cv.imshow("Frames", frames)

    # Hit "q" to close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        logging.info("Saving parameters!")
        file = open("params.txt", "w")
        for corrd in positions:
            file.write(str(corrd[0]) + "," + str(corrd[1]) + "," + str(corrd[2]) + "\n")
        file.close()
        capLeft.stop()
        capRight.stop()
        break

cv.destroyAllWindows()

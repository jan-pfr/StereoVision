import logging
import time

import cv2 as cv
import numpy as np

from bin.cameraCapture import CameraCapture
from bin.object_tracking.counter import CountsPerSec
from bin.object_tracking.filter import Filter
from bin.object_tracking.objectDetection import ObjectDetection
from bin.object_tracking.trajectoryPrediction import TrajectoryPrediction
from bin.object_tracking.triangulation import Triangulation


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


class ObjectTracking:

    def __init__(self, _config):
        """
        Initialize the object.

        :param _config:
        """
        # Import settings, fallbacks if the import fails

        # Trajectory Settings
        self.config = _config
        self.min_samples = self.config['TrajectoryPredictionSettings'].getint('minSamples', fallback=2)
        self.min_time_diff = self.config['TrajectoryPredictionSettings'].getfloat('minTimeDifference', fallback=0.04)

        # Camera settings
        self.baseline = self.config['CameraSettings'].getfloat('baseline', fallback=18.1)
        self.focal_length = self.config['CameraSettings'].getint('focalLength', fallback=3)
        self.fov = self.config['CameraSettings'].getint('fieldOfView', fallback=60)
        self.left_id = self.config['CameraSettings'].getint('leftID', fallback=0)
        self.right_id = self.config['CameraSettings'].getint('rightID', fallback=1)

        # HSV ranges
        self.low_hsv_range = self.config['HSVRange'].gettuple('lowHSVRange')
        self.high_hsv_range = self.config['HSVRange'].gettuple('highHSVRange')

        # Logger configuration
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

        # Initialize helper objects
        self.tp = TrajectoryPrediction(self.min_samples)
        self.tr = Triangulation()
        self.fltr = Filter(self.low_hsv_range, self.high_hsv_range)
        self.od = ObjectDetection()

        # Array for coordinates of the object, if one is detected in the picture
        self.positions = []
        self.trans_matrix = self.config['AffineTransformationMatrix'].getarray('matrix')
        self.predicted_path = None

        # Empty Arrays for old frames
        self.left_frame_old = np.array([])
        self.right_frame_old = np.array([])

    def start(self):
        """
        Main loop. Press q to quit.


        :return: Nothing.
        """

        # Cameras attributes in independent threads
        capLeft = CameraCapture(self.left_id).start()
        capRight = CameraCapture(self.right_id).start()
        time.sleep(1)  # warm up cams

        # create openCV window
        window_name = 'Stereo Vision Application'
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 1280, 1080)

        # optional counter for iterations per second
        cps = CountsPerSec().start()

        while True:

            # Collect frames from the camera threads.
            leftTimestamp, leftFrame = capLeft.getFrame()
            rightTimestamp, rightFrame = capRight.getFrame()

            # Flip the frames.
            leftFrame = cv.flip(leftFrame, 0)
            rightFrame = cv.flip(rightFrame, 0)

            # Check if the images are too far apart in time.
            diff = abs(leftTimestamp - rightTimestamp)
            if 0.029 > diff > self.min_time_diff:
                logging.info(f'Diff too big: {round(diff, 4)} seconds')
                continue

            # Check, if the new and the old images from the camera are the same.
            if self.left_frame_old.shape[0] == 0 or self.right_frame_old.shape[0] == 0:
                self.left_frame_old = leftFrame
                self.right_frame_old = rightFrame
            elif np.array_equal(leftFrame, self.left_frame_old) or np.array_equal(rightFrame, self.right_frame_old):
                logging.info('duplicate in one of the cams')
                continue
            # the pictures getting processed
            leftFound, leftCenter, leftFrameMasked = self.process_frame(leftFrame)
            rightFound, rightCenter, rightFrameMasked = self.process_frame(rightFrame)

            # If there is no object in left or right frame, this text is getting put onto the frame.
            if not leftFound or not rightFound:
                cv.putText(rightFrameMasked, "No traceable object found", (10, 700), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (0, 0, 255), 2)
                cv.putText(leftFrameMasked, "No traceable object found", (10, 700), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (0, 0, 255), 2)

                # in the case that the tracking is lost, the counter and the array of coordinates is getting reset
                self.positions = []

            else:
                # calculate the distance between the object and the baseline of the cameras [cm]
                depth = self.tr.find_depth(right_point=rightCenter,
                                           left_point=leftCenter,
                                           frame_right=rightFrameMasked,
                                           frame_left=leftFrameMasked,
                                           baseline=self.baseline,
                                           f=self.focal_length,
                                           alpha=self.fov)

                # the coordinates are put into a tuple
                position = (leftCenter[0], leftCenter[1], int(depth))

                # if the array is empty, the current coordinate is put as the first entry
                self.positions.append(position)

                # determine if already enough coordinates were collected
                if len(self.positions) > self.min_samples:
                    # return an array, containing predicted n (20) coordinates.
                    self.predicted_path = self.tp.predict_path(np.asarray(self.positions), 20)
                    leftFrame = draw_points_to_frame(self.predicted_path, leftFrame)
                    intercept = self.find_interception(self.predicted_path)
                    cv.circle(leftFrame, (int(intercept[1][0]), int(intercept[1][1])), 2, (255, 255, 0), 18)
                    self.move_to_robot(intercept[0])

                # since a depth has been calculated, it will be shown in the frame
                cv.putText(rightFrameMasked, "Distance: " + str(round(depth, 1)), (10, 700),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1.2,
                           (0, 255, 0), 2)
                cv.putText(leftFrameMasked, "Distance: " + str(round(depth, 1)), (10, 700), cv.FONT_HERSHEY_SIMPLEX,
                           1.2,
                           (0, 255, 0), 2)

            # the counter of iterations per second is put on to the frame as well
            cps.increment()
            # leftFrame = put_iterations_per_sec(leftFrame, cps.countsPerSec())

            # the left and right frame are stacked together for better view.
            frames = np.hstack((leftFrame, rightFrame))
            cv.imshow(window_name, frames)

            # Hit "q" to close the window
            if cv.waitKey(1) & 0xFF == ord('q'):
                logging.info("Saving parameters!")
                logging.info('The OpenCV Window will freeze. This is a normal behaviour.')
                capLeft.stop()
                capRight.stop()
                cv.destroyAllWindows()
                break

    def process_frame(self, frame):
        """
        Apply filter and object detection on frame.
        If no Object is found, the boolean found is False and the center is None.
        The frame is returned in any case.

        :param frame: as np.array
        :return: center found or not as boolean, coordinates as array , maskedframe as np.array
        """

        frameMasked = self.fltr.apply_filter(frame)
        found, center, frameMasked = self.od.detect_object(frame, frameMasked)
        return found, center, frameMasked

    def find_interception(self, positions: np.array):
        """
        Find the interception point of the predicted path.
        :param positions:
        :return:
        """
        # baseline
        b = 720
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        # function for Y relative to X
        y_x = np.polyfit(y, x, 2)
        # function for Y relative to Z
        y_z = np.polyfit(y, z, 2)

        # calculate the interception of the two functions by using the y-coordinate as a parameter
        x_intercept = np.polyval(y_x, b)
        z_intercept = np.polyval(y_z, b)
        # return the interception point
        interception_rob = np.dot(self.trans_matrix, np.asarray((x_intercept, b, z_intercept, 1)))
        interception_cam = np.array([x_intercept, b, z_intercept])
        return interception_rob, interception_cam

    def move_to_robot(self, pos: np.array):
        """
        Move the robot to the given position.
        :param pos:
        :return:
        """
        print(pos[0], pos[1], pos[2])
        return

    def __del__(self):
        """
        Destroy current instance of the object.

        :return: Nothing
        """
        pass

import logging
import time

import cv2 as cv
import numpy as np

from bin.cameraCapture import CameraCapture
from bin.object_tracking.filter import Filter
from bin.object_tracking.objectDetection import ObjectDetection
from bin.object_tracking.trajectoryPrediction import TrajectoryPrediction
from bin.object_tracking.triangulation import Triangulation


class CoordCalibration:

    def __init__(self, _config):
        self.config = _config

        self.hsv_low = self.config['HSVRange'].gettuple('lowHSVRange')
        self.hsv_high = self.config['HSVRange'].gettuple('highHSVRange')

        #
        self.calibration_points = self.config['CoordCalibration'].getarray('calibpoints')

        self.min_samples = self.config['TrajectoryPredictionSettings'].getint('minSamples', fallback=2)
        self.min_time_diff = self.config['TrajectoryPredictionSettings'].getfloat('minTimeDifference', fallback=0.04)

        # Camera settings
        self.baseline = self.config['CameraSettings'].getfloat('baseline', fallback=18.1)
        self.fov = self.config['CameraSettings'].getint('fieldOfView', fallback=60)
        self.left_id = self.config['CameraSettings'].getint('leftID', fallback=0)
        self.right_id = self.config['CameraSettings'].getint('rightID', fallback=1)

        self.window_name = 'Coordinate Calibration'
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, 1280, 720)

        # Logger configuration
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

        # Initialize helper objects
        self.tp = TrajectoryPrediction(self.min_samples)
        self.tr = Triangulation()
        self.fltr = Filter(self.hsv_low, self.hsv_high)
        self.od = ObjectDetection()
        # Array for coordinates of the object, if one is detected in the picture
        self.positions = []
        self.trans_matrix = np.array([])

    def start(self):
        counter = 1

        # Cameras attributes in independent threads
        capLeft = CameraCapture(self.left_id).start()
        capRight = CameraCapture(self.right_id).start()
        time.sleep(1)  # warm up cams

        while True:

            # Collect frames from the camera threads.
            leftTimestamp, leftFrame = capLeft.getFrame()
            rightTimestamp, rightFrame = capRight.getFrame()

            # Flip the frames.
            leftFrame = cv.flip(leftFrame, 0)
            rightFrame = cv.flip(rightFrame, 0)

            # the pictures getting processed
            leftFound, leftCenter, leftFrameMasked = self.process_frame(leftFrame)
            rightFound, rightCenter, rightFrameMasked = self.process_frame(rightFrame)

            # If there is no object in left or right frame, this text is getting put onto the frame.
            if not leftFound or not rightFound:
                cv.putText(rightFrameMasked, "No traceable object found", (10, 700), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (0, 0, 255), 2)
                cv.putText(leftFrameMasked, "No traceable object found", (10, 700), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (0, 0, 255), 2)

            else:
                # calculate the distance between the object and the baseline of the cameras [cm]
                depth = self.tr.find_depth(right_point=rightCenter,
                                           left_point=leftCenter,
                                           frame_right=rightFrameMasked,
                                           frame_left=leftFrameMasked,
                                           baseline=self.baseline,
                                           alpha=self.fov)

                # the coordinates are put into a tuple
                position = (leftCenter[0], leftCenter[1], int(depth))

                # since a depth has been calculated, it will be shown in the frame
                try:
                    cv.putText(leftFrameMasked, "Distance: " + str(round(depth, 1)), (10, 700), cv.FONT_HERSHEY_SIMPLEX,
                               1.2,
                               (0, 255, 0), 2)
                    cv.putText(leftFrameMasked, "Amount (max. 4): " + str(counter - 1), (10, 650),
                               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1)
                    cv.putText(leftFrameMasked,
                               "Press s for saving the coordinate: " + str(self.calibration_points[counter - 1]),
                               (10, 600), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1)
                except:
                    pass

                    # the left and right frame are stacked together for better view.
            frames = np.hstack((leftFrame, rightFrame))
            cv.imshow(self.window_name, frames)

            key = cv.waitKey(1)
            if key == ord('q') or counter == 5:
                if counter == 5:
                    self.calculate_trans_matrix()
                logging.info("Saving parameters!")
                logging.info('The OpenCV Window will freeze. This is a normal behaviour.')
                capLeft.stop()
                capRight.stop()
                cv.destroyAllWindows()
                break
            elif key == ord('s'):
                self.positions.append(position)
                counter = counter + 1

    def calculate_trans_matrix(self):
        """
        Estimate affine transformation matrix
        :return:
        """
        ret, trans_matrix, mask = cv.estimateAffine3D(np.float32(self.positions),
                                                      np.float32(self.calibration_points),
                                                      confidence=.99)
        if not ret:
            logging.info('Transform failed.')
        else:
            self.update_config(trans_matrix)
            print(trans_matrix)

    def update_config(self, matrix: np.array):
        """
        Takes the numpy array as input and persist it into the file config.ini.

        :return:
        """
        self.config['AffineTransformationMatrix']['matrix'] = str(matrix)

        with open("./config/config.ini", "w") as file:
            self.config.write(file)

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

    def __del__(self):
        pass

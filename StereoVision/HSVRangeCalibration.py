import configparser
import logging
import cv2 as cv
import numpy as np

from cameraCapture import CameraCapture


def do_nothing(i):
    pass


def parse_int_tuple(s: str):
    return tuple(int(i.strip()) for i in s[1:-1].split(','))


class HSVRangeCalibration:

    def __init__(self):
        # Read configuration
        self.config = configparser.ConfigParser(converters={'tuple': parse_int_tuple})
        self.config.read('config.ini')

        self.left_id = self.config['CameraSettings'].getint('leftID', fallback=0)

        self.cam = CameraCapture(self.left_id).start()

        self.window_name = 'HSV Range Calibration'
        cv.namedWindow(self.window_name)

        cv.createTrackbar('H - low', self.window_name, 0, 179, do_nothing)
        cv.createTrackbar('H - high', self.window_name, 179, 179, do_nothing)
        cv.createTrackbar('S - low', self.window_name, 0, 255, do_nothing)
        cv.createTrackbar('S - high', self.window_name, 255, 255, do_nothing)
        cv.createTrackbar('V - low', self.window_name, 0, 255, do_nothing)
        cv.createTrackbar('V - high', self.window_name, 255, 255, do_nothing)

        self.hsv_low = np.array([])
        self.hsv_high = np.array([])

    def update_config(self):
        self.config['HSVRange']['lowHSVRange'] = str(self.hsv_low)
        self.config['HSVRange']['highHSVRange'] = str(self.hsv_high)

        with open("config.ini", "w") as file:
            self.config.write(file)

    def start(self):

        while True:
            # Collect frames from the camera threads.
            _, frame = self.cam.getFrame()

            # Flip the frame
            frame = cv.flip(frame, 0)

            lowH = cv.getTrackbarPos('H - low', self.window_name)
            highH = cv.getTrackbarPos('H - high', self.window_name)
            lowS = cv.getTrackbarPos('S - low', self.window_name)
            highS = cv.getTrackbarPos('S - high', self.window_name)
            lowV = cv.getTrackbarPos('V - low', self.window_name)
            highV = cv.getTrackbarPos('V - high', self.window_name)

            self.hsv_low = (lowH, lowS, lowV)
            self.hsv_high = (highH, highS, highV)
            print(self.hsv_low)

            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_frame, self.hsv_low, self.hsv_high)
            frame = cv.bitwise_and(frame, frame, mask=mask)

            cv.imshow(self.window_name, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                logging.info('Save Params')
                self.update_config()
                self.cam.stop()
                self.cam.join()
                cv.destroyAllWindows()
                logging.info('The OpenCV Window will freeze. This is a normal behaviour.')
                break

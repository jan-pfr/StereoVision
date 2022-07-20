import logging
import cv2 as cv
import numpy as np

from bin.cameraCapture import CameraCapture


def do_nothing(i):
    pass


class HSVRangeCalibration:

    def __init__(self, _config):
        # Read configuration
        self.config = _config
        self.left_id = self.config['CameraSettings'].getint('leftID', fallback=0)

        # initial HSV ranges
        self.initial_hsv_low = self.config['HSVRange'].gettuple('lowHSVRange')
        self.initial_hsv_high = self.config['HSVRange'].gettuple('highHSVRange')

        self.cam = CameraCapture(self.left_id).start()

        self.window_name = 'HSV Range Calibration'
        cv.namedWindow(self.window_name)

        # create Trackbars with loaded in values
        cv.createTrackbar('H - low', self.window_name, self.initial_hsv_low[0], 179, do_nothing)
        cv.createTrackbar('H - high', self.window_name, self.initial_hsv_high[0], 179, do_nothing)
        cv.createTrackbar('S - low', self.window_name, self.initial_hsv_low[1], 255, do_nothing)
        cv.createTrackbar('S - high', self.window_name, self.initial_hsv_high[1], 255, do_nothing)
        cv.createTrackbar('V - low', self.window_name, self.initial_hsv_low[2], 255, do_nothing)
        cv.createTrackbar('V - high', self.window_name, self.initial_hsv_high[2], 255, do_nothing)

        # Logger configuration
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

        self.hsv_low = np.array([])
        self.hsv_high = np.array([])


    def update_config(self):
        self.config['HSVRange']['lowHSVRange'] = str(self.hsv_low)
        self.config['HSVRange']['highHSVRange'] = str(self.hsv_high)

        with open("./config/config.ini", "w") as file:
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

            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv_frame, self.hsv_low, self.hsv_high)
            frame = cv.bitwise_and(frame, frame, mask=mask)

            cv.imshow(self.window_name, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                logging.info('Save Params')
                self.update_config()
                self.cam.stop()
                cv.destroyAllWindows()
                logging.info('The OpenCV Window will freeze. This is a normal behaviour.')
                break

    def __del__(self):
        pass

import cv2 as cv
import numpy as np


class Filter:
    """
    Class to mask out a curtain HSV range on a given frame.
    """

    def __init__(self, low_hsv_range: tuple, high_hsv_range: tuple):
        """
        Creates a Filter Object with given hsv ranges.

        :param low_hsv_range: Lower end of the HSV range
        :param high_hsv_range: Higher end of the HSV range
        """
        self.low_hsv_range = low_hsv_range
        self.high_hsv_range = high_hsv_range

    def apply_filter(self, frame: np.array) -> np.array:
        """
        Applies filter on given frame.

        :param frame: original frame from camera
        :return: masked out binary frame
        """

        # convert to hsv and blur frames
        blurred_frame = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)

        # create mask with hsv range and remove small blobs
        masked_frame = cv.inRange(hsv, self.low_hsv_range, self.high_hsv_range)
        masked_frame = cv.erode(masked_frame, None, iterations=2)
        masked_frame = cv.dilate(masked_frame, None, iterations=2)
        return masked_frame

import cv2 as cv
import imutils
import numpy as np


class ObjectDetection:
    """
    Class to detect a object in a given binary frame.
    """


    def __init__(self):
        """
        Creates a Object Detection object.
        """
        pass

    def detect_object(self, frame: np.array, masked_frame: np.array):

        """
        Detects an object from the given given masked out image.

        :param frame: Original frame
        :param masked_frame: masked out frame
        :return: Boolean if center found, center coordinates, original Frame with circle and center point drawn
        """

        # find contours
        contours = cv.findContours(masked_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        center = None
        # only proceed, if at least one contour have been found
        if len(contours) > 0:

            # find largest contour
            c = max(contours, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 17:
                # draw the circle and center on the frame,
                center_found = True
                cv.circle(frame, (int(x), int(y)), int(radius),
                          (0, 255, 255), 2)
                cv.circle(frame, center, 2, (0, 0, 255), -1)
        else:
            center_found = False
        return center_found, center, frame

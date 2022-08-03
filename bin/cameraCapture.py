import time
from threading import Thread

import cv2


class CameraCapture:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src: int, exposure: int, saturation: int, img_size: tuple):
        """
        Creates cameraCapture object.

        :param src: ID of the camera
        """
        self.img_size = img_size
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.stream.set(cv2.CAP_PROP_SATURATION, saturation)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Start grab frames in a loop as a dedicated thread.
        :return: the object
        """
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        Loop to continuous grab frames from the camera and safe a timestamp,
        when the frame was grabbed.
        :return:
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                self.timestamp = time.perf_counter()

    def stop(self):
        """
        stop the Loop.
        :return:
        """
        self.stopped = True

    def getFrame(self):
        """
        Return Frame with according timestamp
        :return: timestamp as float and frame
        """
        return self.timestamp, cv2.resize(self.frame, self.img_size, interpolation=cv2.INTER_AREA)

from StereoVision.cameraCapture import CameraCapture
import cv2 as cv
from StereoVision.counter import CountsPerSec
import numpy as np
capLeft = CameraCapture(0).start()
capRight = CameraCapture(1).start()

def putIterationsPerSec(frame, iterations_per_sec):


    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 650), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

cps = CountsPerSec().start()
while True:

    leftFrame = cv.flip(capLeft.frame, 0)
    rightFrame = cv.flip(capRight.frame, 0)

    leftFrame = putIterationsPerSec(leftFrame, cps.countsPerSec())
    frames = np.hstack((leftFrame, rightFrame))
    cv.imshow("Frames", frames)
    cps.increment()

    # Hit "q" to close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        capLeft.stop()
        capRight.stop()
        break

cv.destroyAllWindows()
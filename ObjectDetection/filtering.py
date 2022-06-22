import cv2 as cv

def applyFilter(frame, lowerHSVRange, higherHSVRange):
    # convert und blur frames
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # create mask with hsv range and remove small blobs
    mask1 = cv.inRange(hsv, lowerHSVRange, higherHSVRange)
    mask2 = cv.erode(mask1, None, iterations=2)
    mask3 = cv.dilate(mask2, None, iterations=2)
    return mask3


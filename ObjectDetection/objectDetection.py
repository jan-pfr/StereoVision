import cv2 as cv
import imutils
def detectObject(frame, maskedFrame):
    # find contours
    contours = cv.findContours(maskedFrame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        if radius > 20:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv.circle(frame, (int(x), int(y)), int(radius),
                      (0, 255, 255), 2)
            cv.circle(frame, center, 2, (0, 0, 255), -1)
    return center, frame

import cv2 as cv

import bin.object_tracking.objectDetection as od

lowerHSVRange = (18, 172, 55)
higherHSVRange = (33, 255, 183)

img = cv.imread('../images/stereoLeft/stereoLeft0.png')

while True:
    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

img2 = cv.GaussianBlur(img, (11, 11), 0)

while True:
    cv.imshow("img", img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
while True:
    cv.imshow("img", img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

img2 = cv.inRange(img2, lowerHSVRange, higherHSVRange)
while True:
    cv.imshow("img", img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

img2 = cv.erode(img2, None, iterations=10)
img2 = cv.dilate(img2, None, iterations=10)
while True:
    cv.imshow("img", img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

_, _, img = od.detectObject(img, img2)
while True:
    cv.imshow("img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break


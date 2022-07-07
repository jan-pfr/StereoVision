import cv2
import numpy as np

leftCap= cv2.VideoCapture(0)
rightCap = cv2.VideoCapture(1)

width= int(leftCap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(leftCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writerLeft= cv2.VideoWriter('Left.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))
writerRight= cv2.VideoWriter('Right.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))


while True:
    ret,leftFrame= leftCap.read()
    ret, rightFrame = rightCap.read()

    leftFrame = cv2.flip(leftFrame, 0)
    rightFrame = cv2.flip(rightFrame, 0)

    writerLeft.write(leftFrame)
    writerRight.write(rightFrame)

    frames = np.hstack((leftFrame, rightFrame))
    cv2.imshow("Frames", frames)

    if cv2.waitKey(1) & 0xFF == 27:
        break


leftCap.release()
rightCap.release()
writerLeft.release()
writerRight.release()
cv2.destroyAllWindows()
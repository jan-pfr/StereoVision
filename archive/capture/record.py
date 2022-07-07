import numpy as np
import cv2
import glob

# Set the values for your cameras
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

# Use these if you need high resolution.
#capL.set(3, 1280) # width
#capL.set(4, 720) # height
#capR.set(3, 1280) # width
#capR.set(4, 720) # height
i = 0



def main():
    global i
    while True:
        # Grab and retreive for sync
        if not (capL.grab() and capR.grab()):
            print("No more frames")
            break

        leftFrame = cv2.flip(capL.retrieve()[1], 0)
        rightFrame = cv2.flip(capR.retrieve()[1], 0 )

        cv2.imshow('capL', leftFrame)
        cv2.imshow('capR', rightFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite("../images/stereoLeft/stereoLeft" + str(i) + ".png", leftFrame)
            cv2.imwrite("../images/stereoRight/stereoRight" + str(i) + ".png", rightFrame)
            i += 1
            print("total images: " + str(i*2))

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import numpy as np
import cv2

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
num = 0

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open both cameras
cap_right = cv2.VideoCapture(1)
cap_left = cv2.VideoCapture(0)
#cap_left.set(3, 1280) # width
#cap_left.set(4, 720) # height

#cap_right.set(3, 1280) # width
#cap_right.set(4, 720) # height

while cap_right.isOpened() and cap_left.isOpened():
    _, frame_right = cap_right.read()
    _, frame_left = cap_left.read()
    frame_right = cv2.flip(frame_left,0)
    frame_left = cv2.flip(frame_left,0)

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    # Show the framesq
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./images/corrected/stereoLeft/imageL' + str(num) + '.png', frame_left)
        cv2.imwrite('./images/corrected/stereoRight/imageR' + str(num) + '.png', frame_right)
        print("images saved!")
        num += 1

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
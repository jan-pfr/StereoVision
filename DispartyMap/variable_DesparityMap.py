import numpy as np
import cv2

# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 1  # Camera ID for left camera
CamR_id = 0  # Camera ID for right camera

CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Reading the mapping values for stereo image rectification
#cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_READ)
#Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
#Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
#Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
#Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
#cv_file.release()


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)


# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:

    # Capturing and storing left and right camera images
    imgL = cv2.imread('../images/stereoLeft/stereoLeft1.png')
    imgR = cv2.imread('../images/stereoRight/stereoRight1.png')
    small_to_large_image_size_ratio = 0.8
    small_imgL = cv2.resize(imgL,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=small_to_large_image_size_ratio,
                           fy=small_to_large_image_size_ratio,
                           interpolation=cv2.INTER_NEAREST)
    small_imgR = cv2.resize(imgR,  # original image
                           (0, 0),  # set fx and fy, not the final size
                           fx=small_to_large_image_size_ratio,
                           fy=small_to_large_image_size_ratio,
                           interpolation=cv2.INTER_NEAREST)

    # Proceed only if the frames have been captured

    imgR_gray = cv2.cvtColor(small_imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(small_imgL, cv2.COLOR_BGR2GRAY)

    # Applying stereo image rectification on the left image
    #Left_nice = cv2.remap(imgL_gray,
                          #Left_Stereo_Map_x,
                         # Left_Stereo_Map_y,
                         # cv2.INTER_LANCZOS4,
                         # cv2.BORDER_CONSTANT,
                         # 0)

    # Applying stereo image rectification on the right image
    #Right_nice = cv2.remap(imgR_gray,
                           #Right_Stereo_Map_x,
                           #Right_Stereo_Map_y,
                           #cv2.INTER_LANCZOS4,
                           #cv2.BORDER_CONSTANT,
                          # 0)

    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')


    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(imgL_gray, imgR_gray)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Converting to float32
    disparity = disparity.astype(np.float32)

    # Scaling down the disparity values and normalizing them
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    font = cv2.FONT_HERSHEY_SIMPLEX
    disparity = cv2.putText(disparity, "numDisparities ", (10, 30), font, 0.6, (255,255,255), 2)
    disparity = cv2.putText(disparity, "blockSize ", (10, 50), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "preFilterType ", (10, 70), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "preFilterSize ", (10, 90), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "preFilterCap ", (10, 110), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "textureThreshold ", (10, 130), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "uniquenessRatio ", (10, 150), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "speckleRange ", (10, 170), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "speckleWindowSize ", (10, 190), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "disp12MaxDiff ", (10, 210), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, "minDisparity ", (10, 230), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(numDisparities), (200, 30), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(blockSize), (200, 50), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(preFilterType), (200, 70), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(preFilterSize), (200, 90), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(preFilterCap), (200, 110), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(textureThreshold), (200, 130), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(uniquenessRatio), (200, 150), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(speckleRange), (200, 170), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(speckleWindowSize), (200, 190), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(disp12MaxDiff), (200, 210), font, 0.6, (255, 41, 74), 2)
    disparity = cv2.putText(disparity, str(minDisparity), (200, 230), font, 0.6, (255, 41, 74), 2)
    # Displaying the disparity map
    cv2.imshow("disp", disparity)

    # Close window using esc key
    if cv2.waitKey(1) == 27:
         break

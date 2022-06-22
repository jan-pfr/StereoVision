import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('../images/stereoLeft/stereoLeft1.png',0)
imgR = cv.imread('../images/stereoRight/stereoRight1.png',0)

small_to_large_image_size_ratio = 0.4
small_imgL = cv.resize(imgL, # original image
                       (0,0), # set fx and fy, not the final size
                       fx =small_to_large_image_size_ratio,
                       fy =small_to_large_image_size_ratio,
                       interpolation =cv.INTER_NEAREST)
small_imgR = cv.resize(imgR, # original image
                       (0,0), # set fx and fy, not the final size
                       fx =small_to_large_image_size_ratio,
                       fy =small_to_large_image_size_ratio,
                       interpolation =cv.INTER_NEAREST)



stereo = cv.StereoBM_create(numDisparities=128, blockSize=11)
stereo.setPreFilterSize(5)
stereo.setPreFilterCap(63)
stereo.setTextureThreshold(1)
stereo.setUniquenessRatio(0)
stereo.setSpeckleRange(50)
stereo.setSpeckleWindowSize(150)
disparity = stereo.compute(small_imgL, small_imgR)
plt.imshow(disparity, 'gray')
plt.show()
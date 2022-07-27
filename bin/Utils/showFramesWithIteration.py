import cv2
import numpy as np



src = np.array([[50,50,30],[450,450,30],[70,420,210],[420,70,30]],np.float32)
dst = np.array([[0,0, 1],[299,299, 23],[0,299,200],[299,0,1]],np.float32)

ret = cv2.getPerspectiveTransform(src,dst)
print(ret)
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from StereoVision.cameraCapture import CameraCapture

capLeft = CameraCapture(0).start()
capRight = CameraCapture(1).start()
time.sleep(2)


totalStart = time.perf_counter()

# Start default camera
imgSample = 5000
ms = []
# Grab a few frames
for i in range(0, imgSample):
    leftTimestamp, frameLeft = capLeft.getFrame()
    rightTimestamp, frameRight = capRight.getFrame()
    frames = np.hstack((frameLeft, frameRight))
    cv2.imshow("Frames", frames)
    end = time.perf_counter()
    seconds = abs(leftTimestamp-rightTimestamp)
    ms.append(seconds)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        capLeft.stop()
        capRight.stop()
        break

totalEnd = time.perf_counter()
capRight.stop()
capLeft.stop()
cv2.destroyAllWindows()
ms = np.asarray(ms)
samples = np.arange(ms.shape[0])[:, np.newaxis]
fig, ax = plt.subplots()
ax.scatter(samples, ms)
ax.set(xlabel='samples', ylabel='time (seconds)',
       title='time per sample')
fig2, ax2 = plt.subplots()
sorted_ms = np.sort(ms)
p = 1. * np.arange(len(sorted_ms)) / float(len(sorted_ms) - 1)
ax2.scatter(sorted_ms, p)
ax2.set(xlabel='sorted seconds', ylabel='Verteilung')
plt.show()
print(totalEnd - totalStart)

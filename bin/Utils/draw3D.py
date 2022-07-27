import numpy as np
import matplotlib.pyplot as plt
from bin.object_tracking.trajectoryPrediction import TrajectoryPrediction
import seaborn as sns

tp = TrajectoryPrediction(2)


def printSomething (points):
    sns.set_style("whitegrid", {'axes.grid': False})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #ax.elev = -69
    #ax.azim = 90

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.arange(points.shape[0]))
    pointsPredicted = tp.predict_path(points, 15)
    ax.plot(pointsPredicted[:, 0], pointsPredicted[:, 1], pointsPredicted[:, 2])
    ax1.plot(points[:, 0])
    ax1.plot(pointsPredicted[:, 0])
    ax2.plot(points[:, 1])
    ax2.plot(pointsPredicted[:, 1])
    ax3.plot(points[:, 2])
    ax3.plot(pointsPredicted[:, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

file = open("../params.txt", "r")
corrds = []
for corrd in file:
    corrds.append(corrd.split(","))
printSomething(np.asarray(corrds).astype(float))
# labels and input is switched.
# ax.plot3D(X, Z, Y, 'red')
# ax.plot3D(fitx, fitz, fity, 'blue')
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')
# plt.show()
file.close()
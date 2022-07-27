# ur5_3.py: Echo client program to send commands via socket to robot
# see: https://www.zacobria.com/universal-robots-knowledge-base-tech-support-forum-hints-tips/universal-robots-script-programming/
# neu in V2: Ansteuerung mit "korrektem" x-Abschnitt, mit conf-Datei für Parameter
# neu in V3: interne Transformation von Pixel-Koord (0...width [Pixel]) nach Robot-Coords ( [m])
#
import argparse  # for parsing command line argument
import socket  # for sending commands to robot
import time  # for waiting some time for robot to complete move
from enum import Enum  # enumerate some predefined positions

import commentjson  # for parsing config-file with parameters
import numpy as np  # math

# see config file: HOST = "127.0.0.1" # The remote host
# see config file: PORT = 30002 # The same port as used by the server
sloMo = 0.9  # slow motion factor: 0.2 ... 1.0 ... 2.0
t1 = 0.9 * sloMo  # time constants [s] to wait for robot to complete
t2 = 0.5 * sloMo
t3 = 0.9  # time before next command


class Position(Enum):
    # poses in Standard orientation, i.e. +X-axis pointing forward
    right = 0
    middle = 1
    left = 2
    # poses in Robo-Lab, i.e. -Y-axis pointing forward
    rightRL = 3
    middleRL = 4
    leftRL = 5


class urControl():
    # main class to control the URx-robot
    def __init__(self, conf):
        # define some poses 
        # conf: parameters from config file
        self.ToolPoses = [[0.45, -0.70, 0.05, 0.0, 1.57, 0],  # right
                          [0.45, 0.0, 0.05, 0.0, 1.57, 0],  # neutral/mid
                          [0.45, 0.70, 0.05, 0.0, 1.57, 0]  # left
                          ]
        self.JointPoses = [
            # regular Orientation
            [1.864, -0.307, 0.675, -0.371, 0.293, -1.569],  # right
            [2.967, -1.102, 2.704, -1.603, 1.396, -1.571],  # middle
            [3.806, -0.161, 0.371, -0.214, 2.235, -1.571],  # left
            # Robo-Lab Orientation, i.e. -Y pointing forward
            [0.279, -0.307, 0.675, -0.371, 0.293, -1.571],  # right RL
            [1.172, -1.041, 2.358, -1.317, 1.187, -1.571],  # middle RL
            [2.381, -0.63, 1.314, -0.737, 2.395, -1.575]  # left RL
        ]
        # get parameters from config file:
        self.xMin = conf["robotXmin"]
        self.xMax = conf["robotXmax"]
        self.yMin = conf["robotYmin"]
        self.yMax = conf["robotYmax"]
        self.frameWidth = conf["frameWidth"]  # width of Video frame in Pixels
        self.frameDepth = conf["frameDepth"]  # Depth of ball Position in Pixels
        self.atransMat = None  # matrix for affine transformation to Robot coord system
        self.getTransformation()  # calculate transformation matrix
        self.HOST = conf["robotHOST"]
        self.PORT = conf["robotPORT"]

        # open socket to robot
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.s.connect((self.HOST, self.PORT))
                print("Socket connected")
                # self.startPos()
                break
            except OSError as msg:
                self.s.close()
                print(f"Socket open failed: {msg}")
                raise EnvironmentError('No connection to Robot')

    def close(self):
        # close socket when program exits
        data = self.s.recv(1024)
        print("Received", repr(data))
        self.s.close()
        print("Socket closed")

    def getTransformation(self):
        # calculate matrix for affine Transformation
        # to transform pixel coords into global Robot coord system
        # can use OpenCV to get the Matrix
        # see: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        # self.atransMat= cv2.getAffineTransform(ptsOrg, ptsRobot)

        # do it manually to avoid dependency on OpenCV at this point
        self.atransMat = np.array(
            [[(self.xMax - self.xMin) / self.frameWidth, 0, self.xMin],
             [0, (self.yMin - self.yMax) / self.frameDepth, self.yMax]]
            # Example: [[ 0.0025,      0.0,     -0.6],
            # [0.0,              -0.02,      -0.2]]
        )
        print("Matrix= ", self.atransMat)

    def startPos(self):
        print("Robot moves into default pose")
        # start "middle" pos
        # self.s.send (("movel([  0.0, -1.745,-2.217, -2.199, -1.623,  0.0 ], a=1.0, v=2.1, t=0.5)" + "\n").encode())
        # time.sleep(t1*2)
        self.move2JointPose(Position.right)
        time.sleep(t1 * 3)
        # self.move2JointPose(Position.rightRL)     # right (Robo-Lab)
        # time.sleep(t1*3)
        self.move2JointPose(Position.middleRL)
        time.sleep(t1 * 2)

    def move2ToolPose(self, pose):
        # linear movement using x,y,z,rx,ry,rz-Coordinates
        print("Move to Tool-Pose ", pose.name)
        cmd = "movel(p" + str(self.ToolPoses[pose.value]) + ", a=2200, v=250)" + "\n"
        print("Cmd: ", cmd)
        self.s.send(cmd.encode())

    def move2JointPose(self, pose):
        # joint (angular) movement to predefined Joint-Pose using joint-angles in RAD!!
        print("Move to Joint-Pose ", pose.name)
        sloMoCmd = ", a={:f}, v={:f})".format(2200.0 / sloMo, 180.0 / sloMo)
        cmd = "movej(" + str(self.JointPoses[pose.value]) + sloMoCmd + "\n"
        print("Cmd: ", cmd)
        self.s.send(cmd.encode())

    def move2ToolYCoord(self, yCoord):
        # move to Coord [0.4, yCoord, 0.05], used in standard orientation of Robot
        # print("movde2ToolYCoord: y=", yCoord)
        self.move2ToolCoord([0.4, yCoord])

    def move2ToolCoord(self, position):
        # linear move to predefined tool-Pose (i.e. using x,y,z,rx,ry,rz-Coordinates)
        print("Move to Tool-Coord: ", position)
        if position is None:
            return None
        if position[0] < self.xMin:
            position[0] = self.xMin
        elif position[0] > self.xMax:
            position[0] = self.xMax
        if position[1] < self.yMin:
            position[1] = self.yMin
        elif position[1] > self.yMax:
            position[1] = self.yMax
        pose = [position[0], position[1], 0.05, 0.0, 1.57, 0]
        sloMoCmd = ", a={:f}, v={:f})".format(2200.0 / sloMo, 120.1 / sloMo)
        cmd = "movel(p" + str(pose) + sloMoCmd + "\n"
        print("Cmd: ", cmd)
        self.s.send(cmd.encode())

    def move2CamCoord(self, pixCoord):
        # linear move to transformed Camera-Coords [x-Pixel, "depth-Pixel"]
        print("Move to Cam-Coord with aff. Transformation: ", pixCoord)
        if pixCoord is None:
            return None
        if pixCoord[0] < 0:
            pixCoord[0] = 0;
        elif pixCoord[0] > self.frameWidth:
            pixCoord[0] = self.frameWidth
        if pixCoord[1] < 0:
            pixCoord[1] = 0;
        elif pixCoord[1] > 20:
            pixCoord[1] = 20
        pixCoord3 = np.array([[pixCoord[0]], [pixCoord[1]], [1]])
        # result= M*pixCood*(transpose)
        robotCoord = self.atransMat.dot(pixCoord3).T
        print("RobotCoord= ", robotCoord)
        # pose= [ robotCoord[0,0], robotCoord[0,1], 0.05, 1.016, 1.321, -1.020]
        # set z-Coord
        pose = [robotCoord[0, 0], robotCoord[0, 1], 0.12, 1.016, 1.321, -1.020]
        # sloMoCmd= ", a=2200.0, v=120.1)"
        sloMoCmd = ", a={:f}, v={:f})".format(2200.0 / sloMo, 120.1 / sloMo)
        cmd = "movel(p" + str(pose) + sloMoCmd + "\n"
        print("Cmd: ", cmd)
        self.s.send(cmd.encode())


if __name__ == "__main__":
    # main routine for testing, when started "stand-alone"
    print("Starting Program Robot Test")
    # get command-line arg, i.e. -c config-file 
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True,
                    help="path to the JSON configuration file")
    arguments = vars(ap.parse_args())
    # get all parameters from config file
    conf = commentjson.load(open(arguments["conf"]))

    # check communication to robot (or robot-simulation)
    try:
        myUrControl = urControl(conf)  # includes moving to middle position
    except EnvironmentError as msg:
        print("Oops: %s" % msg)
    else:
        while True:
            print('Main test loop')
            # robot is alreade in  "middle" position using Robot-Lab orientation
            # enter command (incl "return"-key)
            userInput = input("Cmd [q, a, b, c, d, e, f, g]: ")
            # not all commands work at all time,
            # try: g --> move to middle position
            # try: e --> move to right / middle / left position
            # q: quit program
            # print ('you entered', userInput)
            if userInput == 'q':
                myUrControl.close()
                break;
            elif userInput == 'a':
                myUrControl.move2ToolCoord([0.5, 0.3])
                time.sleep(t1 * 3)
            elif userInput == 'b':
                myUrControl.move2ToolCoord([0.65, 0.6])
                time.sleep(t1 * 3)
            elif userInput == 'c':
                myUrControl.move2ToolCoord([0.3, 0.1])
                time.sleep(t1 * 3)
            elif userInput == 'd':
                myUrControl.move2ToolPose(Position.left)
                time.sleep(t1 * 2)
                myUrControl.move2ToolPose(Position.right)
                time.sleep(t1 * 2)
                myUrControl.move2ToolPose(Position.middle)
                time.sleep(t1)
                myUrControl.move2ToolPose(Position.right)
                time.sleep(t1 * 2)
            elif userInput == 'e':
                myUrControl.move2JointPose(Position.left)
                time.sleep(t3)
                myUrControl.move2JointPose(Position.right)
                time.sleep(t3)
                myUrControl.move2JointPose(Position.middle)
                time.sleep(t3)
                myUrControl.move2JointPose(Position.right)
                time.sleep(t3)
            elif userInput == 'f':
                myUrControl.move2CamCoord([0, 0])
                time.sleep(t3)
                myUrControl.move2CamCoord([myUrControl.frameWidth / 2, myUrControl.frameDepth])
                time.sleep(t3)
                myUrControl.move2CamCoord([myUrControl.frameWidth, 0])
                time.sleep(t3)
                myUrControl.move2CamCoord([20, 20])
                time.sleep(t3)
            elif userInput == 'g':
                myUrControl.move2JointPose(Position.middleRL)
                time.sleep(t3)

    print("Finished Program Robot Test")
    exit()

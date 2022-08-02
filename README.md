# Stereo Setup for low budget Computer Vision

## Introduction
This application was developed as part of a master's thesis at Furtwangen University.
The goal was to build a low cost stereo setup. This application can track a ball in the image section and predict
the trajectory in three-dimensional space. It can also calibrate itself with another coordinate system and thus
transmit the coordinates of the ball to another coordinate system.

The coordinates transmitted are those at which the ball leaves the image at the bottom edge.
## Installation
Version < 3.8 of python is needed.

### Install with pip
To use the application, the entire project must be downloaded and the necessary packages have to be installed.
For this open a terminal in the project's directory and run the command `pip install -r requirements.txt`.

### Install with anaconda
To use the application, the entire project must be downloaded and the necessary packages have to be installed.
For this open a terminal in the project's directory and run the command `conda env create -f environment.yml`.

## Start
Go inside the project's directory and run the command `python ./bin/core.py`

## Usage
The application can be used in three ways.
The first way is to use the application to calibrate the HSV-Range of the ball.
After the parameters are set, you can return into the main menu by pressing q.
The second way is to use the application to calibrate the coordinates from the camera setup with a given set of coordinates from the robot.
After the calibration was successful, the program will return to the main menu.
The third way is to use the application to predict the trajectory of the ball.

Disclaimer: The OpenCV window will freeze everytime the application returns to the main menu. This is a normal behavior.



## License
This project is licensed under the MIT License.

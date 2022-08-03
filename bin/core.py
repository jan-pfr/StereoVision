import ast
import configparser
import re

import numpy as np

from AppState import AppState
from coordCal.coordCalibration import CoordCalibration
from hsvCal.hsvCalibration import HSVRangeCalibration
from oT.objectTracking import ObjectTracking


# Source: https://stackoverflow.com/questions/2835559/parsing-values-from-a-config-file-in-python
def parse_int_tuple(s: str):
    """
    Parse a string into a tuple of ints.
    :param s: String to parse.
    :return:
    """
    return tuple(int(i.strip()) for i in s[1:-1].split(','))


def str2array(s):
    """
    Remove space after [
    Replace commas and spaces

    :param s: String to parse.
    :return:
    """
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


class Application:

    def __init__(self):
        """
        Import all necessary params and initiate variables.

        """

        # define program state
        self.appState = AppState.STARTUP

        # Configure parser so it is able to parse tuples
        self.config = configparser.ConfigParser(converters={'tuple': parse_int_tuple, 'array': str2array})
        self.config.read('./config/config.ini')

    def main(self):
        """
        Main loop of the application.

        :return:
        """

        self.update_config()

        while True:
            print('Welcome to the StereoVision Setup. \n'
                  'You have to setup the coordination system calibration first. \n'
                  'You can choose:\n'
                  '1: Start the Stereo Vision setup\n'
                  '2: Start the coordinate system calibration\n'
                  '3: Start the HSV calibration\n'
                  'q: Quit the Application.')

            u_input = input('Type here: ')
            if u_input == 'q':
                self.appState = AppState.CLOSESTATE
                break
            elif int(u_input) == 1:
                self.appState = AppState.NORMALSTATE
                break
            elif int(u_input) == 2:
                self.appState = AppState.COORDCALIBRATION
                break
            elif int(u_input) == 3:
                self.appState = AppState.HSVCALIBRATION
                break
            else:
                print('No valid user input, try again.')

    def __del__(self):
        pass

    def update_config(self):
        self.config.read('./config/config.ini')


if __name__ == "__main__":

    app = Application()
    while True:
        app.main()

        if app.appState == AppState.NORMALSTATE:
            object_tracking = ObjectTracking(app.config)
            object_tracking.start()
            del object_tracking
        elif app.appState == AppState.HSVCALIBRATION:
            hsv = HSVRangeCalibration(app.config)
            hsv.start()
            del hsv
        elif app.appState == AppState.COORDCALIBRATION:
            coord_calibration = CoordCalibration(app.config)
            coord_calibration.start()
            del coord_calibration
        elif app.appState == AppState.CLOSESTATE:
            del app
            break

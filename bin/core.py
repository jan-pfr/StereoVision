import configparser
from bin.object_tracking.objectTracking import ObjectTracking
from AppState import AppState
from bin.hsv_calibration.hsvCalibration import HSVRangeCalibration


def parse_int_tuple(s: str):
    return tuple(int(i.strip()) for i in s[1:-1].split(','))


class Application:

    def __init__(self):
        """
        Import all necessary params and initiate variables.

        """

        # define program state
        self.appState = AppState.STARTUP

        # Configure parser so it is able to parse tuples
        self.config = configparser.ConfigParser(converters={'tuple': parse_int_tuple})
        self.config.read('./config/config.ini')

    def main(self) -> AppState:

        self.update_config()

        while True:
            print('Welcome to the bin Setup. You can choose:\n'
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
            print('tbd')
        elif app.appState == AppState.CLOSESTATE:
            del app
            break

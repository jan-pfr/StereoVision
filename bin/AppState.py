from enum import Enum


class AppState(Enum):
    STARTUP = 0
    MAINLOOP = 99
    CLOSESTATE = 100
    OBJECTTRACKING = 1
    HSVCALIBRATION = 2
    COORDCALIBRATION = 3

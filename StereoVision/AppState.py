from enum import Enum


class AppState(Enum):
    STARTUP = 0
    NORMALSTATE = 1
    HSVCALIBRATION = 2
    COORDCALIBRATION = 3
    CLOSESTATE = 4

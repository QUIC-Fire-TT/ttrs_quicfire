"""
Created on Wed Jun 15 12:48:46 2022

@author: cbonesteel
"""
class WindDirOutOfRange(Exception):
    """
    Exception raised when the wind direction is out of range.
    Default range is [0, 360).

    Parameters
    ----------
    dir - The wind direction
    range - The wind direction range
    message - An explanation of the error
    """

    def __init__(self, dir, range='[0, 360)', message='Wind Direction is Not in Range'):
        self.dir = dir
        self.range = range
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dir} -> {self.message} {self.range}'

class WindSpeedOutOfRange(Exception):
    """
    Exception raised when the wind speed is out of range.
    Default range is (0,20].

    Parameters
    ----------
    speed - The wind speed
    range - The wind speed range
    message - An explanation of the error
    """

    def __init__(self, speed, range='(0,20]', message='Wind Speed is Not in Range'):
        self.speed = speed
        self.range = range
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.speed} -> {self.message} {self.range}'

class DataLengthMismatch(Exception):
    """
    Exception raised when two data lengths that need to match don't.

    Parameters
    ----------
    name1 - The name of the first set of data
    len1 - The length of the first set of data
    name2 - The name of the second set of data
    len2 - The length of the second set of data
    message - An explanation of the error
    """

    def __init__(self, name1, len1, name2, len2, message='Data Lengths Do Not Match'):
        self.name1 = name1
        self.len1 = len1
        self.name2 = name2
        self.len2 = len2
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.name1} ({self.len1}), {self.name2} ({self.len2}) -> {self.message}'
from abc import ABC, abstractmethod
from enum import Enum, auto

# Using this abstract base class definition allows software testing of a "dummy" camera when no hardware is available
class CameraModule(ABC):
    __resolutionsModeA__ = [(40, 30), (800, 600)] # 4:3 aspect ratio
    __resolutionsModeB__ = [(45, 30), (900, 600)] # 3:2 aspect ratio
    __resolutionA__ = __resolutionsModeA__[0]
    __resolutionB__ = __resolutionsModeB__[0]

    __active__ = False

    __singleImageReady__ = False
    __streamImageReady__ = False
    __singleImage__ = None
    __streamImage__ = None

    __singleCaptureCallback__ = None
    __streamCaptureCallback__ = None

    __exposureTime__ = 20
    __gain__ = 1

    # set to None if no callback desired
    def setSingleCaptureCallback(self, callback):
        self.__singleCaptureCallback__ = callback

    # set to None if no callback desired
    def setStreamCaptureCallback(self, callback):
        self.__streamCaptureCallback__ = callback

    # camera configuration functions (e.g. 'exposure_time', 'gain', 'image_mode', etc.)
    def getSetting(self, settingName):
        match settingName:
            case 'exposure_time':
                return self.__exposureTime__
            case 'gain':
                return self.__gain__
            case other:
                return None

    def setSetting(self, settingName, settingValue): # returns true on success
        result = False

        match settingName:
            case 'exposure_time':
                if (isinstance(settingValue, int) or isinstance(settingValue, float)) and settingValue >= 0:
                    self.__exposureTime__ = settingValue
                    result = True
            case 'gain':
                if (isinstance(settingValue, int) or isinstance(settingValue, float)) and settingValue >= 0:
                    self.__gain__ = settingValue
                    result = True

        return result

    def getAvailableResolutions(self, mode=None):
        match mode:
            case 'B':
                return self.__resolutionsModeB__
            case other:
                return self.__resolutionsModeA__

    def getResolution(self, mode=None):
        match mode:
            case 'B':
                return self.__resolutionB__
            case other:
                return self.__resolutionA__

    def setResolution(self, resolution, mode=None): # returns true on success
        match mode:
            case 'B':
                if resolution in self.__resolutionsModeB__:
                    self.__resolutionB__ = resolution
                    self.__streamImageReady__ = False
                    return True
            case other:
                if resolution in self.__resolutionsModeA__:
                    self.__resolutionA__ = resolution
                    self.__singleImageReady__ = False
                    return True
        return False

    # camera interfacing functions
    def streamImageReady(self): # returns bool
        return self.__streamImageReady__

    def singleImageReady(self): # returns bool
        return self.__singleImageReady__

    def getSingleCaptureImage(self):
        if singleImageReady():
            return (self.__singleImage__, self.__resolutionA__, 'RGB888')
        else:
            return None

    def getStreamCaptureImage(self):
        if streamImageReady():
            return (self.__streamImage__, self.__resolutionB__, 'RGB888')
        else:
            return None

    def isOpen(self): # returns bool
        return self.__active__

    def open(self): # returns true on success
        self.__active__ = True
        return True
        
    def close(self): # returns true on success
        self.__active__ = False
        self.__singleImageReady__ = False
        self.__streamImageReady__ = False
        del self.__singleImage__
        del self.__streamImage__
        return True

    def startSingleCapture(self): # returns true on success
        if not self.__active__:
            return False

        self.__singleImageReady__ = False
        self.__singleImage__ = [0x00, 0xBF, 0xFF] * (self.__resolutionA__[0] * self.__resolutionA__[1])
        self.__singleImageReady__ = True

        if self.__singleCaptureCallback__ is not None:
            self.__singleCaptureCallback__(self.__singleImage__, self.__resolutionA__, 'RGB888')

        return True

    def startStreamCapture(self, iterations=10): # returns true on success
        if not self.__active__:
            return False

        for i in range(0, iterations):
            self.__streamImageReady__ = False
            self.__streamImage__ = [0x00, 0xFF, 0xFF] * (self.__resolutionB__[0] * self.__resolutionB__[1])
            self.__streamImageReady__ = True

            if self.__streamCaptureCallback__ is not None:
                self.__streamCaptureCallback__(self.__streamImage__, self.__resolutionB__, 'RGB888')

        return True

    def stopSingleCapture(self): # returns true on success
        return True

    def stopStreamCapture(self): # returns true on success
        return True

    # camera description function (e.g. name, vendor, model number, etc.)
    def getDeviceInfo(self, parameterName):
        match parameterName:
            case 'name':
                return "DummyCamera"
            case other:
                return None

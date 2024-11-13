# J. Kent Wirant
# Hacker Fab
# Amscope Camera Module

from camera.camera_module import *
import camera.amscope.amcam as amcam
import time

class AmscopeCamera(CameraModule):

    camera = None

    liveIndex = 0
    liveData = None

    stillIndex = 0
    stillData = None

    liveImageGood = False
    stillImageGood = False

    __resolutionModes = {'stream': None, 'single': None}


    def __init__(self):
        self.singleCaptureCallback = None
        self.streamCaptureCallback = None
        self.close() # reset to known state


    def __del__(self):
        self.close() # reset to known state


    def streamImageReady(self):
        return self.liveImageGood


    def singleImageReady(self):
        return self.stillImageGood


    def isOpen(self):
        return self.camera != None


    def open(self):
        if self.isOpen():
            return True

        self.camera = amcam.Amcam.Open(None)

        if self.camera == None:
            return False

        self.setResolution(self.getAvailableResolutions('stream')[0], 'stream')
        self.setResolution(self.getAvailableResolutions('single')[0], 'single')
        return True


    def close(self):
        if self.isOpen():
            self.camera.Close()
        
        self.stillData = None
        self.liveData = None
        self.camera = None
        self.liveImageGood = False
        self.stillImageGood = False


    def startSingleCapture(self):
        self.stillImageGood = False
        self.camera.StartPullModeWithCallback(self.staticCallback, self)
        self.camera.Snap(self.stillIndex)
        return True


    def startStreamCapture(self):
        self.liveImageGood = False
        self.camera.StartPullModeWithCallback(self.staticCallback, self)
        return True


    @staticmethod
    def staticCallback(nEvent, ctx):
        ctx.amscopeCallback(nEvent)


    def amscopeCallback(self, nEvent):
        if (nEvent == amcam.AMCAM_EVENT_IMAGE):
            self.liveImageGood = False
            self.camera.PullImageV2(self.liveData, 24, None)
            self.liveImageGood = True


            if self.streamCaptureCallback != None:
                r = self.getResolution('stream')
                self.streamCaptureCallback(self.liveData, r, 'RGB888')

        elif nEvent == amcam.AMCAM_EVENT_STILLIMAGE:
            self.stillImageGood = False
            self.camera.PullStillImageV2(self.stillData, 24, None)
            self.stillImageGood = True

            if self.singleCaptureCallback != None:
                r = self.getResolution('single')
                self.singleCaptureCallback(self.stillData, r, 'RGB888')

        else: # for more robust operation, add more event handlers here
            pass


    def getSingleCaptureImage(self):
        if not self.stillImageReady():
            return None
        r = self.getResolution('single')
        img = bytes(r[0] * r[1])
        self.__copyImage(self.stillData, img, r[0], r[1],r[0], r[1], 'RGB888')
        return img


    def getStreamCaptureImage(self):
        if not self.liveImageReady():
            return None
        r = self.getResolution('stream')
        img = bytes(r[0] * r[1])
        self.__copyImage(self.liveData, img, r[0], r[1], r[0], r[1], 'RGB888')
        return img


    def setSingleCaptureCallback(self, callback):
        self.singleCaptureCallback = callback


    def setStreamCaptureCallback(self, callback):
        self.streamCaptureCallback = callback


    # helper function
    def __copyImage(srcData, destData, ws, hs, wd, hd, imageFormat):
        minWidth = ws if (ws < wd) else wd
        minHeight = hs if (hs < hd) else hd

        if True: # normally would check for format here; this assumes 24 bits per pixel
            # copy image data
            for y in range(0, minHeight):
                for x in range(0, minWidth):
                    destData[3*(wd*y + x)] = srcData[3*(ws*y + x)]
                    destData[3*(wd*y + x) + 1] = srcData[3*(ws*y + x) + 1]
                    destData[3*(wd*y + x) + 2] = srcData[3*(ws*y + x) + 2]

            # if there's extrself*(wd*y + x) + 1] = 0
                    destData[3*(wd*y + x) + 2] = 0

            # if there's extra space at the bottom of the destination, set its data to 0
            for y in range(minHeight, hd):
                for x in range(0, wd):
                    destData[3*(wd*y + x)] = 0
                    destData[3*(wd*y + x) + 1] = 0
                    destData[3*(wd*y + x) + 2] = 0

    
    # TODO: implement separate single and stream implementations
    def getAvailableResolutions(self, mode=None):
        if mode != 'stream' or mode != 'single':
            mode = 'stream'

        if self.__resolutionModes['stream'] == None:
            num_resolutions = self.camera.ResolutionNumber()
            self.__resolutionModes['stream'] = [None] * num_resolutions

            for i in range(0, num_resolutions):
                self.__resolutionModes['stream'][i] = self.camera.get_Resolution(i)

        if self.__resolutionModes['single'] == None:
            num_resolutions = self.camera.StillResolutionNumber()
            self.__resolutionModes['single'] = [None] * num_resolutions

            for i in range(0, num_resolutions):
                self.__resolutionModes['single'][i] = self.camera.get_StillResolution(i)

        return self.__resolutionModes[mode]
    

    def getResolution(self, mode=None):
        if mode != 'stream' or mode != 'single':
            mode = 'stream'
        return self.__resolutionModes[mode][self.liveIndex]


    def setResolution(self, resolution, mode=None):
        if mode != 'stream' or mode != 'single':
            mode = 'stream'

        for i in range(0, len(self.__resolutionModes[mode])):
            if self.__resolutionModes[mode][i] is resolution or self.__resolutionModes[mode][i] == resolution:
                if mode == 'stream':
                    self.liveImageGood = False
                    self.liveIndex = i
                    self.liveData = bytes(resolution[0] * resolution[1] * 3)
                elif mode == 'single':
                    self.stillImageGood = False
                    self.stillIndex = i
                    self.stillData = bytes(resolution[0] * resolution[1] * 3)

                self.camera.put_Size(resolution[0], resolution[1])

                return True
        return False


    # camera description functions
    def getDeviceInfo(self, parameterName):
        match parameterName:
            case 'vendor':
                return self.camera.AmcamDeviceV2.displayname
            case 'name':
                return self.camera.AmcamModelV2.name
            case other:
                return None


# test suite
if __name__ == "__main__":
    def testCallback(image, resolution, format):
        print(f"{resolution[0]} {resolution[1]} {format}")

    def testCase(testName, expectedValues, actualValues):
        global testCount
        global testsPassed
        global testPrefixString

        errorStringLimit = 100

        # convert inputs to lists if necessary
        if not isinstance(expectedValues, list):
            expectedValues = [expectedValues]
        if not isinstance(actualValues, list):
            actualValues = [actualValues]
        assert len(expectedValues) == len(actualValues)

        testCount += 1
        success = True
        errorString = "(expected, got) = ["

        for i in range(0, len(expectedValues)):
            if expectedValues[i] is not actualValues[i] or expectedValues[i] != actualValues[i]:
                success = False
                if(len(errorString) <= errorStringLimit):
                    errorString += f"({expectedValues[i]}, {actualValues[i]}); "
                if(len(errorString) > errorStringLimit):
                    errorString = errorString[0:errorStringLimit] + "..."

        if success:
            testsPassed += 1
            print(testPrefixString + "PASSED Test '" + testName + "'")
        else:
            if(len(errorString) > errorStringLimit):
                errorString = errorString + "]"
            else:
                errorString = errorString[:-2] + "]"
            print(testPrefixString + "FAILED Test '" + testName + "': " + errorString)

        return success


    testCount = 0
    testsPassed = 0
    testPrefixString = "[AmscopeCamera] "

    camera = AmscopeCamera()
    openSuccess = camera.open()
    testCase("open()", True, openSuccess)

    if not openSuccess:
        print(testPrefixString + "Further testing requires connection to Amscope Camera")
    else:
        print(testPrefixString + camera.getDeviceInfo('name'))
        print(testPrefixString + camera.getDeviceInfo('vendor'))
        
        resolutions = camera.getAvailableResolutions()
        print(testPrefixString + "Resolutions: ", end='')

        expectedResolutionMode = []
        actualResolutionMode = []
        
        for r in resolutions:
            print(str(r) + " ", end='')
            camera.setResolution(r)
            expectedResolutionMode.append(r)
            actualResolutionMode.append(camera.getResolution())
        print()
        testCase("setResolution(ResolutionMode)", expectedResolutionMode, actualResolutionMode)

        camera.setStreamCaptureCallback(testCallback)
        camera.startStreamCapture()
        time.sleep(5)

    print(testPrefixString + f"Result: {testsPassed}/{testCount} tests passed")

from camera.camera_module import CameraModule
from PIL import Image
import threading
import cv2
import time
import numpy

from pypylon import pylon
from pypylon import genicam

class BaslerPylon(CameraModule):
    def __init__(self):
        # Create an instant camera object with the camera device found first.
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        self.capture_thread = None
        self.should_stop = threading.Event()
    
    def open(self):
        self.camera.Open()

        # Print the model name of the camera.
        print("Using device ", self.camera.GetDeviceInfo().GetModelName())


        # demonstrate some feature access
        new_width = self.camera.Width.Value - self.camera.Width.Inc
        if new_width >= self.camera.Width.Min:
            self.camera.Width.Value = new_width

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        self.converter = pylon.ImageFormatConverter()

        self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return True

    def close(self):
        self.should_stop.set()
        if self.camera.IsOpen():
            self.camera.StopGrabbing()
            self.camera.Close()
        if self.capture_thread is not None:
            self.capture_thread.join()
            self.capture_thread = None

        return True
    
    def startStreamCapture(self):
        self.should_stop.clear()

        def capture_thread():
            while self.camera.IsGrabbing() and not self.should_stop.is_set():
                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                # Image grabbed successfully?
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    image = self.converter.Convert(grabResult)
                    frame = image.GetArray()
                    assert self.__streamCaptureCallback__ is not None
                    self.__streamCaptureCallback__(frame, frame.size, 'RGB888')
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                grabResult.Release()
 
            print('Exited the loop!')

        if self.capture_thread is None and self.__streamCaptureCallback__ is not None:
            self.capture_thread = threading.Thread(target=capture_thread)
            self.capture_thread.start()
            return True
        
        return False

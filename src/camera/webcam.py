from camera.camera_module import CameraModule
from PIL import Image
import threading
import cv2
import time

class Webcam(CameraModule):
    def __init__(self):
        self.camera = None
        self.capture_thread = None
        self.should_stop = threading.Event()
    
    def open(self):
        self.camera = cv2.VideoCapture(1)
        return self.camera.isOpened()
    
    def close(self):
        self.should_stop.set()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.capture_thread is not None:
            self.capture_thread.join()
            self.capture_thread = None

        return True
    
    def startStreamCapture(self):
        self.should_stop.clear()

        def capture_thread():
            assert self.camera is not None
            assert self.__streamCaptureCallback__ is not None
            while not self.should_stop.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.__streamCaptureCallback__(frame, frame.size, 'RGB888')
            print('Exited the loop!')

        if self.capture_thread is None and self.__streamCaptureCallback__ is not None:
            self.capture_thread = threading.Thread(target=capture_thread)
            self.capture_thread.start()
            return True
        
        return False
import threading
import time
from typing import Optional

import cv2
import numpy as np

from camera.camera_module import CameraModule
from core.events import Event


class Webcam(CameraModule):
    """Generic USB webcam interface using OpenCV VideoCapture."""

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index
        self.camera: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._exposure_time: Optional[float] = None

    def __del__(self):
        self.close()

    def is_open(self) -> bool:
        return self.camera is not None and self.camera.isOpened()

    def open(self) -> bool:
        if self.is_open():
            return True

        self.camera = cv2.VideoCapture(self.index)
        if not self.camera.isOpened():
            self.camera = None
            return False

        # Grab an initial frame synchronously so get_latest_frame is immediately ready
        ok, frame = self.camera.read()
        if ok and frame is not None:
            with self._lock:
                self._latest_frame = frame

        self.should_stop.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="WebcamCaptureThread", daemon=True
        )
        self.capture_thread.start()
        return True

    def close(self) -> bool:
        self.should_stop.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        with self._lock:
            self._latest_frame = None

        return True

    def _capture_loop(self):
        while not self.should_stop.is_set():
            if self.camera is None or not self.camera.isOpened():
                break

            ok, frame = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame

            if self.event_bus is not None:
                self.event_bus.emit(Event.CAMERA_FRAME_READY, frame)

            if self._stream_callback is not None:
                try:
                    self._stream_callback(frame, frame.size, "BGR888")
                except Exception as e:
                    print(f"Webcam stream callback error: {e}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if not self.is_open():
            if not self.open():
                return None

        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def set_exposure_time(self, value: float) -> bool:
        self._exposure_time = float(value)
        if self.camera is not None and self.camera.isOpened():
            try:
                success = self.camera.set(cv2.CAP_PROP_EXPOSURE, value)
                return bool(success)
            except Exception as e:
                print(f"Could not set exposure property: {e}")
                return False
        return False

    def get_exposure_time(self) -> Optional[float]:
        return self._exposure_time

    def startStreamCapture(self) -> bool:
        if not self.is_open():
            return self.open()
        return True

    def stopStreamCapture(self) -> bool:
        return True

    def get_device_info(self, parameter_name: str) -> Optional[str]:
        match parameter_name:
            case "name":
                return f"Webcam_{self.index}"
            case "vendor":
                return "OpenCV"
            case other:
                return None

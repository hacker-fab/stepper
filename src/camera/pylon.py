import threading
import time
from typing import Optional

import numpy as np

try:
    from pypylon import pylon
except ImportError:
    pylon = None

from camera.camera_module import CameraModule
from core.events import Event


class BaslerPylon(CameraModule):
    """Basler industrial camera interface using PyPylon."""

    def __init__(self, index: int = 0):
        super().__init__()
        if pylon is None:
            raise RuntimeError(
                "Failed to initialize Basler camera: 'pypylon' is not installed. "
                "Install with: pip install '.[basler]' or pip install pypylon"
            )

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        if index > len(devices) - 1:
            print(f"Could not find baslerpylon camera with index {index}")
            self.camera = None
            return

        self.camera = pylon.InstantCamera(
            tl_factory.CreateDevice(devices[index])
        )

        self.capture_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self.converter = None

    def __del__(self):
        self.close()

    def is_open(self) -> bool:
        return self.camera is not None and self.camera.IsOpen()

    def open(self) -> bool:
        if self.camera is None:
            return False
        if self.is_open():
            return True

        self.camera.Open()
        print("Using device ", self.camera.GetDeviceInfo().GetModelName())

        self.camera.ExposureTime.Value = 8333.0
        self.camera.AcquisitionFrameRate.Value = 30.0

        new_width = self.camera.Width.Value - self.camera.Width.Inc
        if new_width >= self.camera.Width.Min:
            self.camera.Width.Value = new_width

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.should_stop.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="PylonCaptureThread", daemon=True
        )
        self.capture_thread.start()
        return True

    def close(self) -> bool:
        self.should_stop.set()
        if self.camera is not None and self.camera.IsOpen():
            print("Stopping camera")
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
            except Exception as e:
                print(f"Error closing Basler camera: {e}")
            print("Closed camera")

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
            print("Joined capture thread")

        with self._lock:
            self._latest_frame = None

        return True

    def _capture_loop(self):
        while self.camera is not None and self.camera.IsGrabbing() and not self.should_stop.is_set():
            try:
                grabResult = self.camera.RetrieveResult(
                    1000, pylon.TimeoutHandling_Return
                )
            except Exception as e:
                if self.should_stop.is_set():
                    break
                print(f"Pylon grab exception: {e}")
                continue

            if grabResult is None:
                continue

            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                frame = image.GetArray()
                with self._lock:
                    self._latest_frame = frame

                if self.event_bus is not None:
                    self.event_bus.emit(Event.CAMERA_FRAME_READY, frame)

                if self._stream_callback is not None:
                    try:
                        self._stream_callback(frame, frame.size, "BGR888")
                    except Exception as e:
                        print(f"Pylon stream callback error: {e}")
            else:
                if not self.should_stop.is_set():
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()

        print("Exited Pylon capture loop")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if not self.is_open():
            if not self.open():
                return None

        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def set_exposure_time(self, value: float) -> bool:
        if self.camera is not None and self.camera.IsOpen():
            try:
                self.camera.ExposureTime.Value = float(value)
                return True
            except Exception as e:
                print(f"Failed to set exposure time: {e}")
                return False
        return False

    def get_exposure_time(self) -> Optional[float]:
        if self.camera is not None and self.camera.IsOpen():
            try:
                return float(self.camera.ExposureTime.Value)
            except Exception:
                return None
        return None

    def startStreamCapture(self) -> bool:
        if not self.is_open():
            return self.open()
        return True

    def stopStreamCapture(self) -> bool:
        return True

    def get_device_info(self, parameter_name: str) -> Optional[str]:
        if self.camera is None:
            return None
        try:
            device_info = self.camera.GetDeviceInfo()
            match parameter_name:
                case "name":
                    return device_info.GetModelName()
                case "vendor":
                    return device_info.GetVendorName()
                case other:
                    return None
        except Exception:
            return None

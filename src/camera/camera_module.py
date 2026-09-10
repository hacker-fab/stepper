from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np


class CameraModule(ABC):
    """Abstract base class defining the common camera interface.

    Supports industrial cameras (e.g. Basler Pylon) and standard USB webcams.
    """

    def __init__(self):
        self._stream_callback: Optional[Callable[[np.ndarray, int, str], None]] = None

    @abstractmethod
    def open(self) -> bool:
        """Open the camera connection. Returns True on success."""
        pass

    @abstractmethod
    def close(self) -> bool:
        """Close the camera connection and release resources. Returns True on success."""
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the camera connection is open and active."""
        pass

    def isOpen(self) -> bool:
        """Backwards compatibility alias for is_open()."""
        return self.is_open()

    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent camera frame as a numpy array (BGR format), or None."""
        pass

    def set_exposure_time(self, value: float) -> bool:
        """Set exposure time in microseconds. Returns True if supported and successful."""
        return False

    def setExposureTime(self, value: float) -> bool:
        """Backwards compatibility alias for set_exposure_time()."""
        return self.set_exposure_time(value)

    def get_exposure_time(self) -> Optional[float]:
        """Get the current exposure time in microseconds, or None if unsupported."""
        return None

    def setStreamCaptureCallback(
        self, callback: Optional[Callable[[np.ndarray, int, str], None]]
    ) -> None:
        """Set callback for streaming frames: callback(frame, size, format)."""
        self._stream_callback = callback

    def startStreamCapture(self) -> bool:
        """Start streaming capture. Returns True on success."""
        if not self.is_open():
            return self.open()
        return True

    def stopStreamCapture(self) -> bool:
        """Stop streaming capture."""
        return True

    def getDeviceInfo(self, parameterName: str) -> Optional[str]:
        """Return device information string for parameterName (e.g. 'name', 'vendor')."""
        return None

    def getSetting(self, setting_name: str) -> Any:
        """Backwards compatibility for camera settings."""
        if setting_name == "exposure_time":
            return self.get_exposure_time()
        return None

    def setSetting(self, setting_name: str, setting_value: Any) -> bool:
        """Backwards compatibility for camera settings."""
        if setting_name == "exposure_time":
            return self.set_exposure_time(setting_value)
        return False


class DummyCamera(CameraModule):
    """Simulated camera generating synthetic test patterns for testing without hardware."""

    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
        self.width = width
        self.height = height
        self._active = False
        self._frame_count = 0
        self._exposure_time = 20000.0

    def open(self) -> bool:
        self._active = True
        return True

    def close(self) -> bool:
        self._active = False
        return True

    def is_open(self) -> bool:
        return self._active

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if not self._active:
            return None
        self._frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        shift = (self._frame_count * 4) % self.width
        frame[:, :, 0] = np.linspace(0, 255, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 255, self.height, dtype=np.uint8).reshape(-1, 1)
        frame[:, shift : min(shift + 20, self.width), 2] = 255
        return frame

    def set_exposure_time(self, value: float) -> bool:
        self._exposure_time = float(value)
        return True

    def get_exposure_time(self) -> Optional[float]:
        return self._exposure_time

    def getDeviceInfo(self, parameterName: str) -> Optional[str]:
        if parameterName == "name":
            return "DummyCamera"
        return None

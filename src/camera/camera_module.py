import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple
import cv2
import numpy as np
from core.engine_module import EngineModule
from core.events import CameraFlipMode, Event


class CameraModule(EngineModule):
    """Abstract base class defining the common camera interface.

    Supports industrial cameras (e.g. Basler Pylon) and standard USB webcams.
    Automatically applies configured camera flip mode to all frames.
    """

    def __init__(self):
        super().__init__()
        self._stream_callback: Optional[Callable[[np.ndarray, int, str], None]] = None
        self.flip_mode: str = CameraFlipMode.NONE.value
        self._primary_crosshair: Tuple[float, float] = (0.5, 0.5)
        self._secondary_crosshair: Optional[Tuple[float, float]] = None
        self._last_frame_size: Tuple[int, int] = (1920, 1080)

    def set_flip_mode(self, mode: str) -> None:
        """Sets the camera flip mode ('none', 'horizontal', 'vertical', 'rotate_180')."""
        normalized = str(mode).lower().strip()
        if normalized in ("none", "horizontal", "vertical", "rotate_180"):
            self.flip_mode = normalized
        else:
            self.flip_mode = CameraFlipMode.NONE.value

    def _apply_flip(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Applies the configured camera flip mode to a raw camera frame."""
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        self._last_frame_size = (w, h)
        if self.flip_mode == CameraFlipMode.HORIZONTAL.value:
            return cv2.flip(frame, 1)
        elif self.flip_mode == CameraFlipMode.VERTICAL.value:
            return cv2.flip(frame, 0)
        elif self.flip_mode == CameraFlipMode.ROTATE_180.value:
            return cv2.flip(frame, -1)
        return frame

    def set_primary_crosshair(self, u: float, v: float) -> None:
        """Sets normalized primary crosshair position (u, v) in [0.0, 1.0]."""
        self._primary_crosshair = (
            max(0.0, min(1.0, float(u))),
            max(0.0, min(1.0, float(v))),
        )

    def get_primary_crosshair(self) -> Tuple[float, float]:
        """Returns normalized primary crosshair position (u, v) in [0.0, 1.0]."""
        return self._primary_crosshair

    def set_secondary_crosshair(self, pos: Optional[Tuple[float, float]]) -> None:
        """Sets normalized secondary crosshair position (u, v) or None."""
        if pos is None:
            self._secondary_crosshair = None
        else:
            self._secondary_crosshair = (
                max(0.0, min(1.0, float(pos[0]))),
                max(0.0, min(1.0, float(pos[1]))),
            )

    def get_secondary_crosshair(self) -> Optional[Tuple[float, float]]:
        """Returns normalized secondary crosshair position (u, v) or None."""
        return self._secondary_crosshair

    def get_primary_crosshair_offset_px(self) -> Tuple[float, float]:
        """Returns primary crosshair offset (dx_px, dy_px) from the camera frame center."""
        w, h = self._last_frame_size
        u, v = self._primary_crosshair
        return ((u - 0.5) * float(w), (v - 0.5) * float(h))

    def get_primary_crosshair_offset_um(self, camera_pixel_size: float) -> Tuple[float, float]:
        """Returns primary crosshair offset (dx_um, dy_um) from camera center in µm."""
        dx_px, dy_px = self.get_primary_crosshair_offset_px()
        return (dx_px * float(camera_pixel_size), dy_px * float(camera_pixel_size))

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

    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent camera frame as a numpy array (BGR format), or None."""
        pass

    def set_exposure_time(self, value: float) -> bool:
        """Set exposure time in microseconds. Returns True if supported and successful."""
        return False

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

    def get_device_info(self, parameter_name: str) -> Optional[str]:
        """Return device information string for parameter_name (e.g. 'name', 'vendor')."""
        return None


class DummyCamera(CameraModule):
    """Simulated camera generating synthetic test patterns for testing without hardware."""

    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
        self.width = width
        self.height = height
        self._active = False
        self._frame_count = 0
        self._exposure_time = 20000.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None

    def open(self) -> bool:
        self._active = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="DummyCaptureThread", daemon=True
        )
        self._thread.start()
        return True

    def close(self) -> bool:
        self._active = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
        return True

    def is_open(self) -> bool:
        return self._active

    def _generate_frame(self) -> np.ndarray:
        self._frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        shift = (self._frame_count * 4) % self.width
        frame[:, :, 0] = np.linspace(0, 255, self.width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 255, self.height, dtype=np.uint8).reshape(-1, 1)
        frame[:, shift : min(shift + 20, self.width), 2] = 255
        return frame

    def _capture_loop(self):
        while not self._stop_event.is_set():
            raw_frame = self._generate_frame()
            with self._lock:
                self._latest_frame = raw_frame
            frame = self._apply_flip(raw_frame)
            if self.event_bus is not None and frame is not None:
                self.event_bus.emit(Event.CAMERA_FRAME_READY, frame)
            if self._stream_callback is not None and frame is not None:
                try:
                    self._stream_callback(frame, frame.size, "BGR888")
                except Exception as e:
                    print(f"Dummy stream callback error: {e}")
            time.sleep(0.033)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if not self._active:
            return None
        with self._lock:
            if self._latest_frame is not None:
                return self._apply_flip(self._latest_frame.copy())
        raw_frame = self._generate_frame()
        with self._lock:
            self._latest_frame = raw_frame
        frame = self._apply_flip(raw_frame)
        if self.event_bus is not None and frame is not None:
            self.event_bus.emit(Event.CAMERA_FRAME_READY, frame)
        return frame

    def set_exposure_time(self, value: float) -> bool:
        self._exposure_time = float(value)
        return True

    def get_exposure_time(self) -> Optional[float]:
        return self._exposure_time

    def get_device_info(self, parameter_name: str) -> Optional[str]:
        if parameter_name == "name":
            return "DummyCamera"
        return None

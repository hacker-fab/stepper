"""Camera subsystem providing dynamic driver discovery and factory instantiation."""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

from camera.camera_module import CameraModule, DummyCamera


def _check_webcam() -> tuple[bool, Optional[str]]:
    try:
        import cv2
        return True, None
    except (ImportError, ModuleNotFoundError):
        return False, "Missing 'opencv-python'. Install with: pip install opencv-python"


def _check_basler() -> tuple[bool, Optional[str]]:
    try:
        import pypylon
        return True, None
    except (ImportError, ModuleNotFoundError):
        return False, "Missing optional package 'pypylon'. Install with: pip install '.[basler]' or pip install pypylon"


def _check_flir() -> tuple[bool, Optional[str]]:
    try:
        spec = importlib.util.find_spec("camera.flir.flir_camera")
        if spec is None:
            return False, "FLIR camera submodule not initialized. Clone flir-private into src/camera/flir"
        import camera.flir.flir_camera  # type: ignore
        return True, None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False, "FLIR camera submodule not initialized. Clone flir-private into src/camera/flir"
    except Exception as e:
        return False, f"FLIR camera driver error: {e}"


def _check_amscope() -> tuple[bool, Optional[str]]:
    try:
        import camera.amscope.amscope_camera  # type: ignore
        return True, None
    except (ModuleNotFoundError, ImportError) as e:
        return False, f"AmScope camera module not available: {e}"
    except Exception as e:
        return False, f"AmScope camera driver error: {e}"


CAMERA_REGISTRY: dict[str, dict[str, Any]] = {
    "none": {
        "description": "No camera (disabled)",
        "check": lambda: (True, None),
    },
    "webcam": {
        "description": "Generic USB camera (OpenCV)",
        "check": _check_webcam,
    },
    "basler": {
        "description": "Basler / Pylon industrial camera",
        "check": _check_basler,
    },
    "pylon": {
        "description": "Basler / Pylon industrial camera (alias)",
        "check": _check_basler,
    },
    "flir": {
        "description": "FLIR Machine Vision camera",
        "check": _check_flir,
    },
    "amscope": {
        "description": "AmScope optical camera",
        "check": _check_amscope,
    },
    "dummy": {
        "description": "Simulated dummy camera (test pattern)",
        "check": lambda: (True, None),
    },
}


def get_available_camera_types(print_missing: bool = False) -> dict[str, dict[str, Any]]:
    """Inspect all registered camera drivers and return availability status.

    If print_missing is True, prints what is missing for each unavailable module.
    """
    statuses = {}
    for name, info in CAMERA_REGISTRY.items():
        is_avail, err_msg = info["check"]()
        statuses[name] = {
            "available": is_avail,
            "description": info["description"],
            "error": err_msg,
        }
        if not is_avail and print_missing and name != "pylon":  # avoid duplicate alias log
            print(f"[Camera] Driver '{name}' unavailable: {err_msg}")

    return statuses


def get_camera(camera_config: dict) -> Optional[CameraModule]:
    """Factory function to instantiate a CameraModule from configuration."""
    camera_type = str(camera_config.get("type", "none")).lower()

    if camera_type == "none":
        return None

    if camera_type == "webcam":
        try:
            from camera.webcam import Webcam
            try:
                index = int(camera_config.get("index", 0))
            except (ValueError, TypeError):
                index = 0
            return Webcam(index)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(f"Failed to initialize webcam: {e}") from e

    elif camera_type in ("basler", "pylon"):
        try:
            from camera.pylon import BaslerPylon
            try:
                index = int(camera_config.get("index", 0))
            except (ValueError, TypeError):
                index = 0
            return BaslerPylon(index)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                "Failed to initialize Basler camera: 'pypylon' is not installed. "
                "Install with: pip install '.[basler]'"
            ) from e

    elif camera_type == "flir":
        try:
            import camera.flir.flir_camera as flir
            return flir.FlirCamera()
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"Failed to initialize FLIR camera: {e}. "
                "Ensure the flir-private submodule is cloned into src/camera/flir"
            ) from e

    elif camera_type == "amscope":
        try:
            import camera.amscope.amscope_camera as amscope
            return amscope.AmscopeCamera()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AmScope camera: {e}") from e

    elif camera_type == "dummy":
        width = int(camera_config.get("width", 640))
        height = int(camera_config.get("height", 480))
        return DummyCamera(width=width, height=height)

    else:
        print(f"Unknown camera type in configuration: '{camera_type}'. Disabling camera.")
        return None


__all__ = [
    "CameraModule",
    "DummyCamera",
    "get_available_camera_types",
    "get_camera",
]


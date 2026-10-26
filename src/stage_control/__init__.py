"""Stage control subsystem providing dynamic driver discovery and factory instantiation."""

from __future__ import annotations

from typing import Any, Optional

from stage_control.stage_controller import StageController
from stage_control.dummy_stage import DummyStage


def _check_grbl() -> tuple[bool, Optional[str]]:
    try:
        import serial
        return True, None
    except ImportError:
        return False, "Missing optional package 'pyserial'. Install with: pip install '.[grbl]' or pip install pyserial"


def _check_omm() -> tuple[bool, Optional[str]]:
    try:
        import open_micro_stage_api
        return True, None
    except ImportError:
        return False, "Missing optional package 'open-micro-stage-api'. Install with: pip install '.[omm]'"


STAGE_REGISTRY: dict[str, dict[str, Any]] = {
    "dummy": {
        "description": "Dummy stage controller (simulated motion with delays)",
        "check": lambda: (True, None),
    },
    "grbl": {
        "description": "GRBL CNC stage controller (Serial)",
        "check": _check_grbl,
    },
    "omm": {
        "description": "Open Micro Manipulator (OMM) stage controller",
        "check": _check_omm,
    },
}


def get_available_stage_types(print_missing: bool = False) -> dict[str, dict[str, Any]]:
    """Inspect all registered stage drivers and return availability status.

    If print_missing is True, prints what is missing for each unavailable module.
    """
    statuses = {}
    for name, info in STAGE_REGISTRY.items():
        is_avail, err_msg = info["check"]()
        statuses[name] = {
            "available": is_avail,
            "description": info["description"],
            "error": err_msg,
        }
        if not is_avail and print_missing:
            print(f"[Stage] Driver '{name}' unavailable: {err_msg}")

    return statuses


def get_stage_controller(stage_config: dict, tiling: Optional[bool] = None) -> StageController:
    """Factory function to instantiate and connect a StageController from configuration."""
    def _create_dummy_stage() -> DummyStage:
        dummy_cfg = stage_config.get("dummy", {}) if isinstance(stage_config.get("dummy"), dict) else {}
        delay = float(dummy_cfg.get("delay", stage_config.get("delay", 0.01)))
        speed = dummy_cfg.get("speed", stage_config.get("speed"))
        if speed is not None:
            speed = float(speed)
        return DummyStage(delay=delay, speed=speed)

    if not stage_config.get("enabled", True):
        return _create_dummy_stage()

    stage_type = str(stage_config.get("type", "grbl")).lower()

    if stage_type == "omm":
        try:
            from stage_control.omm_stage import OMMStage
        except ImportError as e:
            raise RuntimeError(
                "Failed to initialize OMM stage: 'open-micro-stage-api' is not installed. "
                "Install with: pip install '.[omm]'"
            ) from e

        omm_config = stage_config.get("omm") or stage_config.get("oom") or {}
        z_max = omm_config.get("z-max", stage_config.get("z-max", -1))
        stage = OMMStage(z_max)
        port = omm_config.get("port", stage_config.get("port"))
        baud = omm_config.get("baud-rate", stage_config.get("baud-rate", 921600))
        if not port:
            raise ValueError("Serial port not specified for OMM stage in config")
        stage.connect(port, baud)
        return stage

    elif stage_type == "grbl":
        try:
            import serial
            from stage_control.grbl_stage import GrblStage
        except ImportError as e:
            raise RuntimeError(
                "Failed to initialize GRBL stage: 'pyserial' is not installed. "
                "Install with: pip install '.[grbl]'"
            ) from e

        grbl_config = stage_config.get("grbl", {}) if isinstance(stage_config.get("grbl"), dict) else {}
        port = grbl_config.get("port", stage_config.get("port"))
        baud = grbl_config.get("baud-rate", stage_config.get("baud-rate", 115200))
        if not port:
            raise ValueError("Serial port not specified for GRBL stage in config")
        try:
            serial_port = serial.Serial(port, baud)
            print(f"Using serial port {serial_port.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {port} at {baud} baud: {e}") from e

        effective_tiling = tiling if tiling is not None else grbl_config.get("tiling", stage_config.get("tiling", False))
        homing = grbl_config.get("homing", stage_config.get("homing", False))

        return GrblStage(serial_port, homing, effective_tiling)

    elif stage_type in ("dummy", "none"):
        return _create_dummy_stage()

    else:
        print(f"Unknown stage type: '{stage_type}'. Falling back to dummy stage controller.")
        return _create_dummy_stage()


__all__ = [
    "StageController",
    "DummyStage",
    "get_available_stage_types",
    "get_stage_controller",
]



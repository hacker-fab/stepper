"""Stage control subsystem providing dynamic driver discovery and factory instantiation."""

from __future__ import annotations

from typing import Any, Optional

from stage_control.stage_controller import StageController


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
        "description": "Dummy stage controller (no motion / simulation)",
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


def get_stage_controller(stage_config: dict) -> StageController:
    """Factory function to instantiate and connect a StageController from configuration."""
    if not stage_config.get("enabled", True):
        return StageController()

    stage_type = str(stage_config.get("type", "grbl")).lower()

    if stage_type == "omm":
        try:
            from stage_control.omm_stage import OMMStage
        except ImportError as e:
            raise RuntimeError(
                "Failed to initialize OMM stage: 'open-micro-stage-api' is not installed. "
                "Install with: pip install '.[omm]'"
            ) from e

        omm_config = stage_config.get("omm", {})
        z_max = omm_config.get("z-max", -1)
        autofocus = stage_config.get("autofocus", 0)
        stage = OMMStage(autofocus, z_max)
        stage.connect(stage_config["port"], stage_config["baud-rate"])
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

        port = stage_config["port"]
        baud = stage_config["baud-rate"]
        try:
            serial_port = serial.Serial(port, baud)
            print(f"Using serial port {serial_port.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {port} at {baud} baud: {e}") from e

        # default features to False if they aren't specified -> supports legacy config.toml files
        tiling = stage_config.get("tiling", False)
        autofocus = stage_config.get("autofocus", 0)
        homing = stage_config.get("homing", False)

        return GrblStage(serial_port, homing, tiling, autofocus)

    elif stage_type in ("dummy", "none"):
        return StageController()

    else:
        print(f"Unknown stage type: '{stage_type}'. Falling back to dummy stage controller.")
        return StageController()


__all__ = [
    "StageController",
    "get_available_stage_types",
    "get_stage_controller",
]


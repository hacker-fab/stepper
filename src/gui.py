"""Entry point for the Hacker Fab Stepper V2 application using PySide6."""

import os
import platform
import shutil
import sys
from pathlib import Path

# Ensure src/ is in sys.path
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import toml
from PySide6.QtWidgets import QApplication, QFileDialog

from camera import get_available_camera_types, get_camera
from core.engine import StepperEngine
from core.lithography import AlignmentConfig, LithographerConfig
from stage_control import get_available_stage_types, get_stage_controller
from ui.bridge import QtEngineBridge
from ui.main_window import MainWindow
from ui.projector import QtProjector

DEFAULT_RED_EXPOSURE: float = 4167.0
DEFAULT_UV_EXPOSURE: float = 25000.0


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Hacker Fab Stepper V2")

    # Determine config file (from argv, default.toml, or prompt)
    config_path = "default.toml"
    if len(sys.argv) > 1 and sys.argv[1].endswith(".toml"):
        config_path = sys.argv[1]
    elif not os.path.exists(config_path):
        selected_path, _ = QFileDialog.getOpenFileName(
            None, "Select Config File", "", "TOML files (*.toml);;All files (*.*)"
        )
        if selected_path:
            config_path = selected_path

    print(f"Loading configuration: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
    except FileNotFoundError:
        print("config file does not exist, copying settings from default.toml")
        shutil.copy("default.toml", config_path)
        with open(config_path, "r") as f:
            config = toml.load(f)

    print("--- Hardware Driver Discovery ---")
    get_available_camera_types(print_missing=True)
    get_available_stage_types(print_missing=True)
    print("---------------------------------")

    # STAGE CONFIG
    stage_config = config.get("stage", {})
    try:
        stage = get_stage_controller(stage_config)
    except Exception as e:
        print(f"Error initializing stage controller: {e}")
        return 1

    # CAMERA CONFIG
    camera_config = config.get("camera", {})
    try:
        camera = get_camera(camera_config)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return 1

    camera_scale = float(camera_config.get("gui-scale", 1.0))
    red_exposure = float(camera_config.get("red-exposure", DEFAULT_RED_EXPOSURE))
    uv_exposure = float(camera_config.get("uv-exposure", DEFAULT_UV_EXPOSURE))

    # ALIGNMENT CONFIG
    alignment_config = config.get("alignment", {})
    alignment_enabled = alignment_config.get("enabled", False)
    alignment_model = alignment_config.get("model_path", "ckpts/best.pt")

    right_marker_x = float(alignment_config.get("right_marker_x", 1820.0))
    left_marker_x = float(alignment_config.get("left_marker_x", 280.0))
    top_marker_y = float(alignment_config.get("top_marker_y", 269.0))
    bottom_marker_y = float(alignment_config.get("bottom_marker_y", 1075.0))
    x_scale_factor = float(alignment_config.get("x_scale_factor", -1100))
    y_scale_factor = float(alignment_config.get("y_scale_factor", 800))

    alignment_obj = AlignmentConfig(
        enabled=alignment_enabled,
        model_path=alignment_model,
        right_marker_x=right_marker_x,
        left_marker_x=left_marker_x,
        top_marker_y=top_marker_y,
        bottom_marker_y=bottom_marker_y,
        x_scale_factor=x_scale_factor,
        y_scale_factor=y_scale_factor,
    )

    # DLPC CONFIG
    dlpc = None
    projector_config = config.get("projector", {})
    if projector_config.get("dlpc_enabled", False):
        try:
            from dlpc import connect as dlpc_connect

            pid = projector_config.get("dlpc_pid", None)
            dlpc = dlpc_connect(pid)
            if dlpc is not None:
                print("Connected to DLPC6540 projector controller")
                SAFE_DEFAULT_LEVEL = 150
                SAFE_MAX_LEVEL = 400
                uv_level = projector_config.get("uv_led_drive_level", SAFE_DEFAULT_LEVEL)
                try:
                    uv_level = int(uv_level)
                    if not 0 <= uv_level <= SAFE_MAX_LEVEL:
                        uv_level = SAFE_DEFAULT_LEVEL
                except (TypeError, ValueError):
                    uv_level = SAFE_DEFAULT_LEVEL
                dlpc.set_led_drive_level(SAFE_DEFAULT_LEVEL, SAFE_DEFAULT_LEVEL, uv_level)
        except Exception as e:
            print(f"Warning: DLPC USB connection could not be established: {e}")

    lithographer_config = LithographerConfig(
        stage=stage,
        camera=camera,
        camera_scale=camera_scale,
        red_exposure=red_exposure,
        uv_exposure=uv_exposure,
        alignment=alignment_obj,
        dlpc=dlpc,
    )

    # Projector window (secondary monitor)
    projector = QtProjector()
    app.aboutToQuit.connect(projector.close)

    # Stepper process engine
    engine = StepperEngine(
        stage=stage,
        projector=projector,
        camera=camera,
        red_exposure=red_exposure,
        uv_exposure=uv_exposure,
        dlpc=dlpc,
    )
    engine.initialize_alignment(lithographer_config)

    # Qt Bridge & Main Application Window
    bridge = QtEngineBridge(engine)
    main_win = MainWindow(lithographer_config, engine, bridge)
    main_win.show()

    exit_code = app.exec()

    if dlpc is not None:
        try:
            dlpc.set_illumination_enable(0)
            dlpc.close()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

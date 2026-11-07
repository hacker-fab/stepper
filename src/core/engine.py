from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from camera.camera_module import CameraModule
    from projector import ProjectorController
    from stage_control.stage_controller import StageController

from .chip_project import ChipProgress, ChipProject
from .events import Event, EventBus
from .machine_config import MachineConfig
from .operation import ExecutionContext, Operation, OperationManager


class StepperEngine:
    """Core process controller for the lithography stepper.

    Adheres strictly to the 3-pillar architecture:
    1. Chip Project (data): Self-contained ChipProject emitting layer/project events.
    2. Operation (workflow): Single active operation lifecycle managed by OperationManager.
    3. Machine Control (hardware): Directly accessible submodules (stage, projector, camera, dlpc).
    """

    def __init__(
        self,
        stage: StageController,
        projector: ProjectorController,
        camera: CameraModule,
        dlpc: Optional[Any] = None,
        delay_func: Optional[Callable[[float], None]] = None,
        warning_callback: Optional[Callable[[str], None]] = None,
        machine_config: Optional[MachineConfig] = None,
    ):
        self.stage = stage
        self.projector = projector
        self.camera = camera
        self.dlpc = dlpc
        
        self.delay_func = delay_func or time.sleep
        self.warning_callback = warning_callback or self._default_warning

        # Link event bus
        self.event_bus = EventBus()
        self.stage.event_bus = self.event_bus
        self.projector.event_bus = self.event_bus
        self.camera.event_bus = self.event_bus

        self.projector.dlpc = self.dlpc

        # 1. Machine Configuration
        if machine_config is not None:
            self.machine_config = machine_config
        else:
            self.machine_config = MachineConfig.load_or_default("machine_config.toml")

        if self.camera is not None:
            self.camera.set_flip_mode(self.machine_config.optics.camera_flip_mode)

        # Auto-load stage calibration from config if available
        self._init_stage_calibration()

        # 2. Chip Project Data Object
        proj_size = self.projector.projector_size() if self.projector is not None else (1920, 1080)
        self.project = ChipProject(
            events=self.event_bus,
            machine_config=self.machine_config,
            projector_size=proj_size,
        )
        self.projector.set_project(self.project)

        # 3. Execution Context & Operation Manager
        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            project=self.project,
            event_bus=self.event_bus,
            machine_config=self.machine_config,
            warning_callback=self.create_warning,
            delay_func=self.delay_func,
        )
        self.operations = OperationManager(self.context, self.event_bus)

        # Snapshot folder
        self.snapshot_directory = Path("stepper_captures")
        self.snapshot_directory.mkdir(exist_ok=True)

    def _init_stage_calibration(self):
        cal_path = self.machine_config.stage_calibration.calibration_file_path
        if cal_path and os.path.exists(cal_path) and self.stage is not None:
            try:
                from stage_control.calibration import StageCalibration
                cal = StageCalibration.load_from_json(cal_path)
                if abs(self.machine_config.stage_calibration.additional_stage_rotation_deg) > 1e-6:
                    cal.set_additional_rotation(self.machine_config.stage_calibration.additional_stage_rotation_deg)
                if abs(self.machine_config.stage_calibration.additional_stage_skew) > 1e-9:
                    cal.set_additional_skew(self.machine_config.stage_calibration.additional_stage_skew)
                if (
                    abs(self.machine_config.stage_calibration.additional_stage_scale_x - 1.0) > 1e-7
                    or abs(self.machine_config.stage_calibration.additional_stage_scale_y - 1.0) > 1e-7
                ):
                    cal.set_additional_scale(
                        self.machine_config.stage_calibration.additional_stage_scale_x,
                        self.machine_config.stage_calibration.additional_stage_scale_y,
                    )
                self.stage.set_calibration(cal)
                print(f"[StepperEngine] Loaded stage calibration from {cal_path}", flush=True)
            except Exception as e:
                print(f"[StepperEngine] Warning: Could not load stage calibration from {cal_path}: {e}", flush=True)

    def set_machine_config(self, config: MachineConfig):
        """Sets the active machine config, updates context and calibration, and emits change event."""
        self.machine_config = config
        self.context.machine_config = config
        if self.camera is not None:
            self.camera.set_flip_mode(config.optics.camera_flip_mode)
        if self.project is not None:
            self.project.machine_config = config
            self.project.invalidate_all_tile_caches()
        if self.stage is not None:
            cal = self.stage.get_calibration()
            if cal is not None:
                cal.set_additional_rotation(config.stage_calibration.additional_stage_rotation_deg)
                cal.set_additional_skew(config.stage_calibration.additional_stage_skew)
                cal.set_additional_scale(
                    config.stage_calibration.additional_stage_scale_x,
                    config.stage_calibration.additional_stage_scale_y,
                )
            else:
                self._init_stage_calibration()
        else:
            self._init_stage_calibration()
        self.event_bus.emit(Event.MACHINE_CONFIG_CHANGED, self.machine_config)

    # -------------------------------------------------------------------------
    # Crosshair Position Interface
    # -------------------------------------------------------------------------
    def set_primary_crosshair(self, u: float, v: float) -> None:
        if self.camera is not None:
            self.camera.set_primary_crosshair(u, v)

    def get_primary_crosshair(self) -> tuple[float, float]:
        if self.camera is not None:
            return self.camera.get_primary_crosshair()
        return (0.5, 0.5)

    def set_secondary_crosshair(
        self,
        u: Optional[float],
        v: Optional[float] = None,
    ) -> None:
        if self.camera is not None:
            if u is None:
                self.camera.set_secondary_crosshair(None)
            elif v is not None:
                self.camera.set_secondary_crosshair((float(u), float(v)))

    def get_secondary_crosshair(self) -> Optional[tuple[float, float]]:
        if self.camera is not None:
            return self.camera.get_secondary_crosshair()
        return None

    def save_machine_config(self, path: str = "machine_config.toml"):
        """Saves current machine config to TOML file."""
        self.machine_config.save(path)

    def load_machine_config(self, path: str = "machine_config.toml"):
        """Loads machine config from TOML file and applies it."""
        config = MachineConfig.load(path)
        self.set_machine_config(config)

    def _default_warning(self, msg: str):
        print(f"Warning: {msg}")

    def create_warning(self, msg: str):
        self.event_bus.emit(Event.WARNING_MESSAGE, msg)
        self.warning_callback(msg)

    # -------------------------------------------------------------------------
    # 1. Chip Project & Chip Progress Persistence
    # -------------------------------------------------------------------------
    def load_project(self, path: str):
        existing_progress = self.project.chip_progress if self.project is not None else ChipProgress(events=self.event_bus)
        proj_size = self.projector.projector_size() if self.projector is not None else (1920, 1080)
        self.project = ChipProject.load(path, events=self.event_bus)
        self.project.machine_config = self.machine_config
        self.project.projector_size = proj_size
        self.project.chip_progress = existing_progress
        self.project.chip_progress.events = self.event_bus
        self.context.project = self.project
        self.projector.set_project(self.project)
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)

    def save_project(self, path: str):
        self.project.save(path)

    def new_project(self):
        existing_progress = self.project.chip_progress if self.project is not None else ChipProgress(events=self.event_bus)
        proj_size = self.projector.projector_size() if self.projector is not None else (1920, 1080)
        self.project = ChipProject(
            events=self.event_bus,
            machine_config=self.machine_config,
            projector_size=proj_size,
            chip_progress=existing_progress,
        )
        self.context.project = self.project
        self.projector.set_project(self.project)
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)

    def load_chip_progress(self, path: str):
        progress = ChipProgress.load(path, events=self.event_bus)
        self.project.chip_progress = progress
        self.project.invalidate_all_tile_caches()
        self.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, progress)
        self.event_bus.emit(Event.EXPOSURE_HISTORY_CHANGED, progress.exposure_history)
        self.event_bus.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def save_chip_progress(self, path: str):
        self.project.chip_progress.save(path)

    def reset_chip_progress(self):
        self.project.chip_progress.reset()
        self.project.invalidate_all_tile_caches()

    # -------------------------------------------------------------------------
    # 2. Operations
    # -------------------------------------------------------------------------
    def run_operation(self, operation: Operation) -> Any:
        """Executes an operation synchronously using the OperationManager."""
        return self.operations.run(operation)

    @property
    def current_operation(self) -> Optional[Operation]:
        """Returns the currently active operation or None."""
        return self.operations.current_operation

    def abort_operation(self):
        """Aborts the currently running operation."""
        self.operations.abort_current()


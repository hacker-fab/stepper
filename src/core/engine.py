from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from camera.camera_module import CameraModule
    from projector import ProjectorController
    from stage_control.stage_controller import StageController

from .chip_project import ChipProject
from .events import Event, EventBus
from .operation import ExecutionContext, Operation, OperationManager


class StepperEngine:
    """Core process controller for the lithography stepper.

    Adheres strictly to the 3-pillar architecture:
    1. Chip Project (data): Self-contained ChipProject emitting layer/project events.
    2. Operation (workflow): Single active operation lifecycle managed by OperationManager.
    3. Machine Control (hardware): Directly accessible submodules (stage, projector, camera, dlpc, alignment).
    """

    def __init__(
        self,
        stage: StageController,
        projector: ProjectorController,
        camera: CameraModule,
        dlpc: Optional[Any] = None,
        delay_func: Optional[Callable[[float], None]] = None,
        warning_callback: Optional[Callable[[str], None]] = None,
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

        if hasattr(self.projector, "dlpc"):
            self.projector.dlpc = self.dlpc

        # 1. Chip Project Data Object
        self.project = ChipProject(events=self.event_bus)

        # 2. Execution Context & Operation Manager
        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            project=self.project,
            event_bus=self.event_bus,
            warning_callback=self.create_warning,
            delay_func=self.delay_func,
        )
        self.operations = OperationManager(self.context, self.event_bus)

        # Snapshot folder
        self.snapshot_directory = Path("stepper_captures")
        self.snapshot_directory.mkdir(exist_ok=True)

    def _default_warning(self, msg: str):
        print(f"Warning: {msg}")

    def create_warning(self, msg: str):
        self.event_bus.emit(Event.WARNING_MESSAGE, msg)
        self.warning_callback(msg)

    # -------------------------------------------------------------------------
    # 1. Chip Project Persistence
    # -------------------------------------------------------------------------
    def load_project(self, path: str):
        self.project = ChipProject.load(path, events=self.event_bus)
        self.context.project = self.project
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)
        self.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED)

    def save_project(self, path: str):
        self.project.save(path)

    def new_project(self):
        self.project = ChipProject(events=self.event_bus)
        self.context.project = self.project
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)
        self.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED)

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


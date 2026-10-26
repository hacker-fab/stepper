from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from camera.camera_module import CameraModule
    from projector import ProjectorController
    from stage_control.stage_controller import StageController

from core.chip_project import ChipProject
from core.events import Event, EventBus


@dataclass
class ExecutionContext:
    """Hardware and project context provided to operations during execution."""
    stage: StageController
    projector: ProjectorController
    camera: CameraModule

    project: Optional[ChipProject] = None
    event_bus: Optional[EventBus] = None

    warning_callback: Optional[Callable[[str], None]] = None
    delay_func: Callable[[float], None] = time.sleep


class Operation(ABC):
    """Abstract base class for all stepper operations."""

    def __init__(self, name: str):
        self.name = name
        self.is_aborted = False
        self.progress = 0.0

    @abstractmethod
    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        """Run the operation synchronously in a background thread. Return None on success, error message if failed."""
        pass

    def abort(self):
        """Cooperatively request abortion of the operation."""
        self.is_aborted = True


class OperationManager:
    """Manages the single global running operation and coordinates abortion."""

    def __init__(self, context: ExecutionContext, events: EventBus):
        self.context = context
        self.events = events
        self.current_operation: Optional[Operation] = None

    def can_start_operation(self) -> bool:
        return self.current_operation is None

    def run(self, operation: Operation) -> Any:
        """Synchronously execute operation using the execution context."""
        if not self.can_start_operation():
            operation_name = self.current_operation.name if self.current_operation is not None else "unknown"
            msg = f"Cannot start '{operation.name}': another operation ('{operation_name}') is currently running."
            print(f"[Operation Error] {msg}", flush=True)
            if self.context.warning_callback:
                self.context.warning_callback(msg)
            raise RuntimeError(msg)

        self.current_operation = operation
        print(f"[Operation] Starting '{operation.name}'...", flush=True)
        self.events.emit(Event.OPERATION_STARTED, operation.name)

        def report(progress: float, message: str):
            operation.progress = progress
            self.events.emit(Event.OPERATION_PROGRESS, progress, message)

        err_msg: Optional[str] = None
        try:
            res = operation.execute(self.context, report)
            if res is not None:
                err_msg = str(res)
            return res
        except Exception as e:
            err_msg = str(e)
            traceback.print_exc()
            raise
        finally:
            was_aborted = operation.is_aborted
            op_name = operation.name
            self.current_operation = None

            if was_aborted:
                print(f"[Operation Aborted] Operation '{op_name}' was aborted.", flush=True)
                self.events.emit(Event.OPERATION_ABORTED, op_name)
            elif err_msg is not None:
                print(f"[Operation Error] Operation '{op_name}' failed: {err_msg}", flush=True)
                if self.context.warning_callback:
                    self.context.warning_callback(f"Operation '{op_name}' failed: {err_msg}")
                self.events.emit(Event.OPERATION_FAILED, op_name, err_msg)
            else:
                print(f"[Operation Finished] Operation '{op_name}' completed successfully.", flush=True)
                self.events.emit(Event.OPERATION_FINISHED, op_name)

    def start_operation(
        self,
        operation: Operation,
        run_async_callback: Callable[[Callable], None],
        on_finished: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Attempts to start an operation asynchronously."""
        if not self.can_start_operation():
            operation_name = self.current_operation.name if self.current_operation is not None else "unknown"
            msg = f"Cannot start '{operation.name}': another operation ('{operation_name}') is currently running."
            print(f"[Operation Error] {msg}", flush=True)
            if self.context.warning_callback:
                self.context.warning_callback(msg)
            if on_error:
                on_error(msg)
            return False

        self.current_operation = operation
        print(f"[Operation] Starting '{operation.name}'...", flush=True)
        self.events.emit(Event.OPERATION_STARTED, operation.name)

        def worker():
            err_msg: Optional[str] = None
            try:
                def report(progress: float, message: str):
                    operation.progress = progress
                    self.events.emit(Event.OPERATION_PROGRESS, progress, message)

                res = operation.execute(self.context, report)
                if res is not None:
                    err_msg = str(res)
            except Exception as e:
                err_msg = str(e)
                traceback.print_exc()
            finally:
                was_aborted = operation.is_aborted
                op_name = operation.name
                self.current_operation = None

                if was_aborted:
                    print(f"[Operation Aborted] Operation '{op_name}' was aborted.", flush=True)
                    self.events.emit(Event.OPERATION_ABORTED, op_name)
                elif err_msg is not None:
                    print(f"[Operation Error] Operation '{op_name}' failed: {err_msg}", flush=True)
                    if self.context.warning_callback:
                        self.context.warning_callback(f"Operation '{op_name}' failed: {err_msg}")
                    self.events.emit(Event.OPERATION_FAILED, op_name, err_msg)
                    if on_error:
                        try:
                            on_error(err_msg)
                        except Exception as cb_err:
                            print(f"[Operation Error] Exception in on_error callback: {cb_err}", flush=True)
                else:
                    print(f"[Operation Finished] Operation '{op_name}' completed successfully.", flush=True)
                    self.events.emit(Event.OPERATION_FINISHED, op_name)

                if on_finished:
                    try:
                        on_finished()
                    except Exception as cb_err:
                        print(f"[Operation Error] Exception in on_finished callback: {cb_err}", flush=True)

        run_async_callback(worker)
        return True

    def abort_current(self):
        """Aborts the current active operation if one is running."""
        if self.current_operation is not None:
            print(f"[Operation] Aborting '{self.current_operation.name}'...", flush=True)
            self.current_operation.abort()
        else:
            print("[Operation] Abort requested but no operation is currently running.", flush=True)

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from core.engine import StepperEngine
from core.events import Event, MovementLock, ShownImage


class WorkerRunnable(QRunnable):
    """Executes a function in a background thread and emits signals on finish or error."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class QtEngineBridge(QObject):
    """Bridges StepperEngine events and async operations to Qt Signals."""

    stage_position_changed = Signal(tuple)
    shown_image_changed = Signal(object)
    movement_lock_changed = Signal(object)
    pattern_progress_changed = Signal(float, float)
    patterning_busy_changed = Signal(bool)
    chip_changed = Signal(object)
    warning_emitted = Signal(str)
    status_message = Signal(str)
    snapshot_saved = Signal(str)
    pattern_image_changed = Signal()
    image_adjust_changed = Signal()
    camera_frame_ready = Signal(object)

    def __init__(self, engine: StepperEngine):
        super().__init__()
        self.engine = engine
        self.threadpool = QThreadPool.globalInstance()

        # Intercept warnings from engine
        self.engine.warning_callback = self._on_engine_warning

        # Subscribe to EventBus
        self.engine.events.add_listener(
            Event.STAGE_POSITION_CHANGED,
            lambda: self.stage_position_changed.emit(self.engine.stage_setpoint),
        )
        self.engine.events.add_listener(
            Event.SHOWN_IMAGE_CHANGED,
            lambda: self.shown_image_changed.emit(self.engine.shown_image),
        )
        self.engine.events.add_listener(
            Event.MOVEMENT_LOCK_CHANGED,
            lambda: self.movement_lock_changed.emit(self.engine.movement_lock),
        )
        self.engine.events.add_listener(
            Event.EXPOSURE_PATTERN_PROGRESS_CHANGED,
            lambda: self.pattern_progress_changed.emit(
                self.engine.patterning_progress, self.engine.exposure_progress
            ),
        )
        self.engine.events.add_listener(
            Event.PATTERNING_BUSY_CHANGED,
            lambda: self.patterning_busy_changed.emit(self.engine.patterning_busy),
        )
        self.engine.events.add_listener(
            Event.CHIP_CHANGED,
            lambda: self.chip_changed.emit(self.engine.chip),
        )
        self.engine.events.add_listener(
            Event.PATTERN_IMAGE_CHANGED,
            lambda: self.pattern_image_changed.emit(),
        )
        self.engine.events.add_listener(
            Event.IMAGE_ADJUST_CHANGED,
            lambda: self.image_adjust_changed.emit(),
        )
        self.engine.events.add_listener(
            Event.SNAPSHOT,
            lambda filename: self.snapshot_saved.emit(filename),
        )

    def _on_engine_warning(self, msg: str):
        print(f"[Warning] {msg}")
        self.warning_emitted.emit(msg)

    def run_async(
        self,
        fn: Callable,
        *args,
        on_finished: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        """Dispatches a long-running function to a background worker thread."""
        worker = WorkerRunnable(fn, *args, **kwargs)
        if on_finished:
            worker.signals.finished.connect(on_finished)
        if on_error:
            worker.signals.error.connect(on_error)
        self.threadpool.start(worker)


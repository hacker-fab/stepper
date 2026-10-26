import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from core.engine import StepperEngine
from core.events import Event


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

    # Project related
    project_changed = Signal(object)
    active_layer_changed = Signal(int)
    active_tile_changed = Signal(int)
    exposure_config_changed = Signal()
    layer_cache_recomputed = Signal(object)
    exposure_history_changed = Signal(object)

    # Stage
    stage_position_changed = Signal(tuple)

    # Projector
    projector_on_off_changed = Signal(bool)
    projector_color_mode_changed = Signal(object)
    projector_image_source_changed = Signal(object)
    projector_brightness_changed = Signal(float)
    projector_image_changed = Signal(object)

    # Camera
    camera_frame_ready = Signal(object)

    # Operations
    operation_started = Signal(str)
    operation_progress = Signal(float, str)
    operation_finished = Signal(str)
    operation_aborted = Signal(str)
    operation_failed = Signal(str, str)

    # Warnings & Status
    warning_emitted = Signal(str)
    status_message = Signal(str)

    def __init__(self, engine: StepperEngine):
        super().__init__()
        self.engine = engine
        self.threadpool = QThreadPool.globalInstance()

        # Intercept warnings from engine
        self.engine.warning_callback = self._on_engine_warning

        # Subscribe to EventBus
        # Project related
        self.engine.event_bus.add_listener(
            Event.PROJECT_CHANGED,
            lambda *args: self.project_changed.emit(self.engine.project),
        )
        self.engine.event_bus.add_listener(
            Event.ACTIVE_LAYER_CHANGED,
            lambda idx=0, *args: self.active_layer_changed.emit(
                idx if isinstance(idx, int) else self.engine.project.active_layer_index
            ),
        )
        self.engine.event_bus.add_listener(
            Event.ACTIVE_TILE_CHANGED,
            lambda idx=0, *args: self.active_tile_changed.emit(
                idx if isinstance(idx, int) else self.engine.project.active_tile_index
            ),
        )
        self.engine.event_bus.add_listener(
            Event.EXPOSURE_CONFIG_CHANGED,
            lambda *args: self.exposure_config_changed.emit(),
        )
        self.engine.event_bus.add_listener(
            Event.LAYER_CACHE_RECOMPUTED,
            lambda layer=None, *args: self.layer_cache_recomputed.emit(layer),
        )
        self.engine.event_bus.add_listener(
            Event.EXPOSURE_HISTORY_CHANGED,
            lambda history=None, *args: self.exposure_history_changed.emit(
                history if history is not None else (self.engine.project.exposure_history if self.engine.project else [])
            ),
        )

        # Stage
        self.engine.event_bus.add_listener(
            Event.STAGE_POSITION_CHANGED,
            lambda *args: self.stage_position_changed.emit(self.engine.stage.get_position()),
        )

        # Projector
        self.engine.event_bus.add_listener(
            Event.PROJECTOR_ON_OFF_CHANGED,
            lambda on=False, *args: self.projector_on_off_changed.emit(
                on if isinstance(on, bool) else self.engine.projector.is_on
            ),
        )
        self.engine.event_bus.add_listener(
            Event.PROJECTOR_COLOR_MODE_CHANGED,
            lambda mode=None, *args: self.projector_color_mode_changed.emit(
                mode if mode is not None else self.engine.projector.color_mode
            ),
        )
        self.engine.event_bus.add_listener(
            Event.PROJECTOR_IMAGE_SOURCE_CHANGED,
            lambda src=None, *args: self.projector_image_source_changed.emit(
                src if src is not None else self.engine.projector.image_source
            ),
        )
        self.engine.event_bus.add_listener(
            Event.PROJECTOR_BRIGHTNESS_CHANGED,
            lambda val=1.0, *args: self.projector_brightness_changed.emit(
                float(val) if val is not None else self.engine.projector.brightness
            ),
        )
        self.engine.event_bus.add_listener(
            Event.PROJECTOR_IMAGE_CHANGED,
            lambda img=None, *args: self.projector_image_changed.emit(
                img if img is not None else self.engine.projector._displayed_image_cache
            ),
        )

        # Camera
        self.engine.event_bus.add_listener(
            Event.CAMERA_FRAME_READY,
            lambda frame=None, *args: self.camera_frame_ready.emit(frame),
        )

        # Operations
        self.engine.event_bus.add_listener(
            Event.OPERATION_STARTED,
            lambda name="", *args: self.operation_started.emit(str(name)),
        )
        self.engine.event_bus.add_listener(
            Event.OPERATION_PROGRESS,
            lambda pct=0.0, msg="", *args: self.operation_progress.emit(float(pct), str(msg)),
        )
        self.engine.event_bus.add_listener(
            Event.OPERATION_FINISHED,
            lambda name="", *args: self.operation_finished.emit(str(name)),
        )
        self.engine.event_bus.add_listener(
            Event.OPERATION_ABORTED,
            lambda name="", *args: self.operation_aborted.emit(str(name)),
        )
        self.engine.event_bus.add_listener(
            Event.OPERATION_FAILED,
            lambda name="", err="", *args: self.operation_failed.emit(str(name), str(err)),
        )

        # Warning
        self.engine.event_bus.add_listener(
            Event.WARNING_MESSAGE,
            lambda msg="", *args: self.warning_emitted.emit(str(msg)),
        )

    def start_operation(
        self,
        operation,
        on_finished: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Starts an operation using the engine's OperationManager with async thread dispatch."""
        return self.engine.operations.start_operation(
            operation,
            run_async_callback=lambda worker: self.run_async(worker),
            on_finished=on_finished,
            on_error=on_error or (lambda err: self.warning_emitted.emit(f"Operation failed: {err}")),
        )

    def _on_engine_warning(self, msg: str):
        print(f"[Warning] {msg}", flush=True)
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


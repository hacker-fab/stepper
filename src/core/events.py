from enum import Enum, auto
from typing import Callable, Dict, List


class StrAutoEnum(str, Enum):
    """Base class for string-valued enums that use auto()"""

    def _generate_next_value_(name, *_):
        return name.lower()


class ColorMode(StrAutoEnum):
    """Projector color display mode."""

    DISABLE = auto()
    RED = auto()
    UV = auto()


class ProjectorImageSource(StrAutoEnum):
    """Source of the pattern/image displayed by the projector."""

    ACTIVE_LAYER = auto()
    CUSTOM_FILE = auto()


class Event(StrAutoEnum):
    """Events dispatched across the stepper software system."""

    # Project related
    PROJECT_CHANGED = auto()  # Emitted when a project is loaded, created, saved, or layer structure changes
    ACTIVE_LAYER_CHANGED = auto()  # Emitted when the selected active layer index changes
    ACTIVE_TILE_CHANGED = auto()  # Emitted when the selected active tile index changes
    EXPOSURE_CONFIG_CHANGED = auto()  # Emitted when patterning/exposure config changes (global settings or per-layer overrides)

    # Stage
    STAGE_POSITION_CHANGED = auto()  # Emitted when stage position coordinates change

    # Projector
    PROJECTOR_COLOR_MODE_CHANGED = auto()  # Emitted when the projector color mode changes
    PROJECTOR_IMAGE_SOURCE_CHANGED = auto()  # Emitted when the projector image source changes
    PROJECTOR_IMAGE_CHANGED = auto()  # Emitted exclusively by the projector when displayed image changes

    # Camera
    CAMERA_FRAME_READY = auto()  # Emitted when a new camera frame is captured and ready

    # Operations
    OPERATION_STARTED = auto()  # Emitted when an operation begins execution
    OPERATION_PROGRESS = auto()  # Emitted during operation execution with progress fraction and status message
    OPERATION_FINISHED = auto()  # Emitted when an operation finishes successfully
    OPERATION_ABORTED = auto()  # Emitted when an operation is aborted or cancelled

    # Warning
    WARNING_MESSAGE = auto()  # Emitted when a warning message or non-fatal issue occurs




class EventBus:
    """Generic event dispatcher decoupled from any UI framework."""

    def __init__(self):
        self._listeners: Dict[Event, List[Callable]] = {}

    def add_listener(self, event: Event, listener: Callable):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def remove_listener(self, event: Event, listener: Callable):
        if event in self._listeners and listener in self._listeners[event]:
            self._listeners[event].remove(listener)

    def emit(self, event: Event, *args, **kwargs):
        if event not in self._listeners:
            return
        for listener in list(self._listeners[event]):
            listener(*args, **kwargs)

from enum import Enum, auto

class StrAutoEnum(str, Enum):
    """Base class for string-valued enums that use auto()"""

    def _generate_next_value_(name, *_):
        return name.lower()


class ShownImage(StrAutoEnum):
    """The type of image currently being displayed by the projector"""

    CLEAR = auto()
    PATTERN = auto()
    FLATFIELD = auto()
    RED_FOCUS = auto()
    UV_FOCUS = auto()


class PatterningStatus(StrAutoEnum):
    """The current state of the patterning process"""

    IDLE = auto()
    PATTERNING = auto()
    ABORTING = auto()


class Event(StrAutoEnum):
    """Events that can be dispatched to listeners"""

    SNAPSHOT = auto()
    SHOWN_IMAGE_CHANGED = auto()
    STAGE_POSITION_CHANGED = auto()
    IMAGE_ADJUST_CHANGED = auto()
    PATTERN_IMAGE_CHANGED = auto()
    MOVEMENT_LOCK_CHANGED = auto()
    EXPOSURE_PATTERN_PROGRESS_CHANGED = auto()
    PATTERNING_BUSY_CHANGED = auto()
    PATTERNING_FINISHED = auto()
    CHIP_CHANGED = auto()


class MovementLock(StrAutoEnum):
    """Controls whether stage position can be manually adjusted"""

    UNLOCKED = auto()  # X, Y, and Z are free to move
    XY_LOCKED = auto() # Only Z (focus) is free to move to avoid smearing UV focus pattern
    LOCKED = auto()  # No positions can move to avoid disrupting patterning


class RedFocusSource(StrAutoEnum):
    """The source image to use for red focus mode"""

    IMAGE = auto()  # Uses the dedicated red focus image
    SOLID = auto()  # Shows a solid red screen
    PATTERN = auto()  # Uses the blue channel from the pattern image
    INV_PATTERN = auto()  # Uses the inverse of the blue channel from the pattern image

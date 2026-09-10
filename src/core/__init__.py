"""Core lithography process package for Hacker Fab Stepper.

Contains hardware state management, exposure sequencing, autofocus routines,
alignment marker vision, and tiling algorithms decoupled from any GUI toolkit.
"""

from .alignment import (
    compute_standard_alignment_offset,
    compute_tiling_alignment_offset,
    detect_alignment_markers,
    detect_alignment_markers_tiling,
    load_alignment_model,
)
from .engine import StepperEngine
from .events import (
    Event,
    EventBus,
    MovementLock,
    PatterningStatus,
    RedFocusSource,
    ShownImage,
    StrAutoEnum,
)
from .focus import compute_focus_score, execute_autofocus, fetch_focus_score
from .lithography import AlignmentConfig, Chip, ChipLayer, ExposureLog, LithographerConfig
from .tiling import calculate_tile_position, generate_snake_sequence, split_image_with_overlap

__all__ = [
    "StrAutoEnum",
    "ShownImage",
    "PatterningStatus",
    "Event",
    "MovementLock",
    "RedFocusSource",
    "EventBus",
    "AlignmentConfig",
    "LithographerConfig",
    "ExposureLog",
    "ChipLayer",
    "Chip",
    "fetch_focus_score",
    "compute_focus_score",
    "execute_autofocus",
    "load_alignment_model",
    "detect_alignment_markers",
    "detect_alignment_markers_tiling",
    "compute_standard_alignment_offset",
    "compute_tiling_alignment_offset",
    "split_image_with_overlap",
    "generate_snake_sequence",
    "calculate_tile_position",
    "StepperEngine",
]

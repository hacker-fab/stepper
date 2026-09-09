# This serves as configurations for the Tiling Feature
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List
from PIL import Image

from camera.camera_module import CameraModule
from stage_control.stage_controller import StageController

@dataclass
class ExposureLog:
    time: datetime
    path: str
    coords: tuple[float, float, float]
    duration: float # ms
    aborted: bool

    def to_disk(self):
        return {
            "time": str(self.time),
            "path": self.path,
            "coords": self.coords,
            "duration": self.duration,
            "aborted": self.aborted,
        }

    @classmethod
    def from_disk(cls, d):
        return cls(
            datetime.fromisoformat(d["time"]),
            d["path"],
            d["coords"],
            d["duration"],
            d["aborted"],
        )

@dataclass
class ChipLayer:
    exposures: List[ExposureLog]

    def to_disk(self):
        return {"exposures": [ex.to_disk() for ex in self.exposures]}

    @classmethod
    def from_disk(cls, d):
        return cls([ExposureLog.from_disk(ex) for ex in d["exposures"]])

@dataclass
class Chip:
    layers: List[ChipLayer]

    def to_disk(self):
        return {"layers": [layer.to_disk() for layer in self.layers]}

    @classmethod
    def from_disk(cls, d):
        return cls([ChipLayer.from_disk(layer) for layer in d["layers"]])

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
    EXPOSURE_TIME_CHANGED = auto()
    EXPOSURE_PATTERN_PROGRESS_CHANGED = auto()
    PATTERNING_BUSY_CHANGED = auto()
    PATTERNING_FINISHED = auto()
    CHIP_CHANGED = auto()
    STITCH_COMPLETED = auto()
    START_TILING = auto()

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

@dataclass
class AlignmentConfig:
    enabled: bool
    model_path: str
    right_marker_x: float
    left_marker_x: float
    top_marker_y: float
    bottom_marker_y: float
    x_scale_factor: float
    y_scale_factor: float

@dataclass
class LithographerConfig:
    stage: StageController
    camera: CameraModule
    camera_scale: float
    red_exposure: float
    uv_exposure: float
    alignment: AlignmentConfig


@dataclass
class TilingParameters:
    align_image: Image
    ratio: float
    stride_x: int
    stride_y: int
    num_rows: int
    num_cols: int
    prefix_path: str
    px_to_step_x: float
    px_to_step_y: float
    step_error_threshold_x: int
    step_error_threshold_y: int

@dataclass
class ImageCaptureSettings:
    stride_x_um: int # x direction stride in um(steps) for stage movement during image capture
    stride_y_um: int # y direction stride in um(steps) for stage movement during image capture
    total_x_um: int  # total steps in x direction we need to move
    total_y_um: int  # total steps in y direction we need to move
    capture_folder: str # capture folder where we store all data + logs

@dataclass
class ImageStitchSettings:
    num_rows: int # number of tile rows during pattern segmentation
    num_cols: int # number of tile col during pattern segmentation
    output_folder: str # output folder where we store the stitched image (set the same as capture_folder)
    resize: float # resize factor of the stitched image before we save to the output folder (we do this to prevent it from being massive)
    debug: bool # flag for debug print
    threshold: int # error margin we allow before defaulting to expected_dx and expected_dy during stitching

@dataclass
class TilePreprocessSettings:
    gaussian_kernel_size: tuple[int, int]

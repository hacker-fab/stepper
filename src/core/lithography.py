from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from camera import CameraModule
from stage_control import StageController


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
    camera: Optional[CameraModule]
    camera_scale: float
    red_exposure: float
    uv_exposure: float
    alignment: AlignmentConfig
    dlpc: Optional[Any] = None  # DLPC instance if USB projector control is enabled


@dataclass
class ExposureLog:
    time: datetime
    path: str
    coords: tuple[float, float, float]
    duration: float  # ms
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

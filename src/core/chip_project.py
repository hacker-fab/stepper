import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Any
from core.events import Event, EventBus
from lib.tiling import split_image_into_tiles
import cv2
import numpy as np



@dataclass
class ExposureRecord:
    """Record of a single exposure event on the platform."""

    coords: Tuple[float, float, float]
    time: datetime = field(default_factory=datetime.now)
    duration: float = 0.0  # ms
    aborted: bool = False
    layer_index: Optional[int] = None
    tile_index: Optional[int] = None


@dataclass
class PatterningSettings:
    """Project-level default patterning and exposure settings."""

    exposure_time: float = 8000.0  # ms
    tiling_enabled: bool = False
    tile_width: int = 3840  # px
    tile_height: int = 2160  # px
    overlap_x: int = 200  # px
    overlap_y: int = 200  # px
    pitch_x: float = 983.0  # µm
    pitch_y: float = 512.0  # µm
    border_size: float = 0.0  # px
    posterize_strength: Optional[int] = None

    def to_disk(self) -> dict:
        return {
            "exposure_time": self.exposure_time,
            "tiling_enabled": self.tiling_enabled,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "overlap_x": self.overlap_x,
            "overlap_y": self.overlap_y,
            "pitch_x": self.pitch_x,
            "pitch_y": self.pitch_y,
            "border_size": self.border_size,
            "posterize_strength": self.posterize_strength,
        }

    @classmethod
    def from_disk(cls, d: dict) -> "PatterningSettings":
        return cls(
            exposure_time=float(d.get("exposure_time", 8000.0)),
            tiling_enabled=bool(d.get("tiling_enabled", False)),
            tile_width=int(d.get("tile_width", 3840)),
            tile_height=int(d.get("tile_height", 2160)),
            overlap_x=int(d.get("overlap_x", 200)),
            overlap_y=int(d.get("overlap_y", 200)),
            pitch_x=float(d.get("pitch_x", 983.0)),
            pitch_y=float(d.get("pitch_y", 512.0)),
            border_size=float(d.get("border_size", 0.0)),
            posterize_strength=d.get("posterize_strength", None),
        )

    
    def with_overrides(self, overrides: "LayerSettingsOverride") -> "PatterningSettings":
        """Resolves effective settings for this layer by applying overrides on top of project defaults."""
        return PatterningSettings(
            exposure_time=overrides.exposure_time
            if overrides.exposure_time is not None
            else self.exposure_time,
            tiling_enabled=overrides.tiling_enabled
            if overrides.tiling_enabled is not None
            else self.tiling_enabled,
            tile_width=overrides.tile_width
            if overrides.tile_width is not None
            else self.tile_width,
            tile_height=overrides.tile_height
            if overrides.tile_height is not None
            else self.tile_height,
            overlap_x=overrides.overlap_x
            if overrides.overlap_x is not None
            else self.overlap_x,
            overlap_y=overrides.overlap_y
            if overrides.overlap_y is not None
            else self.overlap_y,
            pitch_x=overrides.pitch_x
            if overrides.pitch_x is not None
            else self.pitch_x,
            pitch_y=overrides.pitch_y
            if overrides.pitch_y is not None
            else self.pitch_y,
            border_size=overrides.border_size
            if overrides.border_size is not None
            else self.border_size,
            posterize_strength=overrides.posterize_strength
            if overrides.posterize_strength is not None
            else self.posterize_strength,
        )


@dataclass
class LayerSettingsOverride:
    """Optional per-layer overrides for patterning settings."""

    exposure_time: Optional[float] = None
    tiling_enabled: Optional[bool] = None
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    overlap_x: Optional[int] = None
    overlap_y: Optional[int] = None
    pitch_x: Optional[float] = None
    pitch_y: Optional[float] = None
    border_size: Optional[float] = None
    posterize_strength: Optional[int] = None

    def to_disk(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_disk(cls, d: dict) -> "LayerSettingsOverride":
        return cls(
            exposure_time=float(d["exposure_time"]) if "exposure_time" in d else None,
            tiling_enabled=bool(d["tiling_enabled"]) if "tiling_enabled" in d else None,
            tile_width=int(d["tile_width"]) if "tile_width" in d else None,
            tile_height=int(d["tile_height"]) if "tile_height" in d else None,
            overlap_x=int(d["overlap_x"]) if "overlap_x" in d else None,
            overlap_y=int(d["overlap_y"]) if "overlap_y" in d else None,
            pitch_x=float(d["pitch_x"]) if "pitch_x" in d else None,
            pitch_y=float(d["pitch_y"]) if "pitch_y" in d else None,
            border_size=float(d["border_size"]) if "border_size" in d else None,
            posterize_strength=int(d["posterize_strength"])
            if "posterize_strength" in d and d["posterize_strength"] is not None
            else None,
        )




@dataclass
class ChipLayer:
    """A single layer in a ChipProject with its own pattern and optional overrides."""

    name: str = "Layer 1"
    pattern_path: Optional[str] = None
    scale_w: int = -1
    scale_h: int = -1
    threshold: int = 50
    overrides: LayerSettingsOverride = field(default_factory=LayerSettingsOverride)

    # the original pattern image
    _pattern_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # sliced and rendered tiles. For tiling disabled layer, this contains a single tile (the full pattern)
    # the tiles are snake-ordered
    _tile_cache: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    _tile_coords: list[Tuple[float, float]] = field(default_factory=list, repr=False, compare=False)

    events: Optional[Any] = field(default=None, repr=False, compare=False)
    _project: Optional["ChipProject"] = field(default=None, repr=False, compare=False)

    def set_pattern_path(self, path: Optional[str]):
        self.pattern_path = path
        self._pattern_cache = None
        self._tile_cache = []
        self._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_threshold(self, threshold: int):
        self.threshold = threshold
        self._pattern_cache = None
        self._tile_cache = []
        self._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_exposure_override(self, exposure_time: Optional[float]):
        self.overrides.exposure_time = exposure_time
        self._tile_cache = []
        self._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_tiling_override(self, tiling_enabled: Optional[bool]):
        self.overrides.tiling_enabled = tiling_enabled
        self._tile_cache = []
        self._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def update_overrides(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.overrides, k):
                setattr(self.overrides, k, v)
        self._tile_cache = []
        self._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)


    def generate_tiles(
        self,
        force: bool = False,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """
        Call this method to generate the tile caches.
        """

        if force:
            self._pattern_cache = None
            self._tile_cache = []
            self._tile_coords = []

        if not force and self._tile_cache:
            return self._tile_cache, self._tile_coords

        assert self._project is not None
        project_settings = self._project.settings

        pattern = self.get_pattern_image()

        # no image set
        if pattern is None:
            self._tile_cache = []
            self._tile_coords = []
            if self.events is not None:
                self.events.emit(Event.LAYER_CACHE_RECOMPUTED, self)
            return self._tile_cache, self._tile_coords


        settings = project_settings.with_overrides(self.overrides)

        if settings.tiling_enabled:
            tiles, snake_coords = split_image_into_tiles(
                pattern,
                tile_width=settings.tile_width,
                tile_height=settings.tile_height,
                overlap_x=settings.overlap_x,
                overlap_y=settings.overlap_y,
            )
            
            self._tile_cache = tiles

            tile_coords: list[Tuple[float, float]] = []
            for x_idx, y_idx in snake_coords:
                tx = settings.pitch_x * x_idx
                ty = settings.pitch_y * y_idx
                tile_coords.append((tx, ty))
            self._tile_coords = tile_coords

        else:
            self._tile_cache = [pattern]
            self._tile_coords = [(0.0, 0.0)]


        if self.events is not None:
            self.events.emit(Event.LAYER_CACHE_RECOMPUTED, self)
        return self._tile_cache, self._tile_coords

    def get_tile(
        self,
        index: int
    ) -> Optional[tuple[np.ndarray, tuple[float, float]]]:
        if not self._tile_cache:
            return None
        if 0 <= index < len(self._tile_cache):
            return self._tile_cache[index], self._tile_coords[index]
        return self._tile_cache[0], self._tile_coords[0]

    def get_pattern_image(self) -> Optional[np.ndarray]:
        if self._pattern_cache is None:
            if self.pattern_path and os.path.exists(self.pattern_path):
                try:
                    raw = cv2.imread(self.pattern_path, cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        if raw.ndim == 2:
                            img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                        elif raw.shape[2] == 4:
                            img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                        elif raw.shape[2] == 3:
                            img = raw
                        else:
                            img = raw

                        # Thresholding to black and white image if enabled
                        if self.threshold != -1:
                            mask = np.any(img[..., :3] > self.threshold, axis=-1)
                            bw = np.zeros_like(img)
                            bw[mask] = 255
                            img = bw

                        # Scale right after image loading and store in _pattern_cache
                        if self.scale_w > 0 and self.scale_h > 0:
                            interp = cv2.INTER_NEAREST if self.threshold != -1 else cv2.INTER_LINEAR
                            img = cv2.resize(img, (self.scale_w, self.scale_h), interpolation=interp)
                        elif self.scale_w > 0 and self.scale_h == -1:
                            orig_h, orig_w = img.shape[:2]
                            target_h = max(1, int(round(orig_h * (self.scale_w / orig_w))))
                            interp = cv2.INTER_NEAREST if self.threshold != -1 else cv2.INTER_LINEAR
                            img = cv2.resize(img, (self.scale_w, target_h), interpolation=interp)
                        elif self.scale_h > 0 and self.scale_w == -1:
                            orig_h, orig_w = img.shape[:2]
                            target_w = max(1, int(round(orig_w * (self.scale_h / orig_h))))
                            interp = cv2.INTER_NEAREST if self.threshold != -1 else cv2.INTER_LINEAR
                            img = cv2.resize(img, (target_w, self.scale_h), interpolation=interp)

                        self._pattern_cache = img
                    else:
                        self._pattern_cache = None
                except Exception as e:
                    print(f"Error loading pattern image from {self.pattern_path}: {e}")
                    self._pattern_cache = None
            else:
                self._pattern_cache = None
        return self._pattern_cache

    def to_disk(self) -> dict:
        return {
            "name": self.name,
            "pattern_path": self.pattern_path,
            "scale_w": self.scale_w,
            "scale_h": self.scale_h,
            "threshold": self.threshold,
            "overrides": self.overrides.to_disk(),
        }

    @classmethod
    def from_disk(cls, d: dict) -> "ChipLayer":
        name = d.get("name", "Layer")
        pattern_path = d.get("pattern_path", None)
        scale_w = int(d.get("scale_w", -1))
        scale_h = int(d.get("scale_h", -1))
        threshold = int(d.get("threshold", 50))
        overrides = LayerSettingsOverride.from_disk(d.get("overrides", {}))
        return cls(
            name=name,
            pattern_path=pattern_path,
            scale_w=scale_w,
            scale_h=scale_h,
            threshold=threshold,
            overrides=overrides,
        )



@dataclass
class ChipProject:
    """Project-level data structure managing patterning settings and layers."""

    name: str = "Untitled Project"
    settings: PatterningSettings = field(default_factory=PatterningSettings)
    layers: List[ChipLayer] = field(default_factory=lambda: [ChipLayer(name="Layer 1")])
    active_layer_index: int = 0
    active_tile_index: int = 0
    exposure_history: List[ExposureRecord] = field(default_factory=list, repr=False, compare=False)
    events: Optional[EventBus] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.layers:
            self.layers = [ChipLayer(name="Layer 1", events=self.events, _project=self)]
        for layer in self.layers:
            layer.events = self.events
            layer._project = self
        if self.active_layer_index >= len(self.layers):
            self.active_layer_index = max(0, len(self.layers) - 1)

    def set_eventbus(self, events: Optional[Any]) -> None:
        self.events = events
        for layer in self.layers:
            layer.events = events
            layer._project = self

    @property
    def active_layer(self) -> ChipLayer:
        return self.layers[self.active_layer_index]

    @property
    def active_tile(self) -> tuple[np.ndarray, tuple[float, float]] | None:
        return self.active_layer.get_tile(self.active_tile_index)



    def add_layer(self, name: Optional[str] = None) -> ChipLayer:
        layer_num = len(self.layers) + 1
        layer_name = name or f"Layer {layer_num}"
        layer = ChipLayer(name=layer_name, events=self.events, _project=self)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        self.active_tile_index = 0
        if self.events is not None:
            from core.events import Event
            self.events.emit(Event.PROJECT_CHANGED, self)
            self.events.emit(Event.ACTIVE_LAYER_CHANGED, self.active_layer_index)
            self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
        return layer

    def remove_layer(self, index: int) -> bool:
        if len(self.layers) <= 1:
            raise ValueError("A chip project must contain at least one layer.")
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            if self.active_layer_index >= len(self.layers):
                self.active_layer_index = len(self.layers) - 1
            self.active_tile_index = 0
            if self.events is not None:
                from core.events import Event
                self.events.emit(Event.PROJECT_CHANGED, self)
                self.events.emit(Event.ACTIVE_LAYER_CHANGED, self.active_layer_index)
                self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
            return True
        return False

    def select_layer(self, index: int) -> bool:
        if 0 <= index < len(self.layers):
            self.active_layer_index = index
            self.active_tile_index = 0
            if self.events is not None:
                from core.events import Event
                self.events.emit(Event.ACTIVE_LAYER_CHANGED, self.active_layer_index)
                self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
            return True
        return False

    def select_tile(self, index: int) -> bool:
        self.active_tile_index = max(0, index)
        if self.events is not None:
            from core.events import Event
            self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
        return True



    def add_exposure_record(self, record: ExposureRecord) -> None:
        self.exposure_history.append(record)
        if self.events is not None:
            from core.events import Event
            self.events.emit(Event.EXPOSURE_HISTORY_CHANGED, self.exposure_history)

    def clear_exposure_history(self) -> None:
        self.exposure_history.clear()
        if self.events is not None:
            from core.events import Event
            self.events.emit(Event.EXPOSURE_HISTORY_CHANGED, self.exposure_history)



    def update_settings(self, **kwargs) -> None:
        """Updates project settings, clears all layer tile caches, and emits EXPOSURE_CONFIG_CHANGED."""
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        for layer in self.layers:
            layer._tile_cache = []
            layer._tile_coords = []
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def get_active_layer_tile_count(self) -> int:
        tiles, coords = self.active_layer.generate_tiles()
        return len(tiles)

    def to_disk(self) -> dict:
        return {
            "name": self.name,
            "settings": self.settings.to_disk(),
            "layers": [layer.to_disk() for layer in self.layers],
            "active_layer_index": self.active_layer_index,
            "active_tile_index": self.active_tile_index,
        }

    @classmethod
    def from_disk(cls, d: dict, events: Optional[Any] = None) -> "ChipProject":
        name = d.get("name", "Untitled Project")
        settings = PatterningSettings.from_disk(d.get("settings", {}))
        layers = [ChipLayer.from_disk(l) for l in d.get("layers", [])]
        active_idx = int(d.get("active_layer_index", 0))
        active_tile = int(d.get("active_tile_index", 0))
        return cls(
            name=name,
            settings=settings,
            layers=layers,
            active_layer_index=active_idx,
            active_tile_index=active_tile,
            events=events,
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_disk(), f, indent=2)

    @classmethod
    def load(cls, filepath: str, events: Optional[Any] = None) -> "ChipProject":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_disk(data, events=events)

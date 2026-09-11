import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Any
from core.events import Event

from PIL import Image



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
    image_adjust: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (shift_x, shift_y, theta)
    overrides: LayerSettingsOverride = field(default_factory=LayerSettingsOverride)

    # the original pattern image
    _pattern_cache: Optional[Image.Image] = field(default=None, repr=False, compare=False)
    _pattern_cache_dirty: bool = field(default=False, repr=False, compare=False)

    # sliced and rendered tiles. For tiling disabled layer, this contains a single tile (the full pattern)
    # the tiles are snake-ordered
    _tile_cache: list[Image.Image] = field(default_factory=list, repr=False, compare=False)
    _tile_cache_dirty: bool = field(default=False, repr=False, compare=False)

    events: Optional[Any] = field(default=None, repr=False, compare=False)

    def set_pattern_path(self, path: Optional[str]):
        self.pattern_path = path
        self._pattern_cache = None
        self._pattern_cache_dirty = True
        self._tile_cache = []
        self._tile_cache_dirty = True
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_image_adjust(self, adjust: Tuple[float, float, float]):
        self.image_adjust = adjust
        self._tile_cache = []
        self._tile_cache_dirty = True
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_overrides(self, overrides: LayerSettingsOverride):
        self.overrides = overrides
        self._tile_cache = []
        self._tile_cache_dirty = True
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_exposure_override(self, exposure_time: Optional[float]):
        self.overrides.exposure_time = exposure_time
        self.mark_dirty()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_tiling_override(self, tiling_enabled: Optional[bool]):
        self.overrides.tiling_enabled = tiling_enabled
        self.mark_dirty()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def update_overrides(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.overrides, k):
                setattr(self.overrides, k, v)
        self.mark_dirty()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def mark_dirty(self):
        self._pattern_cache_dirty = True
        self._tile_cache_dirty = True
        self._tile_cache = []

    def regenerate_tiles(self):
        self.mark_dirty()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def get_pattern_image(self) -> Optional[Image.Image]:
        if self._pattern_cache_dirty or self._pattern_cache is None:
            if self.pattern_path and os.path.exists(self.pattern_path):
                try:
                    self._pattern_cache = Image.open(self.pattern_path)
                except Exception as e:
                    print(f"Error loading pattern image from {self.pattern_path}: {e}")
                    self._pattern_cache = None
            else:
                self._pattern_cache = None
            self._pattern_cache_dirty = False
        return self._pattern_cache

    def get_tiles(
        self,
        project_settings: PatterningSettings,
        projector_size: Tuple[int, int],
    ) -> list[Image.Image]:
        """Returns the list of snake-ordered rendered tiles for this layer."""
        if self._tile_cache_dirty or not self._tile_cache:
            pattern = self.get_pattern_image()
            if pattern is None:
                self._tile_cache = []
                self._tile_cache_dirty = False
                return self._tile_cache

            eff = self.get_effective_settings(project_settings)
            from lib.img import ImageProcessSettings, process_img

            if eff.tiling_enabled:
                from lib.tiling import split_image_into_tiles

                tiles, _, _ = split_image_into_tiles(
                    pattern,
                    tile_width=eff.tile_width,
                    tile_height=eff.tile_height,
                    overlap_x=eff.overlap_x,
                    overlap_y=eff.overlap_y,
                )
                proc_settings = ImageProcessSettings(
                    posterization=eff.posterize_strength,
                    flatfield=None,
                    color_channels=(True, True, True),
                    size=projector_size,
                    image_adjust=self.image_adjust,
                    border_size=eff.border_size,
                )
                self._tile_cache = [process_img(t, proc_settings) for t in tiles]
            else:
                proc_settings = ImageProcessSettings(
                    posterization=eff.posterize_strength,
                    flatfield=None,
                    color_channels=(True, True, True),
                    size=projector_size,
                    image_adjust=self.image_adjust,
                    border_size=eff.border_size,
                )
                self._tile_cache = [process_img(pattern, proc_settings)]

            self._tile_cache_dirty = False

        return self._tile_cache

    def get_tile(
        self,
        index: int,
        project_settings: PatterningSettings,
        projector_size: Tuple[int, int],
    ) -> Optional[Image.Image]:
        tiles = self.get_tiles(project_settings, projector_size)
        if not tiles:
            return None
        if 0 <= index < len(tiles):
            return tiles[index]
        return tiles[0]

    def get_effective_settings(self, project_settings: PatterningSettings) -> PatterningSettings:
        """Resolves effective settings for this layer by applying overrides on top of project defaults."""
        return PatterningSettings(
            exposure_time=self.overrides.exposure_time
            if self.overrides.exposure_time is not None
            else project_settings.exposure_time,
            tiling_enabled=self.overrides.tiling_enabled
            if self.overrides.tiling_enabled is not None
            else project_settings.tiling_enabled,
            tile_width=self.overrides.tile_width
            if self.overrides.tile_width is not None
            else project_settings.tile_width,
            tile_height=self.overrides.tile_height
            if self.overrides.tile_height is not None
            else project_settings.tile_height,
            overlap_x=self.overrides.overlap_x
            if self.overrides.overlap_x is not None
            else project_settings.overlap_x,
            overlap_y=self.overrides.overlap_y
            if self.overrides.overlap_y is not None
            else project_settings.overlap_y,
            pitch_x=self.overrides.pitch_x
            if self.overrides.pitch_x is not None
            else project_settings.pitch_x,
            pitch_y=self.overrides.pitch_y
            if self.overrides.pitch_y is not None
            else project_settings.pitch_y,
            border_size=self.overrides.border_size
            if self.overrides.border_size is not None
            else project_settings.border_size,
            posterize_strength=self.overrides.posterize_strength
            if self.overrides.posterize_strength is not None
            else project_settings.posterize_strength,
        )

    def to_disk(self) -> dict:
        return {
            "name": self.name,
            "pattern_path": self.pattern_path,
            "image_adjust": list(self.image_adjust),
            "overrides": self.overrides.to_disk(),
        }

    @classmethod
    def from_disk(cls, d: dict) -> "ChipLayer":
        name = d.get("name", "Layer")
        pattern_path = d.get("pattern_path", None)
        image_adjust = tuple(d.get("image_adjust", (0.0, 0.0, 0.0)))
        overrides = LayerSettingsOverride.from_disk(d.get("overrides", {}))
        return cls(
            name=name,
            pattern_path=pattern_path,
            image_adjust=image_adjust,
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
    events: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.layers:
            self.layers = [ChipLayer(name="Layer 1", events=self.events)]
        for layer in self.layers:
            layer.events = self.events
        if self.active_layer_index >= len(self.layers):
            self.active_layer_index = max(0, len(self.layers) - 1)

    def set_events(self, events: Optional[Any]) -> None:
        self.events = events
        for layer in self.layers:
            layer.events = events

    @property
    def active_layer(self) -> ChipLayer:
        return self.layers[self.active_layer_index]

    def add_layer(self, name: Optional[str] = None) -> ChipLayer:
        layer_num = len(self.layers) + 1
        layer_name = name or f"Layer {layer_num}"
        layer = ChipLayer(name=layer_name, events=self.events)
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

    def update_settings(self, **kwargs) -> None:
        """Updates project settings, marks all layer tile caches dirty, and emits EXPOSURE_CONFIG_CHANGED."""
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        for layer in self.layers:
            layer.mark_dirty()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def get_active_layer_tile_count(self, projector_size: Tuple[int, int] = (1920, 1080)) -> int:
        tiles = self.active_layer.get_tiles(self.settings, projector_size)
        return len(tiles)

    def render_for_projector(
        self,
        color_mode: Any,
        image_source: Any,
        custom_path: Optional[str] = None,
        projector_size: Tuple[int, int] = (1920, 1080),
    ) -> Optional[Image.Image]:
        """Renders the appropriate image frame for the projector."""
        from core.events import ColorMode, ProjectorImageSource
        from lib.img import select_channels

        if color_mode == ColorMode.DISABLE:
            return None

        img: Optional[Image.Image] = None

        if image_source == ProjectorImageSource.ACTIVE_LAYER:
            tile = self.active_layer.get_tile(self.active_tile_index, self.settings, projector_size)
            if tile is None:
                return None
            img = tile.copy()
        elif image_source == ProjectorImageSource.CUSTOM_FILE:
            if custom_path and os.path.exists(custom_path):
                try:
                    raw = Image.open(custom_path)
                    from lib.img import ImageProcessSettings, process_img

                    proc_settings = ImageProcessSettings(
                        posterization=None,
                        flatfield=None,
                        color_channels=(True, True, True),
                        size=projector_size,
                        image_adjust=(0.0, 0.0, 0.0),
                        border_size=0.0,
                    )
                    img = process_img(raw, proc_settings)
                except Exception as e:
                    print(f"Error loading custom image from {custom_path}: {e}")
                    return None
            else:
                return None

        if img is None:
            return None

        if color_mode == ColorMode.RED:
            return select_channels(img, red=True, green=False, blue=False)
        elif color_mode == ColorMode.UV:
            return select_channels(img, red=False, green=False, blue=True)

        return img

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

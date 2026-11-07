import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from core.coordinate_system import (
    apply_homography_2d,
    compute_two_point_rigid_transform,
)
from core.events import Event, EventBus


@dataclass
class ExposureRecord:
    """Record of a single exposure event in chip-space physical units (µm)."""

    coords: Tuple[float, float, float]
    time: datetime = field(default_factory=datetime.now)
    duration: float = 0.0  # ms
    aborted: bool = False
    layer_index: Optional[int] = None
    tile_index: Optional[int] = None
    footprint_chip_corners: Optional[List[Tuple[float, float]]] = None

    def to_disk(self) -> dict:
        d = {
            "coords": [float(self.coords[0]), float(self.coords[1]), float(self.coords[2])],
            "time": self.time.isoformat() if isinstance(self.time, datetime) else str(self.time),
            "duration": float(self.duration),
            "aborted": bool(self.aborted),
            "layer_index": self.layer_index,
            "tile_index": self.tile_index,
        }
        if self.footprint_chip_corners is not None:
            d["footprint_chip_corners"] = [[float(pt[0]), float(pt[1])] for pt in self.footprint_chip_corners]
        return d

    @classmethod
    def from_disk(cls, d: dict) -> "ExposureRecord":
        raw_coords = d.get("coords", [0.0, 0.0, 0.0])
        coords = (
            float(raw_coords[0]) if len(raw_coords) > 0 else 0.0,
            float(raw_coords[1]) if len(raw_coords) > 1 else 0.0,
            float(raw_coords[2]) if len(raw_coords) > 2 else 0.0,
        )
        time_str = d.get("time", "")
        try:
            dt = datetime.fromisoformat(time_str) if time_str else datetime.now()
        except Exception:
            dt = datetime.now()
        raw_fp = d.get("footprint_chip_corners")
        fp = (
            [(float(pt[0]), float(pt[1])) for pt in raw_fp]
            if isinstance(raw_fp, list) and len(raw_fp) == 4
            else None
        )
        return cls(
            coords=coords,
            time=dt,
            duration=float(d.get("duration", 0.0)),
            aborted=bool(d.get("aborted", False)),
            layer_index=int(d["layer_index"]) if d.get("layer_index") is not None else None,
            tile_index=int(d["tile_index"]) if d.get("tile_index") is not None else None,
            footprint_chip_corners=fp,
        )


@dataclass
class ChipProgress:
    """Stores chip-space marker calibration, pattern origins, and exposure records."""

    marker1_chip: Optional[Tuple[float, float]] = None
    marker2_chip: Optional[Tuple[float, float]] = None
    marker1_corr: Optional[Tuple[float, float]] = None
    marker2_corr: Optional[Tuple[float, float]] = None
    pattern_origins_chip: List[Tuple[float, float]] = field(default_factory=list)
    pattern_sizes_chip: List[Tuple[float, float]] = field(default_factory=list)
    pattern_transforms_chip: List[Optional[List[List[float]]]] = field(default_factory=list)
    exposure_history: List[ExposureRecord] = field(default_factory=list)
    events: Optional[EventBus] = field(default=None, repr=False, compare=False)

    def has_reference_markers(self) -> bool:
        """Returns True if both reference marker positions are stored in chip-space."""
        return self.marker1_chip is not None and self.marker2_chip is not None

    def is_aligned(self) -> bool:
        """Returns True if both reference markers and measured corr-stage markers are set."""
        return (
            self.marker1_chip is not None
            and self.marker2_chip is not None
            and self.marker1_corr is not None
            and self.marker2_corr is not None
        )

    def locate_marker(self, idx: int, corr_xy: Tuple[float, float]) -> str:
        """Records or updates marker 1 or 2 position from corr-stage-space coordinate.

        When chip progress does not yet have both reference markers in chip-space,
        keeps the identity chip-space transform and sets the reference marker coordinate.
        When both reference markers are already stored in chip-space, updates the measured
        corr-stage coordinate and computes the corr-stage -> chip-space transform.

        Returns:
            Status message describing the action taken.
        """
        if idx not in (1, 2):
            raise ValueError(f"Marker index must be 1 or 2, got {idx}")

        pt = (float(corr_xy[0]), float(corr_xy[1]))
        if not self.has_reference_markers():
            if idx == 1:
                self.marker1_chip = pt
                self.marker1_corr = pt
            else:
                self.marker2_chip = pt
                self.marker2_corr = pt
            msg = f"Recorded reference Marker {idx} in chip-space at ({pt[0]:.1f}, {pt[1]:.1f}) µm (Identity transform)."
        else:
            if idx == 1:
                self.marker1_corr = pt
            else:
                self.marker2_corr = pt
            rot_deg = self.get_rotation_deg()
            msg = (
                f"Relocated Marker {idx} at ({pt[0]:.1f}, {pt[1]:.1f}) µm. "
                f"Chip transform updated (rotation: {rot_deg:+.2f}°)."
            )

        if self.events is not None:
            self.events.emit(Event.CHIP_PROGRESS_CHANGED, self)
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)
        return msg

    def get_corr_to_chip_matrix(self) -> np.ndarray:
        """Returns the 3x3 rigid transform mapping corr-stage-space (µm) -> chip-space (µm)."""
        if self.is_aligned():
            assert self.marker1_corr is not None and self.marker2_corr is not None
            assert self.marker1_chip is not None and self.marker2_chip is not None
            return compute_two_point_rigid_transform(
                self.marker1_corr,
                self.marker2_corr,
                self.marker1_chip,
                self.marker2_chip,
            )
        return np.eye(3, dtype=np.float64)

    def get_chip_to_corr_matrix(self) -> np.ndarray:
        """Returns the 3x3 rigid transform mapping chip-space (µm) -> corr-stage-space (µm)."""
        return np.linalg.inv(self.get_corr_to_chip_matrix())

    def get_rotation_deg(self) -> float:
        """Returns the rotation angle in degrees from corr-stage-space to chip-space."""
        T = self.get_corr_to_chip_matrix()
        return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))

    def corr_to_chip(self, corr_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Transforms a 2D point from corr-stage-space (µm) to chip-space (µm)."""
        res = apply_homography_2d(corr_xy, self.get_corr_to_chip_matrix())
        return (float(res[0]), float(res[1]))

    def chip_to_corr(self, chip_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Transforms a 2D point from chip-space (µm) to corr-stage-space (µm)."""
        res = apply_homography_2d(chip_xy, self.get_chip_to_corr_matrix())
        return (float(res[0]), float(res[1]))

    def add_pattern_origin(
        self,
        chip_xy: Tuple[float, float],
        size_um: Optional[Tuple[float, float]] = None,
        transform_chip: Optional[Any] = None,
    ) -> int:
        """Appends a pattern origin, optional physical size, and optional 3x3 pattern-to-chip transform in chip-space (µm) and returns its 0-based index."""
        pt = (float(chip_xy[0]), float(chip_xy[1]))
        sz = (
            (float(size_um[0]), float(size_um[1]))
            if size_um is not None and len(size_um) >= 2
            else (-1.0, -1.0)
        )
        t_list: Optional[List[List[float]]] = None
        if transform_chip is not None:
            arr = np.asarray(transform_chip, dtype=np.float64)
            if arr.shape == (3, 3):
                t_list = [[float(v) for v in row] for row in arr]

        self.pattern_origins_chip.append(pt)
        self.pattern_sizes_chip.append(sz)
        self.pattern_transforms_chip.append(t_list)
        if self.events is not None:
            self.events.emit(Event.CHIP_PROGRESS_CHANGED, self)
        return len(self.pattern_origins_chip) - 1

    def get_pattern_transform_chip(self, origin_index: int) -> np.ndarray:
        """Returns the 3x3 matrix mapping local pattern coordinates (µm) to local chip-space offsets (µm) for the given origin."""
        if 0 <= origin_index < len(self.pattern_transforms_chip):
            mat = self.pattern_transforms_chip[origin_index]
            if mat is not None:
                arr = np.asarray(mat, dtype=np.float64)
                if arr.shape == (3, 3):
                    return arr
        return np.eye(3, dtype=np.float64)

    def add_exposure_record(self, record: ExposureRecord) -> None:
        self.exposure_history.append(record)
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_HISTORY_CHANGED, self.exposure_history)
            self.events.emit(Event.CHIP_PROGRESS_CHANGED, self)

    def clear_exposure_history(self) -> None:
        self.exposure_history.clear()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_HISTORY_CHANGED, self.exposure_history)
            self.events.emit(Event.CHIP_PROGRESS_CHANGED, self)

    def reset(self) -> None:
        """Resets all markers, pattern origins, and exposure records."""
        self.marker1_chip = None
        self.marker2_chip = None
        self.marker1_corr = None
        self.marker2_corr = None
        self.pattern_origins_chip.clear()
        self.pattern_sizes_chip.clear()
        self.pattern_transforms_chip.clear()
        self.exposure_history.clear()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_HISTORY_CHANGED, self.exposure_history)
            self.events.emit(Event.CHIP_PROGRESS_CHANGED, self)
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def to_disk(self) -> dict:
        return {
            "marker1_chip": list(self.marker1_chip) if self.marker1_chip is not None else None,
            "marker2_chip": list(self.marker2_chip) if self.marker2_chip is not None else None,
            "marker1_corr": list(self.marker1_corr) if self.marker1_corr is not None else None,
            "marker2_corr": list(self.marker2_corr) if self.marker2_corr is not None else None,
            "pattern_origins_chip": [[float(p[0]), float(p[1])] for p in self.pattern_origins_chip],
            "pattern_sizes_chip": [[float(s[0]), float(s[1])] for s in self.pattern_sizes_chip],
            "pattern_transforms_chip": [
                [[float(v) for v in row] for row in t] if t is not None else None
                for t in self.pattern_transforms_chip
            ],
            "exposure_history": [rec.to_disk() for rec in self.exposure_history],
        }

    @classmethod
    def from_disk(cls, d: dict, events: Optional[EventBus] = None) -> "ChipProgress":
        def _parse_pt(val) -> Optional[Tuple[float, float]]:
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
            return None

        m1_chip = _parse_pt(d.get("marker1_chip"))
        m2_chip = _parse_pt(d.get("marker2_chip"))
        m1_corr = _parse_pt(d.get("marker1_corr")) or m1_chip
        m2_corr = _parse_pt(d.get("marker2_corr")) or m2_chip

        origins = [
            (float(p[0]), float(p[1]))
            for p in d.get("pattern_origins_chip", [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        sizes = [
            (float(s[0]), float(s[1]))
            for s in d.get("pattern_sizes_chip", [])
            if isinstance(s, (list, tuple)) and len(s) >= 2
        ]
        while len(sizes) < len(origins):
            sizes.append((-1.0, -1.0))

        raw_transforms = d.get("pattern_transforms_chip", [])
        transforms: List[Optional[List[List[float]]]] = []
        if isinstance(raw_transforms, list):
            for t in raw_transforms:
                if isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in t):
                    transforms.append([[float(v) for v in r] for r in t])
                else:
                    transforms.append(None)
        while len(transforms) < len(origins):
            transforms.append(None)

        history = [
            ExposureRecord.from_disk(r)
            for r in d.get("exposure_history", [])
            if isinstance(r, dict)
        ]
        return cls(
            marker1_chip=m1_chip,
            marker2_chip=m2_chip,
            marker1_corr=m1_corr,
            marker2_corr=m2_corr,
            pattern_origins_chip=origins,
            pattern_sizes_chip=sizes,
            pattern_transforms_chip=transforms,
            exposure_history=history,
            events=events,
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_disk(), f, indent=2)

    @classmethod
    def load(cls, filepath: str, events: Optional[EventBus] = None) -> "ChipProgress":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_disk(data, events=events)


@dataclass
class PatterningSettings:
    """Project-level default patterning and overlap settings in physical units."""

    exposure_time: float = 8000.0  # ms
    overlap_x: float = 5.0  # µm
    overlap_y: float = 5.0  # µm
    width_um: float = -1.0  # µm (-1 when height_um == -1 uses Fullscreen Projector mode)
    height_um: float = -1.0  # µm

    def to_disk(self) -> dict:
        return {
            "exposure_time": self.exposure_time,
            "overlap_x": self.overlap_x,
            "overlap_y": self.overlap_y,
            "width_um": self.width_um,
            "height_um": self.height_um,
        }

    @classmethod
    def from_disk(cls, d: dict) -> "PatterningSettings":
        return cls(
            exposure_time=float(d.get("exposure_time", 8000.0)),
            overlap_x=float(d.get("overlap_x", 5.0)),
            overlap_y=float(d.get("overlap_y", 5.0)),
            width_um=float(d.get("width_um", -1.0)),
            height_um=float(d.get("height_um", -1.0)),
        )

    def with_overrides(self, overrides: "LayerSettingsOverride") -> "PatterningSettings":
        """Resolves effective settings for this layer by applying overrides on top of project defaults."""
        return PatterningSettings(
            exposure_time=overrides.exposure_time
            if overrides.exposure_time is not None
            else self.exposure_time,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            width_um=self.width_um,
            height_um=self.height_um,
        )


@dataclass
class LayerSettingsOverride:
    """Optional per-layer override for exposure time."""

    exposure_time: Optional[float] = None

    def to_disk(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_disk(cls, d: dict) -> "LayerSettingsOverride":
        return cls(
            exposure_time=float(d["exposure_time"]) if "exposure_time" in d and d["exposure_time"] is not None else None,
        )


@dataclass
class ChipLayer:
    """A single layer in a ChipProject with its pattern, physical dimensions (µm), and exposure override."""

    name: str = "Layer 1"
    pattern_path: Optional[str] = None
    width_um: float = -1.0
    height_um: float = -1.0
    threshold: int = 50
    overrides: LayerSettingsOverride = field(default_factory=LayerSettingsOverride)

    # the original pattern image (binarized if threshold != -1)
    _pattern_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # sliced and warped projector tiles in snake order
    _tile_cache: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    # tile center offsets (dx_chip_um, dy_chip_um) relative to pattern center in chip-space
    _tile_coords: list[Tuple[float, float]] = field(default_factory=list, repr=False, compare=False)

    events: Optional[Any] = field(default=None, repr=False, compare=False)
    _project: Optional["ChipProject"] = field(default=None, repr=False, compare=False)

    @property
    def width_um(self) -> float:
        if self._project is not None:
            return float(self._project.settings.width_um)
        return float(getattr(self, "_w_um", -1.0))

    @width_um.setter
    def width_um(self, val: Any):
        if isinstance(val, property):
            val = -1.0
        val_f = float(val)
        self._w_um = val_f
        if self._project is not None:
            self._project.set_pattern_size(val_f, self._project.settings.height_um)

    @property
    def height_um(self) -> float:
        if self._project is not None:
            return float(self._project.settings.height_um)
        return float(getattr(self, "_h_um", -1.0))

    @height_um.setter
    def height_um(self, val: Any):
        if isinstance(val, property):
            val = -1.0
        val_f = float(val)
        self._h_um = val_f
        if self._project is not None:
            self._project.set_pattern_size(self._project.settings.width_um, val_f)

    @property
    def is_fullscreen_mode(self) -> bool:
        """Returns True if layer is configured for uncalibrated Fullscreen Projector mode (-1 x -1)."""
        return self.width_um <= 0 and self.height_um <= 0

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

    def set_dimensions_um(self, width_um: float, height_um: float):
        if self._project is not None:
            self._project.set_pattern_size(float(width_um), float(height_um))
        else:
            self._width_um = float(width_um)
            self._height_um = float(height_um)
            self._tile_cache = []
            self._tile_coords = []
            if self.events is not None:
                self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_physical_size(self, width_um: float, height_um: float):
        self.set_dimensions_um(width_um, height_um)

    def get_effective_settings(self) -> PatterningSettings:
        if self._project is not None:
            return self._project.settings.with_overrides(self.overrides)
        return PatterningSettings().with_overrides(self.overrides)

    def set_exposure_override(self, exposure_time: Optional[float]):
        self.overrides.exposure_time = exposure_time
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
        single_fullscreen_tile: bool = False,
        pattern_transform_chip: Optional[Any] = None,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Generates projector-space tiles and their relative chip-space coordinates."""
        assert self._project is not None
        cached_single = getattr(self, "_tile_cache_single_fullscreen", False)
        cached_transform = getattr(self, "_tile_cache_pattern_transform", None)

        if not force and self._tile_cache and cached_single == single_fullscreen_tile:
            if pattern_transform_chip is None:
                return self._tile_cache, self._tile_coords
            if cached_transform is not None and np.allclose(
                np.asarray(pattern_transform_chip, dtype=np.float64),
                np.asarray(cached_transform, dtype=np.float64),
            ):
                return self._tile_cache, self._tile_coords

        self._pattern_cache = None
        self._tile_cache = []
        self._tile_coords = []

        project_settings = self._project.settings
        pattern = self.get_pattern_image()

        if pattern is None:
            self._tile_cache = []
            self._tile_coords = []
            self._tile_cache_single_fullscreen = single_fullscreen_tile
            self._tile_cache_pattern_transform = pattern_transform_chip
            if self.events is not None:
                self.events.emit(Event.LAYER_CACHE_RECOMPUTED, self)
            return self._tile_cache, self._tile_coords

        proj_size = self._project.projector_size
        cam_size = self._project.camera_size
        proj_px_size = 1.0
        cam_px_size = 1.0
        proj_to_cam_mat = None

        if self._project.machine_config is not None:
            cfg = self._project.machine_config
            proj_px_size = cfg.optics.projector_pixel_size
            cam_px_size = cfg.stage_calibration.camera_pixel_size
            proj_to_cam_mat = cfg.optics.proj_to_cam_matrix

        corr_to_chip_mat = (
            self._project.chip_progress.get_corr_to_chip_matrix()
            if self._project.chip_progress is not None
            else np.eye(3, dtype=np.float64)
        )

        from lib.tiling import generate_projector_aligned_tiles

        tiles, tile_coords = generate_projector_aligned_tiles(
            img=pattern,
            width_um=self.width_um,
            height_um=self.height_um,
            overlap_x_um=project_settings.overlap_x,
            overlap_y_um=project_settings.overlap_y,
            projector_size=proj_size,
            camera_size=cam_size,
            projector_pixel_size=proj_px_size,
            camera_pixel_size=cam_px_size,
            proj_to_cam_matrix=proj_to_cam_mat,
            corr_to_chip_matrix=corr_to_chip_mat,
            pattern_transform_chip=pattern_transform_chip,
            nearest_interp=(self.threshold != -1),
            single_fullscreen_tile=single_fullscreen_tile,
        )

        self._tile_cache = tiles
        self._tile_coords = tile_coords
        self._tile_cache_single_fullscreen = single_fullscreen_tile
        self._tile_cache_pattern_transform = (
            np.asarray(pattern_transform_chip, dtype=np.float64).copy()
            if pattern_transform_chip is not None
            else None
        )

        if self.events is not None:
            self.events.emit(Event.LAYER_CACHE_RECOMPUTED, self)
        return self._tile_cache, self._tile_coords

    def get_tile(
        self,
        index: int,
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
                        else:
                            img = raw

                        if self.threshold != -1:
                            mask = np.any(img[..., :3] > self.threshold, axis=-1)
                            bw = np.zeros_like(img)
                            bw[mask] = 255
                            img = bw

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
            "width_um": self.width_um,
            "height_um": self.height_um,
            "threshold": self.threshold,
            "overrides": self.overrides.to_disk(),
        }

    @classmethod
    def from_disk(cls, d: dict) -> "ChipLayer":
        name = d.get("name", "Layer")
        pattern_path = d.get("pattern_path", None)
        width_um = float(d.get("width_um", -1.0))
        height_um = float(d.get("height_um", -1.0))
        threshold = int(d.get("threshold", 50))
        overrides = LayerSettingsOverride.from_disk(d.get("overrides", {}))
        return cls(
            name=name,
            pattern_path=pattern_path,
            width_um=width_um,
            height_um=height_um,
            threshold=threshold,
            overrides=overrides,
        )


@dataclass
class ChipProject:
    """Project-level data structure managing patterning settings, layers, and chip progress."""

    name: str = "Untitled Project"
    settings: PatterningSettings = field(default_factory=PatterningSettings)
    layers: List[ChipLayer] = field(default_factory=lambda: [ChipLayer(name="Layer 1")])
    active_layer_index: int = 0
    active_tile_index: int = 0
    chip_progress: ChipProgress = field(default_factory=ChipProgress, repr=False, compare=False)
    events: Optional[EventBus] = field(default=None, repr=False, compare=False)
    machine_config: Optional[Any] = field(default=None, repr=False, compare=False)
    projector_size: Tuple[int, int] = field(default=(1920, 1080), repr=False, compare=False)
    camera_size: Tuple[int, int] = field(default=(1920, 1080), repr=False, compare=False)

    def __post_init__(self):
        if not self.layers:
            self.layers = [ChipLayer(name="Layer 1", events=self.events, _project=self)]
        # If any layer had width_um/height_um specified while settings was default, initialize settings
        for layer in self.layers:
            if self.settings.width_um <= 0 and getattr(layer, "_w_um", -1.0) > 0:
                self.settings.width_um = layer._w_um
            if self.settings.height_um <= 0 and getattr(layer, "_h_um", -1.0) > 0:
                self.settings.height_um = layer._h_um
            layer.events = self.events
            layer._project = self
        self.chip_progress.events = self.events
        if self.active_layer_index >= len(self.layers):
            self.active_layer_index = max(0, len(self.layers) - 1)

    def set_pattern_size(self, width_um: float, height_um: float):
        """Sets the project-wide pattern physical dimensions in µm for all layers."""
        self.settings.width_um = float(width_um)
        self.settings.height_um = float(height_um)
        self.invalidate_all_tile_caches()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def set_eventbus(self, events: Optional[Any]) -> None:
        self.events = events
        self.chip_progress.events = events
        for layer in self.layers:
            layer.events = events
            layer._project = self

    def invalidate_all_tile_caches(self) -> None:
        for layer in self.layers:
            layer._tile_cache = []
            layer._tile_coords = []
            layer._tile_cache_single_fullscreen = False

    @property
    def exposure_history(self) -> List[ExposureRecord]:
        return self.chip_progress.exposure_history

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
                self.events.emit(Event.ACTIVE_LAYER_CHANGED, self.active_layer_index)
                self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
            return True
        return False

    def select_tile(self, index: int) -> bool:
        self.active_tile_index = max(0, index)
        if self.events is not None:
            self.events.emit(Event.ACTIVE_TILE_CHANGED, self.active_tile_index)
        return True

    def add_exposure_record(self, record: ExposureRecord) -> None:
        self.chip_progress.add_exposure_record(record)

    def clear_exposure_history(self) -> None:
        self.chip_progress.clear_exposure_history()

    def update_settings(self, **kwargs) -> None:
        """Updates project settings, clears all layer tile caches, and emits EXPOSURE_CONFIG_CHANGED."""
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        self.invalidate_all_tile_caches()
        if self.events is not None:
            self.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def get_active_layer_tile_count(self) -> int:
        tiles, _ = self.active_layer.generate_tiles()
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
        settings_dict = d.get("settings", {})
        layers_data = d.get("layers", [])
        settings = PatterningSettings.from_disk(settings_dict)

        # Backwards compatibility: if width_um/height_um not present in settings, pull from first layer
        if ("width_um" not in settings_dict or "height_um" not in settings_dict) and layers_data:
            if "width_um" not in settings_dict:
                settings.width_um = float(layers_data[0].get("width_um", -1.0))
            if "height_um" not in settings_dict:
                settings.height_um = float(layers_data[0].get("height_um", -1.0))

        layers = [ChipLayer.from_disk(l) for l in layers_data]
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
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_disk(), f, indent=2)

    @classmethod
    def load(cls, filepath: str, events: Optional[Any] = None) -> "ChipProject":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_disk(data, events=events)

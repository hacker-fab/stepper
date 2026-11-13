import ctypes
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import time
import tkinter
import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import partial
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.ttk import Progressbar
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import serial
import toml
from PIL import Image, ImageOps, ImageTk
from ultralytics import YOLO

from camera.camera_module import CameraModule
from camera.webcam import Webcam
from hardware import ImageProcessSettings, Lithographer, ProcessedImage
from lib.alignment import pattern_tiles, yolo_detector
from lib.gui import FloatEntry, IntEntry, Thumbnail
from lib.img import (
    aruco_grid_image,
    crop_center,
    cv2_to_pil,
    exposure_mask,
    frame_difference,
    image_to_tk_image,
    load_image,
    sobel_focus_score,
)
from lib.mapping import (
    CAMERA_DEAD_S,
    CAMERA_LIT_FRACTION,
    DEFAULT_CAM_UM_PER_PX,
    FOCUS_SETTLE_S,
    FOCUS_STAGES,
    PROJ_CHANGE_MAX_FRAMES,
    PROJ_CHANGE_MIN_DN,
    PROJ_CHANGE_NOISE_FACTOR,
    MapConfig,
    MapResult,
    background_from_map,
    golden_section_focus,
    map_area,
    scan_area_um,
    scan_summary,
    status,
)
from lib.run_folder import RunRecorder, map_finished_piece
from lib.tiling import (
    PROJ_UM_PER_PX,
    SAMPLE_DIR,
    SAMPLE_SET,
    CrossGeometry,
    PlacementMap,
    TilingTimeModel,
    area_for_tiles,
    budget_report,
    build_placement_map,
    check_seams,
    measure_cross_geometry,
    render_placement,
    same_layout,
    spec_mismatches,
    tile_pattern,
)
from projector import TkProjector
from stage_control.grbl_stage import GrblStage
from stage_control.omm_stage import OMMStage
from stage_control.stage_controller import StageController

# Enable DPI awareness on Windows before any Tk() call so the UI isn't tiny on HiDPI displays
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[unresolved-attribute]
    except Exception:
        pass

# TODO: Don't hardcode
THUMBNAIL_SIZE: tuple[int, int] = (160, 90)
# The values set here are not used and instead come from the config file
DEFAULT_RED_EXPOSURE: float = 4167.0
DEFAULT_UV_EXPOSURE: float = 25000.0


def fetch_focus_score(camera_image, blue_only, ddepth=cv2.CV_64F, kernel_size=5, log=False):
    """fetch_focus_score: computes the laplacian focal score after some
    pre-processing of the camera image. The key is to detect the edges better
    than other parts of the image that might not be suitable to be focused on."""

    camera_image = camera_image.copy()
    camera_image[:, :, 1] = 0  # green should never be used for focus
    if blue_only:
        camera_image[:, :, 0] = 0  # disable red

    src = camera_image
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)

    # Apply Laplace function
    src = cv2.Laplacian(src, ddepth, ksize=kernel_size)

    return src.var()


def compute_focus_score(camera_image, blue_only, save=False):
    camera_image = camera_image.copy()
    camera_image[:, :, 1] = 0  # green should never be used for focus
    if blue_only:
        camera_image[:, :, 0] = 0  # disable red
    img = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    mean = np.sum(img) / (img.shape[0] * img.shape[1])
    img_lapl = (
        np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1))
        + np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1))
    ) / mean
    if save:
        print('saved focus: ', np.min(img_lapl), np.max(img_lapl))
        cv2.imwrite(save, img_lapl * 255.0 / 5.0)
    return img_lapl.var() / mean


def detect_alignment_markers(model, image, draw_rectangle=False):
    detections = []
    display_image = image.copy()
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (640, 640))
        results = model(resized)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * original_width / 640)
            x2 = int(x2 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            y2 = int(y2 * original_height / 640)
            detections.append(((x1, y1), (x2, y2)))
            print('mark at ', (x1 + x2) / 2, (y1 + y2) / 2)
            if draw_rectangle:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    except Exception as e:
        print(f"Detection failed: {e}")

    return detections, display_image


class StrAutoEnum(str, Enum):
    """Base class for string-valued enums that use auto()"""

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> str:
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
    XY_LOCKED = auto()  # Only Z (focus) is free to move to avoid smearing UV focus pattern
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
    dlpc: Optional[object] = None  # DLPC instance if USB projector control is enabled


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


class EventDispatcher:
    hardware: Lithographer
    root: Tk
    model: Optional[YOLO]
    camera: Optional[CameraModule]
    red_focus: ProcessedImage
    uv_focus: ProcessedImage
    pattern: ProcessedImage
    pattern_image: Image.Image
    red_focus_image: Image.Image
    uv_focus_image: Image.Image
    solid_red_image: Image.Image
    image_adjust_position: tuple[float, float, float]
    border_size: float
    posterize_strength: Optional[int]
    red_focus_source: RedFocusSource
    stage_setpoint: tuple[float, float, float]
    shown_image: ShownImage
    autofocus_busy: bool
    patterning_busy: bool
    autofocus_on_mode_switch: bool
    realtime_detection: bool
    first_autofocus: bool
    should_abort: bool
    exposure_time: int
    patterning_progress: float  # ranges from 0.0 to 1.0
    red_exposure_time: float
    uv_exposure_time: float
    exposure_history: List[ExposureLog]
    chip: Chip
    auto_snapshot_on_uv: bool
    snapshot_directory: Path
    listeners: dict[Event, List[Callable]]

    def __init__(
        self,
        stage: StageController,
        proj: TkProjector,
        root: Tk,
        camera: Optional[CameraModule],
        red_exposure: float,
        uv_exposure: float,
        dlpc=None,
    ):
        # Hardware components
        self.hardware = Lithographer(stage, proj)
        self.camera = camera
        self.camera_image_seq = 0  # frame number of camera_image (see set_latest_image)
        self.root = root
        self.dlpc = dlpc

        # Detection model
        self.model = None

        # Image processing objects
        self.red_focus = ProcessedImage()
        self.uv_focus = ProcessedImage()
        self.pattern = ProcessedImage()

        # Source images
        self.pattern_image = Image.new("RGB", (1, 1), "black")
        self.pattern_image_path = ""
        self.red_focus_image = Image.new("RGB", (1, 1), "black")
        self.uv_focus_image = Image.new("RGB", (1, 1), "black")
        self.solid_red_image = Image.new("RGB", (1, 1), "red")

        # Image settings
        self.image_adjust_position = (0.0, 0.0, 0.0)
        self.border_size = 0.0
        self.posterize_strength = None
        self.red_focus_source = RedFocusSource.IMAGE

        # Stage control
        self.stage_setpoint = (0.0, 0.0, 0.0)

        # Status flags
        self.shown_image = ShownImage.CLEAR
        self.autofocus_busy = False
        self.patterning_busy = False
        self.autofocus_on_mode_switch = False
        self.realtime_detection = False
        self.first_autofocus = True
        self.should_abort = False

        # Exposure settings and progress
        self.exposure_time = 8000
        self.patterning_progress = 0.0
        self.red_exposure_time = red_exposure
        self.uv_exposure_time = uv_exposure

        # History and logging
        self.exposure_history = []
        self.chip = Chip([ChipLayer([])])

        # Snapshot settings
        self.auto_snapshot_on_uv = True
        self.snapshot_directory = Path("stepper_captures")
        self.snapshot_directory.mkdir(exist_ok=True)

        # Event handling
        self.listeners = dict()
        self.add_event_listener(Event.SHOWN_IMAGE_CHANGED, lambda: self._update_projector())

    def load_chip(self, path: str):
        print(f"Loading chip at {path!r}")
        with open(path, "r") as f:
            d = json.load(f)
        self.chip = Chip.from_disk(d)
        self.on_event(Event.CHIP_CHANGED)

    def new_chip(self):
        # TODO: Prompt user to save old chip??
        self.chip = Chip([ChipLayer([])])
        self.on_event(Event.CHIP_CHANGED)

    def add_chip_layer(self):
        self.chip.layers.append(ChipLayer([]))
        self.on_event(Event.CHIP_CHANGED)

    def save_chip(self, path: str):
        with open(path, "w") as f:
            json.dump(self.chip.to_disk(), f)

    def delete_chip_exposure(self, layer: int, ex: int):
        self.chip.layers[layer].exposures.pop(ex)
        print(f"Deleted exposure {layer} {ex}")
        self.on_event(Event.CHIP_CHANGED)

    @property
    def current_image(self) -> Optional[Image.Image]:
        match self.shown_image:
            case ShownImage.CLEAR:
                return None
            case ShownImage.RED_FOCUS:
                return self.red_focus.processed()
            case ShownImage.UV_FOCUS:
                return self.uv_focus.processed()
            case ShownImage.PATTERN:
                return self.pattern.processed()

    def _update_projector(self):
        img = self.current_image
        if img is None:
            self.hardware.projector.clear()
            self._sync_led_enable(0)
        else:
            self.hardware.projector.show(img)
            self._sync_led_enable(self._active_channel_mask())

    def _active_channel_mask(self) -> int:
        """Return the DLPC illumination enable bitmask for the currently shown image."""
        match self.shown_image:
            case ShownImage.RED_FOCUS:
                settings = self.red_focus.cached_settings
            case ShownImage.UV_FOCUS:
                settings = self.uv_focus.cached_settings
            case ShownImage.PATTERN:
                settings = self.pattern.cached_settings
            case _:
                return 0
        if settings is None:
            return 0b111  # settings not yet computed; leave all on
        r, g, b = settings.color_channels
        return (0b001 if r else 0) | (0b010 if g else 0) | (0b100 if b else 0)

    def _sync_led_enable(self, mask: int) -> None:
        """Sync the Image channel mask to the DLPC chip's LED enable mask to hard turn on/off LEDs to avoid exposure due to 'uv leakage'."""
        if self.dlpc is None:
            return
        try:
            self.dlpc.set_illumination_enable(mask)
        except Exception as e:
            print(f"DLPC LED sync failed: {e}")

    def _refresh_pattern(self):
        self.pattern.update(
            image=self.pattern_image,
            settings=ImageProcessSettings(
                posterization=self.posterize_strength,
                color_channels=(False, False, True),
                flatfield=None,
                size=self.hardware.projector.size(),
                image_adjust=self.image_adjust_position,
                border_size=self.border_size,
            ),
        )

        if self.red_focus_source in (
            RedFocusSource.PATTERN,
            RedFocusSource.INV_PATTERN,
        ):
            self._refresh_red_focus()

        # TODO:
        # Image adjust, resizing, and flatfield correction are performed *AFTER SLICING*

        self.on_event(Event.PATTERN_IMAGE_CHANGED)

    def set_red_focus_source(self, source: RedFocusSource):
        self.red_focus_source = source
        self._refresh_red_focus()

    def _red_focus_source(self) -> Image.Image:
        match self.red_focus_source:
            case RedFocusSource.IMAGE:
                return self.red_focus_image
            case RedFocusSource.SOLID:
                return self.solid_red_image
            case RedFocusSource.PATTERN:
                return self.pattern_image.getchannel("B").convert("RGBA")
            case RedFocusSource.INV_PATTERN:
                return ImageOps.invert(self.pattern_image.getchannel("B")).convert("RGBA")

    def _refresh_red_focus(self):
        if self.hardware.projector.size() != self.solid_red_image.size:
            self.solid_red_image = Image.new("RGB", self.hardware.projector.size(), "red")

        img = self._red_focus_source()

        self.red_focus.update(
            image=img,
            settings=ImageProcessSettings(
                posterization=self.posterize_strength,
                flatfield=None,
                color_channels=(True, False, False),
                size=self.hardware.projector.size(),
                image_adjust=self.image_adjust_position,
                border_size=self.border_size,
            ),
        )

        if self.shown_image == ShownImage.RED_FOCUS:
            self.on_event(Event.SHOWN_IMAGE_CHANGED)

    def _refresh_uv_focus(self):
        self.uv_focus.update(
            image=self.uv_focus_image,
            settings=ImageProcessSettings(
                posterization=self.posterize_strength,
                flatfield=None,
                color_channels=(False, False, True),
                size=self.hardware.projector.size(),
                image_adjust=self.image_adjust_position,
                border_size=0.0,
            ),
        )

        if self.shown_image == ShownImage.UV_FOCUS:
            self.on_event(Event.SHOWN_IMAGE_CHANGED)

    def set_posterize_strength(self, strength: Optional[int]):
        self.posterize_strength = strength
        self._refresh_red_focus()
        self._refresh_uv_focus()
        self._refresh_pattern()

    def set_border_size(self, border_size: float):
        self.border_size = border_size
        self._refresh_red_focus()
        self._refresh_uv_focus()
        self._refresh_pattern()

    def set_shown_image(self, shown_image: ShownImage):
        print(f"set_shown_image({shown_image})")
        self.shown_image = shown_image
        self.on_event(Event.SHOWN_IMAGE_CHANGED)

    def create_warning(self, msg: str):
        print(f"Warning: {msg}")
        messagebox.showwarning("Warning: ", msg)

    def move_absolute(self, coords: dict[str, float]):
        # 0  to  -($13X - $27)   in WPos space
        if self.hardware.stage.has_homing():  # debugging statements
            print(f"Moving to position: {coords}")
            print(
                f"Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}"
            )

        # find new coordinates -> some nuance exists between work and gui positioning
        # in work position, the x moves in negative direction (away from home) and y moves in positive direction (away from home)
        x = coords.get("x", 0)
        y = coords.get("y", 0)
        z = coords.get("z", 0)
        set_point = (x, y, z)

        if self.hardware.stage.has_homing():
            print(f"Moving to absolute: {set_point}")
            ok, msg = self._check_bounds(set_point)
            if not ok:
                self.create_warning(msg)
                return False

        try:
            self.hardware.stage.move_absolute(coords)
            self.stage_setpoint = set_point
            self.on_event(Event.STAGE_POSITION_CHANGED)
            return True

        except RuntimeError as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            return False

        except Exception as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            self.stage_setpoint = self.hardware.stage.get_position()
            self.on_event(Event.STAGE_POSITION_CHANGED)
            return False

    def _check_bounds(self, set_point):
        bounds = self.hardware.stage.get_bounds()

        if bounds is None:
            return True, None  # no homing, no bounds enforced

        axes = [('x', 0), ('y', 1), ('z', 2)]
        for name, i in axes:
            lo, hi = bounds[name]
            val = set_point[i]
            if not (lo <= val <= hi):
                return False, (
                    f"Moving {name.upper()} to {val} prohibited. Boundaries are [{lo}, {hi}]"
                )
        return True, None

    def move_relative(self, coords: dict[str, float]):
        if self.hardware.stage.has_homing():  # debugging statements
            print(f"Moving: {coords}")
            print(
                f"Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}"
            )
        # find new coordinates -> some nuance exists between work and gui positioning
        # in work position, the x moves in negative direction (away from home) and y moves in positive direction (away from home
        x = self.stage_setpoint[0] + coords.get("x", 0)
        y = self.stage_setpoint[1] + coords.get("y", 0)
        z = self.stage_setpoint[2] + coords.get("z", 0)
        set_point = (x, y, z)

        # if soft limits and max travel set, then enforce boundaries
        if self.hardware.stage.has_homing():
            ok, msg = self._check_bounds(set_point)
            if not ok:
                self.create_warning(msg)
                return

        try:
            self.hardware.stage.move_relative(coords)
            self.stage_setpoint = set_point
            self.on_event(Event.STAGE_POSITION_CHANGED)

        except RuntimeError as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")

        except Exception as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            self.stage_setpoint = self.hardware.stage.get_position()
            self.on_event(Event.STAGE_POSITION_CHANGED)

    def set_use_solid_red(self, use: bool):
        self.use_solid_red = use
        self.set_shown_image(ShownImage.RED_FOCUS)
        self._refresh_red_focus()

    def set_pattern_image(self, img: Image.Image, path: str):
        self.pattern_image = img
        self.pattern_image_path = path
        self._refresh_pattern()

    def set_red_focus_image(self, img: Image.Image):
        self.red_focus_image = img
        self._refresh_red_focus()

    def set_uv_focus_image(self, img: Image.Image):
        self.uv_focus_image = img
        self._refresh_uv_focus()

    def set_patterning_busy(self, busy: bool):
        self.patterning_busy = busy
        self.on_event(Event.MOVEMENT_LOCK_CHANGED)
        self.on_event(Event.PATTERNING_BUSY_CHANGED)

    def set_progress(self, pattern_progress: float, exposure_progress: float):
        self.patterning_progress = pattern_progress
        self.exposure_progress = exposure_progress
        self.on_event(Event.EXPOSURE_PATTERN_PROGRESS_CHANGED)

    def set_latest_image(self, camera_image, seq: Optional[int] = None):
        self.camera_image = camera_image
        if seq is not None:
            # number of the camera frame this is (counts up once per frame the camera delivers), so code that
            # needs a frame taken AFTER some change can tell a new frame from the one that was already here
            self.camera_image_seq = seq

    def set_autofocus_busy(self, busy):
        self.autofocus_busy = busy
        self.on_event(Event.MOVEMENT_LOCK_CHANGED)

    def abort_patterning(self):
        self.should_abort = True
        print("Aborting patterning")

    def in_uv(self):
        return self.shown_image in (ShownImage.PATTERN, ShownImage.UV_FOCUS)

    def home_stage(self):
        """
        Homing stage resets Machine position (Mpos) and sets Work Position (WPos)
        of current state post-homing to (0, 0, 0), which means set_point must
        be updated to reflect the work position
        """
        self.hardware.stage.home()
        self.hardware.stage.set_on_start_location()
        print(f"Post Homing Location: {self.hardware.stage.get_on_start_location()}")
        print("Homing Complete.")

        self.on_event(Event.STAGE_POSITION_CHANGED)

    def query_config(self):
        self.hardware.stage.get_position()
        print("Query Config Complete.")

    def set_image_position(self, x, y, t):
        self.image_adjust_position = (x, y, t)
        self._refresh_red_focus()
        self._refresh_uv_focus()
        self._refresh_pattern()
        self.on_event(Event.IMAGE_ADJUST_CHANGED)

    @property
    def image_position(self):
        return self.image_adjust_position

    @property
    def movement_lock(self):
        if self.patterning_busy or self.autofocus_busy:
            return MovementLock.LOCKED
        # elif (self.shown_image == ShownImage.UV_FOCUS or self.shown_image == ShownImage.PATTERN):
        #     return MovementLock.XY_LOCKED
        else:
            return MovementLock.UNLOCKED

    def on_event(self, event: Event, *args, **kwargs):
        if event not in self.listeners:
            return

        for listener in self.listeners[event]:
            listener(*args, **kwargs)

    def on_event_cb(self, event: Event, *args, **kwargs):
        return lambda: self.on_event(event, *args, **kwargs)

    def add_event_listener(self, event: Event, listener: Callable):
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(listener)

    def begin_patterning(self):
        # TODO: Update patterning preview

        print("Patterning at ", self.stage_setpoint)
        duration = self.exposure_time
        print(f"Patterning 1 tiles for {duration}ms\nTotal time: {str(round((duration) / 1000))}s")

        # TODO: Image slicing.
        # Note that flatfield correction and image adjustment should be applied *after* slicing
        self.set_patterning_busy(True)
        self.set_shown_image(ShownImage.PATTERN)
        end_time = time.time() + duration / 1000.0
        while time.time() < end_time:
            progress = 1.0 - ((end_time - time.time()) * 1000 / duration)
            self.set_progress(0.0, progress)
            self.root.update()
            if self.should_abort:
                break
        self.set_shown_image(ShownImage.CLEAR)
        self.root.update()  # Force image to stop being displayed ASAP
        self.set_progress(1.0, 1.0)

        log = ExposureLog(
            datetime.now(),
            self.pattern_image_path,
            self.stage_setpoint,
            duration,
            self.should_abort,
        )
        self.exposure_history.append(log)
        self.chip.layers[-1].exposures.append(log)

        self.on_event(Event.CHIP_CHANGED)
        self.set_patterning_busy(False)

        if self.should_abort:
            print("Patterning aborted")
            self.should_abort = False

    def expose_pattern(self, red_to_uv_offset: float):
        """Expose the loaded pattern at the current position (used by both tiling flows): move z by the
        red -> UV focus offset, UV autofocus, expose, then back to red mode and the red focus height."""
        self.move_relative({"z": red_to_uv_offset})
        self.non_blocking_delay(0.5)
        self.enter_uv_mode(mode_switch_autofocus=False)
        self.autofocus(blue_only=True)
        self.begin_patterning()
        self.enter_red_mode(mode_switch_autofocus=False)
        self.move_relative({"z": -1 * red_to_uv_offset})

    def non_blocking_delay(self, t: float):
        start = time.time()
        while time.time() - start < t:
            self.root.update()

    def enter_red_mode(self, mode_switch_autofocus=True):
        print("enter_red_mode")
        self.set_shown_image(ShownImage.RED_FOCUS)
        if self.camera:
            self.camera.setExposureTime(self.red_exposure_time)
        if mode_switch_autofocus and self.autofocus_on_mode_switch:
            self.autofocus(blue_only=False)
        self.on_event(Event.MOVEMENT_LOCK_CHANGED)

    def enter_uv_mode(self, mode_switch_autofocus=True):
        if self.auto_snapshot_on_uv:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = self.snapshot_directory / f"uv_mode_entry_{timestamp}.png"
            self.on_event(Event.SNAPSHOT, str(filename))

        if self.camera:
            self.camera.setExposureTime(self.uv_exposure_time)
        if mode_switch_autofocus and not self.autofocus_busy and self.autofocus_on_mode_switch:
            # UV mode usually needs about -70 to be in focus compared to red mode
            # self.move_relative({"z": -85.0})
            pass

        # self.set_shown_image(ShownImage.UV_FOCUS)
        self.set_shown_image(ShownImage.CLEAR)  # enter uv mode: don't project uv

        if mode_switch_autofocus and self.autofocus_on_mode_switch:
            self.non_blocking_delay(2.0)
            self.autofocus(blue_only=True)

        self.on_event(Event.MOVEMENT_LOCK_CHANGED)

    def autofocus(self, blue_only, log=False):
        if not self.camera:
            print("No camera connected, skipping autofocus")
            return

        if self.first_autofocus:
            # TODO: Fix this spuriously triggering
            self.first_autofocus = False
            return

        if self.autofocus_busy:
            print("Skipping nested autofocus!")
            return

        if log:
            try:
                os.mkdir('aftest')
            except FileExistsError:
                pass
            log_file = open('aftest/log.csv', 'w')

        print("Starting autofocus")

        if self.hardware.stage.has_homing():
            counter = 0

            def sample():
                def one_sample():
                    return fetch_focus_score(self.camera_image, blue_only=blue_only, log=True)

                focus_score = sum([one_sample() for _ in range(5)]) / 5
                print("focus average:", focus_score)
                nonlocal counter
                if log:
                    log_file.write(f'{counter},{focus_score}\n')
                    cv2.imwrite(f'aftest/img{counter}.png', self.camera_image)
                counter += 1
                return focus_score

            print("Starting Autofocus...")
            best_score = -1.0
            best_z = 0
            z_base = self.hardware.stage.get_autofocus()

            # account for uv mode, where z-focus is different
            if blue_only:
                z_base += 50.0
                if not (self.move_absolute({"z": z_base})):
                    self.create_warning("Failed autofocus, z-stage can't go past boundary limits")
                    self.set_autofocus_busy(False)
                    return
                self.non_blocking_delay(1.0)

            else:
                for i in range(-20, 20, 2):
                    if not (self.move_absolute({"z": z_base + i})):
                        self.create_warning(
                            "Failed autofocus, z-stage can't go past boundary limits"
                        )
                        self.set_autofocus_busy(False)
                        return
                    self.non_blocking_delay(0.5)
                    new_score = sample()
                    # always check for optimal scores
                    if new_score > best_score:
                        best_score = new_score
                        best_z = self.stage_setpoint[2]

                print(f"Fine grain sampling done, best focus is: {best_score}")
                self.move_absolute({"z": best_z})
                self.non_blocking_delay(1.0)

        else:
            counter = 0

            def sample_focus():
                def do_thing():
                    self.non_blocking_delay(0.1)
                    return compute_focus_score(self.camera_image, blue_only=blue_only)

                focus_score = sorted([do_thing() for _ in range(3)])[1]
                nonlocal counter
                if log:
                    log_file.write(f'{counter},{focus_score}\n')
                    cv2.imwrite(f'aftest/img{counter}.png', self.camera_image)
                counter += 1
                return focus_score

            self.set_autofocus_busy(True)
            self.non_blocking_delay(1.0)
            mid_score = sample_focus()
            self.move_relative({"z": -20.0})
            self.non_blocking_delay(1.0)
            neg_score = sample_focus()
            self.move_relative({"z": 40.0})
            self.non_blocking_delay(1.0)
            pos_score = sample_focus()
            self.move_relative({"z": -20.0})
            self.non_blocking_delay(1.0)

            last_focus = mid_score

            if neg_score < mid_score < pos_score:
                # Improved focus is in the +Z direction
                for i in range(30):
                    self.move_relative({"z": 10.0})
                    self.non_blocking_delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z coarse autofocus {i}")
                        last_focus = new_score
                        break
                    last_focus = new_score

                for i in range(10):
                    self.move_relative({"z": -2.0})
                    self.non_blocking_delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful -Z fine autofocus {i}")
                        break
                    last_focus = new_score
            elif neg_score > mid_score > pos_score:
                # Improved focus is in the -Z direction
                for i in range(30):
                    self.move_relative({"z": -10.0})
                    self.non_blocking_delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful -Z coarse autofocus {i}")
                        break
                    last_focus = new_score

                for i in range(10):
                    self.move_relative({"z": 2.0})
                    self.non_blocking_delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z fine autofocus {i}")
                        break
                    last_focus = new_score
            elif neg_score < mid_score and pos_score < mid_score:
                # We are very close to already being in focus
                print(f"Almost in focus! (neg {neg_score} mid {mid_score} pos {pos_score})")
                self.move_relative({"z": -20.0})
                self.non_blocking_delay(0.5)

                for i in range(30):
                    self.move_relative({"z": 2.0})
                    self.non_blocking_delay(0.5)
                    new_score = sample_focus()
                    if last_focus > new_score:
                        print(f"Successful +Z fine autofocus {i}")
                        break
                    last_focus = new_score
            else:
                print("Autofocus is confused!")

        print("Autofocus Complete.")
        self.set_autofocus_busy(False)
        print("Finished autofocus")

    def initialize_alignment(self, config: LithographerConfig):
        self.config = config
        self.realtime_detection = config.alignment.enabled
        # Attempt loading the model even if detection is off by default
        try:
            print("loading model")
            model_path = config.alignment.model_path
            self.model = YOLO(model_path)
            print("loaded model")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")

    def set_snapshot_directory(self, directory: Path):
        self.snapshot_directory = directory
        self.snapshot_directory.mkdir(exist_ok=True)


class SnapshotFrame:
    """
    Presents a frame with a filename entry and a button to save screenshots of the current camera view.
    """

    def __init__(self, parent, enable, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=1, column=0)

        state = "normal" if enable else "disable"

        # TODO: Allow %X, %Y, %Z formats to save position on chip
        self.name_var = StringVar(value="output_%T.png")
        self.name_var.trace_add("write", lambda _a, _b, _c: self._refresh_name_preview())

        self.counter = 0

        self.name_entry = ttk.Entry(self.frame, textvariable=self.name_var, state=state)
        self.name_entry.grid(row=0, column=0)

        self.name_preview = ttk.Label(self.frame)
        self.name_preview.grid(row=0, column=1)

        def on_snapshot_button():
            event_dispatcher.on_event(Event.SNAPSHOT, self._next_filename())
            self.counter += 1
            self._refresh_name_preview()

        self.button = ttk.Button(
            self.frame, text="Take Snapshot", command=on_snapshot_button, state=state
        )
        self.button.grid(row=0, column=2)

        self._refresh_name_preview()

    def _next_filename(self):
        counter_str = str(self.counter)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        name = self.name_var.get()
        name = name.replace("%C", counter_str).replace("%c", counter_str)
        name = name.replace("%T", time_str).replace("%t", time_str)
        return name

    def _refresh_name_preview(self):
        self.name_preview.configure(text=f"Output File: {self._next_filename()}")


class CameraFrame:
    def __init__(
        self,
        parent,
        event_dispatcher: EventDispatcher,
        c: CameraModule,
        camera_scale: float,
    ):
        self.frame = ttk.Frame(parent)
        self.label = ttk.Label(self.frame, text="No Camera Connected")
        self.label.grid(row=0, column=0, sticky="nesw")
        self._config_camera_scale = camera_scale
        self.gui_camera_scale = camera_scale

        self.focus_score_label = ttk.Label(self.frame, text="Focus Score: N/A")
        self.focus_score_label.grid(row=1, column=0)

        self.snapshot = SnapshotFrame(self.frame, c is not None, event_dispatcher)
        self.snapshot.frame.grid(row=2, column=0)

        self.event_dispatcher = event_dispatcher

        self.snapshots_pending = queue.Queue()
        self.event_dispatcher.add_event_listener(
            Event.SNAPSHOT, lambda filename: self.snapshots_pending.put(filename)
        )

        self.gui_img = None
        self.camera = c
        self.pending_frame = None

    def _on_new_frame(self):
        # FIXME: is this really the only way tkinter exposes to do this??
        # We want to send frames from the callback over to the main thread,
        # but in way where it just grabs the most recently-made-available frame.
        # If you send an event, events will just pile up in the queue if we ever fall behind.
        # This might have the same problem!
        # I have no idea how to fix this
        # self.event_dispatcher.root.update_idletasks()
        # self.event_dispatcher.root.after_idle(lambda: self._on_new_frame())
        try:
            if self.pending_frame is None:
                return
            image, dimensions, format, seq = self.pending_frame
            red_score = fetch_focus_score(image, blue_only=False)
            blue_score = fetch_focus_score(image, blue_only=True)
            self.focus_score_label.configure(text=f"Focus Score: {red_score:.3e} {blue_score:.3e}")

            try:
                filename = self.snapshots_pending.get_nowait()
                print(f"Saving image {filename}")
                fetch_focus_score(image, blue_only=False)
                fetch_focus_score(image, blue_only=False)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(filename, img)
            except queue.Empty:
                pass

            self.gui_camera_preview(image, dimensions, seq)
        finally:
            self.event_dispatcher.root.after(66, lambda: self._on_new_frame())

    def resize(self, ui_scale: float):
        self.gui_camera_scale = self._config_camera_scale * ui_scale

    def start(self):
        if not self.camera:
            print("No camera available")
            return

        # self.event_dispatcher.root.bind('<<NewFrame>>', lambda x: self._on_new_frame(x))

        frame_seq = 0

        def cameraCallback(image, dimensions, format):
            # runs on the camera's thread, once per delivered frame; the number goes with the frame
            nonlocal frame_seq
            frame_seq += 1
            self.pending_frame = (image, dimensions, format, frame_seq)
            # self.event_dispatcher.root.event_generate('<<NewFrame>>', when='tail')

        if not self.camera.open():
            print("Camera failed to start")
        else:
            self.camera.setSetting("image_format", "rgb888")
            self.camera.setStreamCaptureCallback(cameraCallback)
            if not self.camera.startStreamCapture():
                print("Failed to start stream capture for camera")

        self._on_new_frame()

    def cleanup(self):
        if self.camera is not None:
            self.camera.close()

    def gui_camera_preview(self, camera_image, dimensions, seq: Optional[int] = None):
        model = self.event_dispatcher.model
        if model and self.event_dispatcher.realtime_detection:
            _, camera_image = detect_alignment_markers(model, camera_image, draw_rectangle=True)
        self.event_dispatcher.set_latest_image(camera_image, seq)
        resized_img = cv2.resize(
            camera_image, (0, 0), fx=self.gui_camera_scale, fy=self.gui_camera_scale
        )
        gui_img = image_to_tk_image(Image.fromarray(resized_img, mode="RGB"))
        self.label.configure(image=gui_img)
        self.gui_img = gui_img


class StagePositionFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher, uvmode, ui_scale: float = 1.0):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        # Position display at top
        self.position_frame = ttk.LabelFrame(self.frame, text="Current Position (µm)")
        self.position_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

        self.position_intputs = []
        # Track all interactive widgets for locking
        self.xy_widgets: list[tk.Entry | tk.Canvas] = []
        self.z_widgets = []

        self.zlock = False

        # Absolute

        self.absolute_frame = ttk.LabelFrame(self.frame, text="Stage Position")
        self.absolute_frame.grid(row=0, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "z")):
            self.position_intputs.append(IntEntry(parent=self.absolute_frame, default=0))
            self.position_intputs[-1].widget.grid(row=0, column=i)

            if i in (0, 1):
                self.xy_widgets.append(self.position_intputs[i].widget)
            else:
                self.z_widgets.append(self.position_intputs[i].widget)

        def callback_set():  # command for the set position button
            x, y, z = self._position()
            event_dispatcher.move_absolute({"x": x, "y": y, "z": z})

        self.set_position_button = ttk.Button(
            self.absolute_frame, text="Set Stage Position", command=callback_set
        )
        self.set_position_button.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.all_widgets = self.xy_widgets + self.z_widgets + [self.set_position_button]

        # Relative

        control_frame = ttk.Frame(self.frame)
        control_frame.grid(row=1, column=0, columnspan=2)

        if not uvmode:
            # Custom XYZ control
            custom_control_frame = ttk.LabelFrame(control_frame, text="Custom Step")
            custom_control_frame.grid(row=0, column=0, padx=10)
            self.create_custom_xyz_control(custom_control_frame)

            # XY circular control: hide in UV mode
            xy_frame = ttk.Frame(control_frame)
            xy_frame.grid(row=0, column=1, padx=10)
            self.create_xy_control(xy_frame, ui_scale)
            z_col = 2
        else:
            z_col = 0

        # Z vertical control
        z_frame = ttk.Frame(control_frame)
        z_frame.grid(row=0, column=z_col, padx=10)
        self.create_z_control(z_frame, ui_scale)

        def on_lock_change():
            lock = event_dispatcher.movement_lock
            match lock:
                case MovementLock.UNLOCKED:
                    for w in self.all_widgets:
                        if hasattr(w, 'configure'):
                            w.configure(state="normal")
                    self.zlock = False
                    for rect_id, orig_color in self.z_rectangles:
                        self.z_canvas.itemconfig(rect_id, fill=orig_color)
                case MovementLock.LOCKED:
                    for w in self.all_widgets:
                        if hasattr(w, 'configure'):
                            w.configure(state="disabled")
                    self.zlock = True
                    grey_color = '#b9bbb6'
                    for rect_id, _ in self.z_rectangles:
                        self.z_canvas.itemconfig(rect_id, fill=grey_color)

        event_dispatcher.add_event_listener(Event.MOVEMENT_LOCK_CHANGED, on_lock_change)

        def on_position_change():
            pos = event_dispatcher.stage_setpoint
            for i in range(3):
                self.position_intputs[i].set(int(pos[i]))

        event_dispatcher.add_event_listener(Event.STAGE_POSITION_CHANGED, on_position_change)

        # Shortcuts at bottom
        self.shortcut_frame = ttk.LabelFrame(self.frame, text="Shortcuts")
        self.shortcut_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        def on_chip_origin():
            origin = event_dispatcher.hardware.stage.get_on_start_location()
            event_dispatcher.move_absolute({"x": origin[0], "y": origin[1], "z": origin[2]})

        def on_chip_unload():
            """
            As of now, home position is actually the most optimal place to be removed the chip
            """
            on_chip_origin()

        btn_state = "normal" if event_dispatcher.hardware.stage.has_homing() else "disabled"
        self.chip_origin_button = ttk.Button(
            self.shortcut_frame,
            text="Chip origin",
            command=on_chip_origin,
            state=btn_state,
        )
        self.chip_origin_button.grid(row=0, column=0, padx=5)

        self.chip_unload_button = ttk.Button(
            self.shortcut_frame,
            text="Load/unload",
            command=on_chip_unload,
            state=btn_state,
        )
        self.chip_unload_button.grid(row=0, column=1, padx=5)

    def _position(self) -> tuple[int, int, int]:
        p = self.position_intputs
        return (int(p[0].get()), int(p[1].get()), int(p[2].get()))

    def create_custom_xyz_control(self, parent):
        """Create custom step size control"""

        class Direction(Enum):
            POS = 0
            NEG = 1

        class Axis(Enum):
            X = 0
            Y = 1
            Z = 2

        def custom_step_move(direction, axis, step_size=10):
            assert step_size > 0, "step size should be greater than zero"
            movement = step_size if direction == Direction.POS else -step_size
            if axis == Axis.X:
                self.event_dispatcher.move_relative({"x": movement})
            elif axis == Axis.Y:
                self.event_dispatcher.move_relative({"y": movement})
            else:
                self.event_dispatcher.move_relative({"z": movement})

        self.custom_step_size_label = ttk.Label(parent, text="step size")
        self.custom_step_size = IntEntry(parent)

        self.custom_step_pos_x_button = ttk.Button(
            parent,
            text="+X",
            command=lambda: custom_step_move(Direction.POS, Axis.X, self.custom_step_size.get()),
        )
        self.custom_step_pos_y_button = ttk.Button(
            parent,
            text="+Y",
            command=lambda: custom_step_move(Direction.POS, Axis.Y, self.custom_step_size.get()),
        )
        self.custom_step_pos_z_button = ttk.Button(
            parent,
            text="+Z",
            command=lambda: custom_step_move(Direction.POS, Axis.Z, self.custom_step_size.get()),
        )

        self.custom_step_neg_x_button = ttk.Button(
            parent,
            text="-X",
            command=lambda: custom_step_move(Direction.NEG, Axis.X, self.custom_step_size.get()),
        )
        self.custom_step_neg_y_button = ttk.Button(
            parent,
            text="-Y",
            command=lambda: custom_step_move(Direction.NEG, Axis.Y, self.custom_step_size.get()),
        )
        self.custom_step_neg_z_button = ttk.Button(
            parent,
            text="-Z",
            command=lambda: custom_step_move(Direction.NEG, Axis.Z, self.custom_step_size.get()),
        )

        # formatting
        self.custom_step_size_label.grid(row=0, column=0)
        self.custom_step_size.widget.grid(row=0, column=1, sticky="ew")
        self.custom_step_size.widget.config(width=10)
        self.custom_step_pos_x_button.grid(row=1, column=0, sticky="ew")
        self.custom_step_neg_x_button.grid(row=1, column=1, sticky="ew")
        self.custom_step_pos_y_button.grid(row=2, column=0, sticky="ew")
        self.custom_step_neg_y_button.grid(row=2, column=1, sticky="ew")
        self.custom_step_pos_z_button.grid(row=3, column=0, sticky="ew")
        self.custom_step_neg_z_button.grid(row=3, column=1, sticky="ew")

    def create_xy_control(self, parent, ui_scale: float = 1.0):
        """Create circular XY control canvas, then draw it."""
        self.xy_canvas = tkinter.Canvas(
            parent, width=1, height=1, bg='#f0f0f0', highlightthickness=0
        )
        self.xy_canvas.pack()
        self.xy_canvas.bind('<Button-1>', self._on_xy_click)
        self.xy_widgets.append(self.xy_canvas)
        self._draw_xy(ui_scale)

    def _draw_xy(self, ui_scale: float):
        """Clear and redraw the XY joystick canvas at the given scale."""
        canvas_size = int(255 * ui_scale)
        center = canvas_size // 2

        self._xy_canvas_size = canvas_size
        self._xy_radii = [int(r * ui_scale) for r in [31, 62, 94, 125]]

        self.xy_canvas.config(width=canvas_size, height=canvas_size)
        self.xy_canvas.delete("all")

        step_sizes = [10, 50, 100, 250]
        colors = ['#52796f', '#5a9a8b', '#7ab8a8', '#95c9be']

        for i, radius in enumerate(self._xy_radii[::-1]):
            self.xy_canvas.create_oval(
                center - radius,
                center - radius,
                center + radius,
                center + radius,
                fill=colors[i],
                outline='#2c3e50',
                width=2,
            )

        cx, cy = center, center
        half_L = canvas_size / 2
        for angle_deg in [-45, 45]:
            angle_rad = math.radians(angle_deg)
            x0 = cx - half_L * math.sin(angle_rad)
            y0 = cy - half_L * math.cos(angle_rad)
            x1 = cx + half_L * math.sin(angle_rad)
            y1 = cy + half_L * math.cos(angle_rad)
            self.xy_canvas.create_line(x0, y0, x1, y1, fill='#2c3e50', width=3)

        font_large = ('Arial', max(7, int(12 * ui_scale)), 'bold')
        font_small = ('Arial', max(6, int(9 * ui_scale)), 'bold')
        labels = [
            (center, 10, "+Y", '#85c1e9'),
            (center, canvas_size - 10, "-Y", '#85c1e9'),
            (canvas_size - 10, center, "+X", '#f4d03f'),
            (10, center, "-X", '#f4d03f'),
        ]
        for x, y, text, color in labels:
            self.xy_canvas.create_text(x, y, text=text, fill=color, font=font_large)

        for i, (radius, step) in enumerate(zip(self._xy_radii, step_sizes)):
            y_pos = center - radius + max(8, int(20 * ui_scale))
            self.xy_canvas.create_text(center, y_pos, text=str(step), fill='white', font=font_small)

    def _on_xy_click(self, event):
        """Handle clicks on the XY canvas"""
        canvas_size = self._xy_canvas_size
        center = canvas_size // 2

        # Calculate distance from center and angle
        dx = event.x - center
        dy = event.y - center
        distance = math.sqrt(dx**2 + dy**2)

        # Determine which layer (step size)
        radii = self._xy_radii
        step_sizes = [10, 50, 100, 250]

        step_size = None
        for i, radius in enumerate(radii):
            if distance <= radius:
                step_size = step_sizes[i]
                break

        if step_size is None:
            return  # Click outside all layers

        # Determine direction based on quadrant
        # Positive Y is up (negative dy), Negative Y is down (positive dy)
        # Positive X is right (positive dx), Negative X is left (negative dx)

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Determine which direction is dominant
        if abs_dx > abs_dy:
            if dx > 0:
                self.event_dispatcher.move_relative({"x": step_size})
            else:
                self.event_dispatcher.move_relative({"x": -step_size})
        else:
            if dy < 0:
                self.event_dispatcher.move_relative({"y": step_size})
            else:
                self.event_dispatcher.move_relative({"y": -step_size})

    def create_z_control(self, parent, ui_scale: float = 1.0):
        """Create Z slider canvas, then draw it."""
        self.z_canvas = tkinter.Canvas(
            parent, width=1, height=1, bg='#34495e', highlightthickness=0
        )
        self.z_canvas.pack()
        self._z_label = ttk.Label(parent, text="Z Control")
        self._z_label.pack(pady=(0, 5))
        self._draw_z(ui_scale)

    def _draw_z(self, ui_scale: float):
        """Clear and redraw the Z slider canvas at the given scale."""
        bar_width = int(60 * ui_scale)
        section_height = int(30 * ui_scale)
        total_height = section_height * 8

        self.z_canvas.config(width=bar_width, height=total_height)
        self.z_canvas.delete("all")

        step_sizes = [10, 50, 100, 250]
        colors_plus = ['#a8dadc', '#87bdbf', '#66a0a3', '#458486']
        colors_minus = ['#458486', '#66a0a3', '#87bdbf', '#a8dadc']
        font_z = ('Arial', max(7, int(10 * ui_scale)), 'bold')
        grey = '#b9bbb6'

        self.z_rectangles = []

        for i in range(8):
            y_start = i * section_height
            y_end = (i + 1) * section_height

            if i < 4:
                color = colors_plus[i]
                step = step_sizes[3 - i]
                label = f"+Z\n{step}"
                direction = "+"
            else:
                color = colors_minus[i - 4]
                step = step_sizes[i - 4]
                label = f"-Z\n{step}"
                direction = "-"

            fill = grey if self.zlock else color
            rect_id = self.z_canvas.create_rectangle(
                0, y_start, bar_width, y_end, fill=fill, outline='#2c3e50', width=2
            )
            self.z_rectangles.append((rect_id, color))

            text_id = self.z_canvas.create_text(
                bar_width // 2,
                (y_start + y_end) // 2,
                text=label,
                fill='#1a1a1a',
                font=font_z,
            )

            for item_id in [rect_id, text_id]:
                self.z_canvas.tag_bind(
                    item_id,
                    '<Button-1>',
                    lambda e, d=direction, s=step: self._on_z_click(d, s),
                )

        self._z_label.configure(font=('Arial', max(7, int(10 * ui_scale)), 'bold'))

    def _on_z_click(self, direction, step_size):
        """Handle clicks on Z control sections"""
        if not self.zlock:
            if direction == "+":
                self.event_dispatcher.move_relative({"z": step_size})
            else:
                self.event_dispatcher.move_relative({"z": -step_size})

    def resize(self, ui_scale: float):
        """Redraw canvases proportionally to the new window scale."""
        if hasattr(self, 'xy_canvas'):
            self._draw_xy(ui_scale)
        self._draw_z(ui_scale)


class ImageAdjustFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)

        self.position_intputs = []
        self.step_size_intputs = []

        self.lockable_widgets = []

        # Absolute

        self.absolute_frame = ttk.LabelFrame(self.frame, text="Image Adjustment")
        self.absolute_frame.grid(row=0, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "ϴ")):
            if i == 0 or i == 1:
                self.position_intputs.append(IntEntry(parent=self.absolute_frame, default=0))
            else:
                self.position_intputs.append(FloatEntry(parent=self.absolute_frame, default=0))
            self.position_intputs[-1].widget.grid(row=0, column=i)

        def callback_set():
            x, y, t = self._position()
            event_dispatcher.set_image_position(x, y, t)

        self.set_position_button = ttk.Button(
            self.absolute_frame, text="Set Image Position", command=callback_set
        )
        self.set_position_button.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Relative
        self.relative_frame = ttk.LabelFrame(self.frame, text="Adjustment")
        self.relative_frame.grid(row=1, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "ϴ")):
            if i == 0 or i == 1:
                self.step_size_intputs.append(
                    IntEntry(
                        parent=self.relative_frame,
                        default=10,
                        min_value=-1000,
                        max_value=1000,
                    )
                )
            else:
                self.step_size_intputs.append(
                    FloatEntry(
                        parent=self.relative_frame,
                        default=10,
                        min_value=0.0,
                        max_value=360.0,
                    )
                )
            self.step_size_intputs[-1].widget.grid(row=3, column=i)

            def callback_pos(index, c):
                pos = list(event_dispatcher.image_position)
                pos[index] += self.step_sizes()[index]
                event_dispatcher.set_image_position(*pos)

            def callback_neg(index, c):
                pos = list(event_dispatcher.image_position)
                pos[index] -= self.step_sizes()[index]
                event_dispatcher.set_image_position(*pos)

            coord_inc_button = ttk.Button(
                self.relative_frame,
                text=f"+{coord.upper()}",
                command=partial(callback_pos, i, coord),
            )
            coord_dec_button = ttk.Button(
                self.relative_frame,
                text=f"-{coord.upper()}",
                command=partial(callback_neg, i, coord),
            )

            coord_inc_button.grid(row=0, column=i)
            coord_dec_button.grid(row=1, column=i)

            self.lockable_widgets.append(coord_inc_button)
            self.lockable_widgets.append(coord_dec_button)
            self.lockable_widgets.append(self.position_intputs[i].widget)
            self.lockable_widgets.append(self.step_size_intputs[i].widget)
        self.lockable_widgets.append(self.set_position_button)

        ttk.Label(
            self.relative_frame,
            text="Step Size (pixels, pixels, degrees)",
            anchor="center",
        ).grid(row=2, column=0, columnspan=3, sticky="ew")

        def on_position_change():
            pos = event_dispatcher.image_adjust_position
            for i in range(3):
                self.position_intputs[i].set(int(pos[i]))

        event_dispatcher.add_event_listener(Event.IMAGE_ADJUST_CHANGED, on_position_change)

        def on_lock_change():
            if event_dispatcher.movement_lock == MovementLock.UNLOCKED:
                for w in self.lockable_widgets:
                    w.configure(state="normal")
            else:
                for w in self.lockable_widgets:
                    w.configure(state="disabled")

        event_dispatcher.add_event_listener(Event.MOVEMENT_LOCK_CHANGED, on_lock_change)

    def _position(self) -> tuple[int, int, int]:
        p = self.position_intputs
        return (int(p[0].get()), int(p[1].get()), int(p[2].get()))

    def _set_position(self, pos: tuple[int, int, int]):
        for i in range(3):
            self.position_intputs[i].set(int(pos[i]))

    def step_sizes(self) -> tuple[int, int, int]:
        s = self.step_size_intputs
        return (int(s[0].get()), int(s[1].get()), int(s[2].get()))


class PredefinedImageSelector:
    """A widget that shows a selection of predefined images instead of file dialog"""

    def __init__(self, parent, size, predefined_images, on_select=None):
        self.parent = parent
        self.size = size
        self.predefined_images = predefined_images  # List of (name, path) tuples
        self.on_select = on_select
        self.current_image = None
        self.current_path = ""

        # Create main frame
        self.widget = ttk.Frame(parent)

        # Create thumbnail display
        placeholder = Image.new("RGB", size, "gray")
        self.photo = image_to_tk_image(placeholder)
        self.label = ttk.Label(self.widget, image=self.photo, relief="solid", borderwidth=2)
        self.label.grid(row=0, column=0, columnspan=2, pady=5)

        # Create dropdown for image selection
        self.image_var = StringVar()
        self.image_dropdown = ttk.Combobox(
            self.widget,
            textvariable=self.image_var,
            values=[name for name, _ in predefined_images],
            state="readonly",
        )
        self.image_dropdown.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.image_dropdown.bind("<<ComboboxSelected>>", self._on_selection_change)

        # Add a button to Load custom alignment marks
        self.upload_button = ttk.Button(
            self.widget, text="Upload Marks", command=self._upload_marks
        )
        self.upload_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)

        # Add a button to load the selected image
        self.load_button = ttk.Button(
            self.widget, text="Load Selected", command=self._load_selected
        )
        self.load_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

        # Set default selection if images are available
        if predefined_images:
            self.image_dropdown.set(predefined_images[0][0])
            self._load_image(predefined_images[0][1])

    def _on_selection_change(self, event=None):
        """Called when dropdown selection changes"""
        selected_name = self.image_var.get()
        for name, path in self.predefined_images:
            if name == selected_name:
                self._load_image(path)
                break

    def _upload_marks(self):
        """Called when Upload Marks button is clicked"""
        current_directory = StringVar(value=str("~"))
        # checks need to be made before we allow user to do this
        dir_path = filedialog.askopenfilename(
            initialdir=current_directory.get(),
            filetypes=[("All Files", "*.*")],
            title="Select Custom Alignment Marks",
        )
        filename = f"Plus Mark {len(self.predefined_images)}"
        print(f'uploading custom alignment marker file: {filename}')

        if dir_path:  # User didn't cancel
            self.predefined_images.append((filename, dir_path))
            self.image_dropdown['values'] = [name for name, _ in self.predefined_images]

    def _load_selected(self):
        """Called when Load Selected button is clicked"""
        if self.on_select and self.current_image:
            self.on_select(None)  # Call the callback

    def _load_image(self, path):
        """Load and display an image from the given path"""
        try:
            img = Image.open(path)
            self.current_image = img
            self.current_path = path

            # Create thumbnail for display
            thumb = img.copy()
            thumb.thumbnail(self.size, Image.Resampling.LANCZOS)
            self.photo = image_to_tk_image(thumb)
            self.label.configure(image=self.photo)
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            # Show placeholder on error
            placeholder = Image.new("RGB", self.size, "red")
            self.photo = image_to_tk_image(placeholder)
            self.label.configure(image=self.photo)

    @property
    def image(self):
        """Return the current image"""
        return self.current_image

    @property
    def path(self):
        """Return the current image path"""
        return self.current_path


class ImageSelectFrame:
    def __init__(self, parent, button_text, import_command, predefined_images=None):
        self.frame = ttk.Frame(parent)

        if predefined_images:
            # Use predefined image selector
            self.selector = PredefinedImageSelector(
                self.frame, THUMBNAIL_SIZE, predefined_images, on_select=import_command
            )
            self.selector.widget.grid(row=0, column=0)

            # For compatibility with existing code
            self.thumb = self.selector
        else:
            # Use original thumbnail selector (file dialog)
            self.thumb = Thumbnail(self.frame, THUMBNAIL_SIZE, on_import=import_command)
            self.thumb.widget.grid(row=0, column=0)

        self.label = ttk.Label(self.frame, text=button_text)
        self.label.grid(row=1, column=0)


class PatternDisplayFrame:  # read only pattern display in red and uv focusing mode
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher
        # instead of pattern_frame = ImageSelectFrame
        # use pattern_display_frame (read-only)
        self.pattern_display_frame = ttk.LabelFrame(self.frame, text="Current Pattern")
        self.pattern_display_frame.grid(row=0, column=0, padx=5, pady=5)

        placeholder = Image.new("RGB", THUMBNAIL_SIZE, "gray")
        self.pattern_photo = image_to_tk_image(placeholder)
        self.pattern_label = ttk.Label(self.pattern_display_frame, image=self.pattern_photo)
        self.pattern_label.grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(
            self.pattern_display_frame,
            text="(Upload in Pattern Upload tab)",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).grid(row=1, column=0)

        event_dispatcher.add_event_listener(
            Event.PATTERN_IMAGE_CHANGED, self._update_pattern_display
        )
        event_dispatcher.add_event_listener(Event.SHOWN_IMAGE_CHANGED, self._on_shown_image_changed)

    def _update_pattern_display(self):
        """Update the read-only pattern display when pattern changes"""
        if hasattr(self.event_dispatcher, 'pattern_image') and self.event_dispatcher.pattern_image:
            # Create thumbnail for display
            thumb = self.event_dispatcher.pattern_image.copy()
            thumb.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            self.pattern_photo = image_to_tk_image(thumb)
            self.pattern_label.configure(image=self.pattern_photo)

    def _on_shown_image_changed(self):
        """Handle visual highlighting when shown image changes"""
        # TODO: Add highlighting logic if needed
        # For example, highlight the pattern preview when pattern is being shown
        shown_image = self.event_dispatcher.shown_image
        if shown_image == ShownImage.PATTERN:
            # Could add border or background color change here
            pass


class UvFocusFrame:  # crosses
    def __init__(self, parent, event_dispatcher: EventDispatcher, show_uv_focus=False):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher
        # UV focus frame - use predefined images - only shown in UV mode
        uv_focus_predefined = [("Plus Mark", "src/uvFocusImage/plus.png")]
        self.uv_focus_frame = ImageSelectFrame(
            self.frame,
            "UV Focus",
            self._on_uv_focus_change,
            # lambda t: event_dispatcher.set_uv_focus_image(self.uv_focus_image),
            # import_command in ImageSelectFrame --> on_select in PredefinedImageSelector
            predefined_images=uv_focus_predefined,
        )
        self.uv_focus_frame.frame.grid(row=1, column=0, padx=5, pady=5)

        # event_dispatcher.add_event_listener(Event.PATTERN_IMAGE_CHANGED, self._update_pattern_display)
        event_dispatcher.add_event_listener(Event.SHOWN_IMAGE_CHANGED, self._on_shown_image_changed)

    def _on_uv_focus_change(self, _):
        """Handle UV focus image selection and projection"""
        if self.uv_focus_frame.thumb.image:
            self.event_dispatcher.set_uv_focus_image(self.uv_focus_frame.thumb.image)
            self.event_dispatcher.set_shown_image(ShownImage.UV_FOCUS)

    def _on_shown_image_changed(self):
        """Handle visual highlighting when shown image changes"""
        # TODO: Add highlighting logic if needed
        # For example, highlight the UV focus selector when UV focus is being shown
        shown_image = self.event_dispatcher.shown_image
        if shown_image == ShownImage.UV_FOCUS:
            # Could add border or background color change here
            pass

    @property
    def uv_focus_image(self):
        """Get the currently selected UV focus image"""
        return self.uv_focus_frame.thumb.image if hasattr(self.uv_focus_frame, 'thumb') else None


class ChipFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.model = event_dispatcher
        self.path = StringVar()

        self.image_cache = dict()

        self.chip_select_frame = ttk.Frame(self.frame)
        self.chip_select_frame.grid(row=0, column=0)
        ttk.Label(self.chip_select_frame, text="Current Chip: ").grid(row=0, column=0)
        ttk.Label(self.chip_select_frame, textvariable=self.path).grid(row=0, column=1)

        def on_open():
            path = filedialog.askopenfilename(title="Open Chip")
            self.path.set(path)
            self.model.load_chip(path)

        def on_new():
            path = filedialog.asksaveasfilename(title="Create Chip As")
            self.path.set(path)
            self.model.new_chip()

        def on_save():
            path = filedialog.asksaveasfilename(title="Save As")
            if path != "":
                self.path.set(path)
                self.model.save_chip(self.path.get())

        def on_finish_layer():
            print("Layer finished!")
            self.model.add_chip_layer()
            self.model.save_chip(self.path.get())

        def on_delete_exposure():
            pair = self._selected_exposure()
            assert pair is not None
            yes = messagebox.askyesno(
                title="Delete Exposure",
                message="Are you sure you want to delete the selected exposure?",
            )
            if yes:
                self.model.delete_chip_exposure(pair[0], pair[1])
                if self.path.get() != "":
                    self.model.save_chip(self.path.get())

        def on_select(e, cur):
            if cur and len(self.cur_layer_view.selection()) > 0:
                self.prev_layer_view.selection_set([])
                self.delete_exposure_button["state"] = "normal"
            elif not cur and len(self.prev_layer_view.selection()) > 0:
                self.cur_layer_view.selection_set([])
                self.delete_exposure_button["state"] = "normal"
            else:
                self.delete_exposure_button["state"] = "disabled"

        def on_double_click(cur):
            pair = self._selected_exposure()
            assert pair is not None
            x, y, z = self.model.chip.layers[pair[0]].exposures[pair[1]].coords
            self.model.move_absolute({"x": x, "y": y, "z": z})

        def on_chip_changed():
            if len(self.model.chip.layers) < 2:
                self.prev_layer_select.configure(state="disabled")
                self.prev_layer_select.configure(to=0)
            else:
                self.prev_layer_select.configure(state="readonly")
                self.prev_layer_select.configure(to=len(self.model.chip.layers) - 2)
                self.prev_layer_select_var.set(str(len(self.model.chip.layers) - 2))
            self.refresh_prev_layer()
            self.refresh_cur_layer()
            if self.path.get() != "":
                self.model.save_chip(self.path.get())
            if len(self.model.chip.layers[-1].exposures) > 0:
                self.finish_layer_button.configure(state="normal")
            else:
                self.finish_layer_button.configure(state="disabled")

        def prev_layer_index_changed(a, b, c):
            self.refresh_prev_layer()

        self.model.add_event_listener(Event.CHIP_CHANGED, on_chip_changed)

        self.open_chip_button = ttk.Button(self.chip_select_frame, text="Open", command=on_open)
        self.open_chip_button.grid(row=0, column=2)
        self.new_chip_button = ttk.Button(self.chip_select_frame, text="New", command=on_new)
        self.new_chip_button.grid(row=0, column=3)
        self.save_chip_button = ttk.Button(self.chip_select_frame, text="Save As", command=on_save)
        self.save_chip_button.grid(row=0, column=4)
        self.finish_layer_button = ttk.Button(
            self.chip_select_frame, text="Finish Layer", command=on_finish_layer
        )
        self.finish_layer_button.grid(row=0, column=5)
        self.finish_layer_button.configure(state="disabled")
        self.delete_exposure_button = ttk.Button(
            self.chip_select_frame,
            text="Delete Exposure",
            command=on_delete_exposure,
            state="disabled",
        )
        self.delete_exposure_button.grid(row=0, column=6)

        self.layer_frame = ttk.Frame(self.frame)
        self.layer_frame.grid(row=1, column=0)

        self.prev_layer_frame = ttk.Labelframe(self.layer_frame, text="Previous Layer")
        self.prev_layer_frame.grid(row=0, column=0, sticky="ns")
        self.cur_layer_frame = ttk.Labelframe(self.layer_frame, text="Current Layer")
        self.cur_layer_frame.grid(row=0, column=1, sticky="ns")

        self.tree_view_style = ttk.Style()
        self.tree_view_style.configure("Treeview", rowheight=30)

        self.prev_layer_view = ttk.Treeview(
            self.prev_layer_frame, selectmode="browse", columns=("XYZ",), height=5
        )
        self.prev_layer_view.grid(row=0, column=0)
        self.prev_layer_view.bind("<<TreeviewSelect>>", lambda e: on_select(e, cur=False))
        self.prev_layer_view.bind("<Double-1>", lambda e: on_double_click(cur=False))
        self.cur_layer_view = ttk.Treeview(
            self.cur_layer_frame, selectmode="browse", columns=("XYZ",), height=5
        )
        self.cur_layer_view.grid(row=0, column=0)
        self.cur_layer_view.bind("<<TreeviewSelect>>", lambda e: on_select(e, cur=True))
        self.cur_layer_view.bind("<Double-1>", lambda e: on_double_click(cur=True))

        select_frame = ttk.Frame(self.prev_layer_frame)
        select_frame.grid(row=1, column=0)

        prev_layer_select_label = ttk.Label(select_frame, text="Select previous layer:")
        prev_layer_select_label.grid(row=0, column=0)

        self.prev_layer_select_var = StringVar()
        self.prev_layer_select_var.trace_add("write", prev_layer_index_changed)
        self.prev_layer_select = ttk.Spinbox(
            select_frame, from_=0, to=0, textvariable=self.prev_layer_select_var
        )
        self.prev_layer_select.configure(state="disabled")
        self.prev_layer_select.grid(row=0, column=1)

    def _selected_exposure(self):
        cur_sel = self.cur_layer_view.selection()
        if len(cur_sel) > 0:
            layer_idx, ex_idx = cur_sel[0].split("_")
            return int(layer_idx), int(ex_idx)
        prev_sel = self.prev_layer_view.selection()
        if len(prev_sel) > 0:
            layer_idx, ex_idx = prev_sel[0].split("_")
            return int(layer_idx), int(ex_idx)
        return None

    def _get_thumbnail(self, path: str):
        try:
            return self.image_cache[path][1]
        except KeyError:
            img = Image.open(path).resize((80, 45))
            photo = image_to_tk_image(img)
            self.image_cache[path] = (img, photo)
            return photo

    def refresh_prev_layer(self):
        chip = self.model.chip

        for item in self.prev_layer_view.get_children():
            self.prev_layer_view.delete(item)

        try:
            idx = int(self.prev_layer_select_var.get())
        except ValueError:
            print(
                f"Leaving previous layer empty because select var is {self.prev_layer_select_var.get()!r}"
            )
            return

        if len(chip.layers) < 2:
            return

        for i, ex in enumerate(chip.layers[idx].exposures):
            ex_id = f"{idx}_{i}"
            pos = f"{ex.coords[0]},{ex.coords[1]},{ex.coords[2]}"
            self.prev_layer_view.insert(
                "", "end", ex_id, image=self._get_thumbnail(ex.path), values=(pos,)
            )

    def refresh_cur_layer(self):
        chip = self.model.chip

        for item in self.cur_layer_view.get_children():
            self.cur_layer_view.delete(item)

        for i, ex in enumerate(chip.layers[-1].exposures):
            ex_id = f"{len(chip.layers) - 1}_{i}"
            pos = f"{ex.coords[0]},{ex.coords[1]},{ex.coords[2]}"
            self.cur_layer_view.insert(
                "", "end", ex_id, image=self._get_thumbnail(ex.path), values=(pos,)
            )


class ExposureFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        ttk.Label(self.frame, text="Exposure Time (ms)", anchor="w").grid(row=0, column=0)
        self.exposure_time_entry = IntEntry(self.frame, default=8000, min_value=0)
        self.exposure_time_entry.widget.grid(row=0, column=1, columnspan=2, sticky="nesw")

        def on_exposure_time_change(_a, _b, _c):
            event_dispatcher.exposure_time = self.exposure_time_entry._var.get()

        self.exposure_time_entry._var.trace_add("write", on_exposure_time_change)

        # Posterization
        def on_posterize_change(*args):
            event_dispatcher.set_posterize_strength(self._posterize_strength())

        def on_posterize_check():
            if self.posterize_enable_var.get():
                self.posterize_scale["state"] = "normal"
                self.posterize_cutoff_entry.widget["state"] = "normal"
            else:
                self.posterize_scale["state"] = "disabled"
                self.posterize_cutoff_entry.widget["state"] = "disabled"
            on_posterize_change()

        self.posterize_enable_var = BooleanVar()
        self.posterize_checkbutton = ttk.Checkbutton(
            self.frame,
            text="Posterize Cutoff (%)",
            command=on_posterize_check,
            variable=self.posterize_enable_var,
            onvalue=True,
            offvalue=False,
        )
        self.posterize_checkbutton.grid(row=2, column=0)
        self.posterize_strength_var = IntVar()
        self.posterize_strength_var.trace_add("write", on_posterize_change)
        self.posterize_scale = ttk.Scale(
            self.frame, variable=self.posterize_strength_var, from_=0.0, to=100.0
        )
        self.posterize_scale.grid(row=2, column=1, sticky="nesw")
        self.posterize_scale["state"] = "disabled"
        self.posterize_cutoff_entry = IntEntry(
            self.frame,
            var=self.posterize_strength_var,
            default=50,
            min_value=0,
            max_value=100,
        )
        self.posterize_cutoff_entry.widget.grid(row=2, column=2, sticky="nesw")
        self.posterize_cutoff_entry.widget["state"] = "disabled"

    # returns threshold percentage if posterizing is enabled, else None
    def _posterize_strength(self) -> Optional[int]:
        if self.posterize_enable_var.get():
            return self.posterize_cutoff_entry.get()
        else:
            return None


class PatterningFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)

        self.preview_tile = ttk.Label(self.frame, text="Next Pattern Tile", compound="top")
        self.preview_tile.grid(row=0, column=0)

        self.begin_patterning_button = ttk.Button(
            self.frame,
            text="Begin Patterning",
            command=lambda: event_dispatcher.begin_patterning(),
            state="enabled",
        )
        self.begin_patterning_button.grid(row=1, column=0)

        self.abort_patterning_button = ttk.Button(
            self.frame,
            text="Abort Patterning",
            command=lambda: event_dispatcher.abort_patterning(),
            state="disabled",
        )
        self.abort_patterning_button.grid(row=2, column=0)

        ttk.Label(self.frame, text="Exposure Progress", anchor="s").grid(row=3, column=0)
        self.exposure_progress = Progressbar(
            self.frame, orient="horizontal", mode="determinate", maximum=1000
        )
        self.exposure_progress.grid(row=4, column=0, sticky="ew")

        self.set_image(Image.new("RGB", (1, 1)))

        def on_change_patterning_status():
            if event_dispatcher.patterning_busy:
                self.begin_patterning_button["state"] = "disabled"
                self.abort_patterning_button["state"] = "normal"
            else:
                self.begin_patterning_button["state"] = "normal"
                self.abort_patterning_button["state"] = "disabled"

        event_dispatcher.add_event_listener(
            Event.PATTERNING_BUSY_CHANGED, on_change_patterning_status
        )
        event_dispatcher.add_event_listener(
            Event.PATTERN_IMAGE_CHANGED,
            lambda: self.set_image(event_dispatcher.pattern.processed()),
        )

        def on_progress_changed():
            self.exposure_progress["value"] = event_dispatcher.exposure_progress * 1000.0

        event_dispatcher.add_event_listener(
            Event.EXPOSURE_PATTERN_PROGRESS_CHANGED, on_progress_changed
        )

    def set_image(self, img: Image.Image):
        # TODO: What is the correct size?
        self.thumb_image = image_to_tk_image(img.resize(THUMBNAIL_SIZE))
        self.preview_tile.configure(image=self.thumb_image)


class RedModeFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher, ui_scale: float = 1.0):
        self.frame = ttk.Frame(parent, name="redmodeframe")
        self.event_dispatcher = event_dispatcher

        # Create left middle right sections
        self.left_frame = ttk.Frame(self.frame)
        self.left_frame.grid(row=0, column=0)

        self.middle_frame = ttk.Frame(self.frame)
        self.middle_frame.grid(row=0, column=1)

        self.right_frame = ttk.Frame(self.frame)
        self.right_frame.grid(row=0, column=2)

        # center the frames
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=0)
        self.frame.grid_columnconfigure(2, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        # Stage position controls (middle)
        self.stage_position_frame = StagePositionFrame(
            self.middle_frame, event_dispatcher, False, ui_scale=ui_scale
        )
        self.stage_position_frame.frame.grid(row=0, column=0)

        # test tiling check button & preview
        # self.tiling_check_frame  = TilingCheckFrame(self.middle_frame, event_dispatcher)
        # self.tiling_check_frame.frame.grid(row=0, column = 1)

        # Pattern preview display (right side)
        self.pattern_display = PatternDisplayFrame(self.right_frame, event_dispatcher)
        self.pattern_display.frame.grid(row=0, column=0)

        # Red focus image selection (left)
        self.red_select_var = StringVar(value="focus")
        self.red_focus_rb = ttk.Radiobutton(
            self.left_frame,
            variable=self.red_select_var,
            text="Red Focus",
            value=RedFocusSource.IMAGE.value,
        )
        self.solid_red_rb = ttk.Radiobutton(
            self.left_frame,
            variable=self.red_select_var,
            text="Solid Red",
            value=RedFocusSource.SOLID.value,
        )
        self.pattern_rb = ttk.Radiobutton(
            self.left_frame,
            variable=self.red_select_var,
            text="Same as Pattern",
            value=RedFocusSource.PATTERN.value,
        )
        self.inv_pattern_rb = ttk.Radiobutton(
            self.left_frame,
            variable=self.red_select_var,
            text="Inverse of Pattern",
            value=RedFocusSource.INV_PATTERN.value,
        )

        self.solid_red_rb.grid(row=0, column=0)
        self.pattern_rb.grid(row=1, column=0)
        self.inv_pattern_rb.grid(row=2, column=0)
        self.red_focus_rb.grid(row=3, column=0)

        self.red_focus_frame = ImageSelectFrame(
            self.left_frame,
            "Red Focus",
            lambda t: event_dispatcher.set_red_focus_image(self.red_focus_image()),
        )
        self.red_focus_frame.frame.grid(row=5, column=0)

        def on_radiobutton(*_):
            print(f"red select var {self.red_select_var.get()}")
            for s in RedFocusSource:
                if s.value == self.red_select_var.get():
                    event_dispatcher.set_red_focus_source(s)
                    break
            else:
                raise Exception()

        self.red_select_var.trace_add("write", on_radiobutton)

    def red_focus_image(self):
        return self.red_focus_frame.thumb.image


class UvModeFrame:
    def __init__(self, parent, event_dispatcher, ui_scale: float = 1.0):
        self.frame = ttk.Frame(parent, name="uvmodeframe")
        # Create left and right sections
        self.left_frame = ttk.Frame(self.frame)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self.middle_frame = ttk.Frame(self.frame)
        self.middle_frame.grid(row=0, column=1, sticky="ns")

        self.right_frame = ttk.Frame(self.frame)
        self.right_frame.grid(row=0, column=2, sticky="ns")

        # center the frames
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=0)
        self.frame.grid_columnconfigure(2, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        # Predefined UV focus image selection
        self.uv_focus_frame = UvFocusFrame(self.left_frame, event_dispatcher)
        self.uv_focus_frame.frame.grid(row=0, column=0, pady=(10, 0))

        # Stage position controls (middle)
        self.stage_position_frame = StagePositionFrame(
            self.middle_frame, event_dispatcher, True, ui_scale=ui_scale
        )
        self.stage_position_frame.frame.grid(row=0, column=0, sticky="n")

        # Pattern preview and UV focus selector (right side)
        # self.pattern_display = PatternDisplayFrame(self.right_frame, event_dispatcher)
        # self.pattern_display.frame.grid(row=0, column=0)

        # Exposure and patterning controls (right side, below images)
        self.exposure_frame = ExposureFrame(self.right_frame, event_dispatcher)
        self.exposure_frame.frame.grid(row=0, column=0)
        self.patterning_frame = PatterningFrame(self.right_frame, event_dispatcher)
        self.patterning_frame.frame.grid(row=1, column=0)


class PatternUploadFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        # Create container frame for centering
        container = ttk.Frame(self.frame)
        container.grid(row=0, column=0)

        # Main pattern upload section
        self.upload_frame = ttk.LabelFrame(container, text="Pattern Upload")
        self.upload_frame.grid(row=0, column=0)

        # Pattern selector (using existing ImageSelectFrame functionality)
        self.pattern_selector = ImageSelectFrame(
            self.upload_frame, "Select Pattern Image", self._on_pattern_upload
        )
        self.pattern_selector.frame.grid(row=0, column=0)

        # Pattern info display
        self.info_frame = ttk.LabelFrame(container, text="Pattern Information")
        self.info_frame.grid(row=1, column=0)

        self.pattern_path_var = StringVar(value="No pattern loaded")
        ttk.Label(self.info_frame, text="Current Pattern:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.info_frame, textvariable=self.pattern_path_var, foreground="blue").grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )

        # Pattern preview (larger than thumbnail)
        self.preview_frame = ttk.LabelFrame(container, text="Pattern Preview")
        self.preview_frame.grid(row=0, column=1, rowspan=2, padx=10)

        # Center the container
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        # Create larger preview image
        preview_size = (320, 240)  # Larger than THUMBNAIL_SIZE
        placeholder = Image.new("RGB", preview_size, "gray")
        self.preview_photo = image_to_tk_image(placeholder)
        self.preview_label = ttk.Label(self.preview_frame, image=self.preview_photo)
        self.preview_label.grid(row=0, column=0, padx=5, pady=5)

        # Upload instructions
        instruction_text = (
            "Upload your pattern image using the selector above. "
            "This pattern will be used in UV mode for lithography."
        )
        ttk.Label(self.upload_frame, text=instruction_text, wraplength=300).grid(
            row=1, column=0, padx=5, pady=5
        )

    def _on_pattern_upload(self, _):
        """Handle pattern upload"""
        if self.pattern_selector.thumb.image:
            # Update the event dispatcher with the new pattern
            self.event_dispatcher.set_pattern_image(
                self.pattern_selector.thumb.image, self.pattern_selector.thumb.path
            )

            # Update the info display
            if self.pattern_selector.thumb.path:
                filename = Path(self.pattern_selector.thumb.path).name
                self.pattern_path_var.set(filename)
            else:
                self.pattern_path_var.set("Pattern uploaded")

            # Update preview image
            if self.pattern_selector.thumb.image:
                preview_img = self.pattern_selector.thumb.image.copy()
                preview_img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                self.preview_photo = image_to_tk_image(preview_img)
                self.preview_label.configure(image=self.preview_photo)


class ModeSelectFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher, ui_scale: float = 1.0):
        self.notebook = ttk.Notebook(parent)

        # Add Pattern Upload tab first
        self.pattern_upload_frame = PatternUploadFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.pattern_upload_frame.frame, text="Pattern Upload")
        self.red_mode_frame = RedModeFrame(self.notebook, event_dispatcher, ui_scale=ui_scale)
        self.notebook.add(self.red_mode_frame.frame, text="Red Light Alignment Mode")
        self.uv_mode_frame = UvModeFrame(self.notebook, event_dispatcher, ui_scale=ui_scale)
        self.notebook.add(self.uv_mode_frame.frame, text="UV Exposure Mode")

        def on_tab_change():
            current_tab = self._current_tab()
            if current_tab == "uv":
                event_dispatcher.enter_uv_mode()
            elif current_tab == "red":
                event_dispatcher.enter_red_mode()

        self.notebook.bind("<<NotebookTabChanged>>", lambda _: on_tab_change())

        # def on_tab_event(evt):
        #  self.notebook.select(1 if evt == Event.EnterUvMode else 0)

        # event_dispatcher.add_event_listener(Event.EnterRedMode, lambda: on_tab_event(Event.EnterRedMode))
        # event_dispatcher.add_event_listener(Event.EnterUvMode, lambda: on_tab_event(Event.EnterUvMode))

    def _current_tab(self):
        selected = self.notebook.select()
        if "patternupload" in selected.lower() or self.notebook.index("current") == 0:
            return "pattern"
        elif "redmode" in selected or self.notebook.index("current") == 1:
            return "red"
        else:
            return "uv"


class GlobalSettingsFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher, enable_detection: bool = False):
        self.frame = ttk.LabelFrame(parent, text="Global Settings")

        def set_autofocus_on_mode_switch(*_):
            event_dispatcher.autofocus_on_mode_switch = self.autofocus_on_mode_switch_var.get()

        self.autofocus_on_mode_switch_var = BooleanVar(value=False)
        self.autofocus_on_mode_switch_check = ttk.Checkbutton(
            self.frame,
            text="Autofocus on Mode Change",
            variable=self.autofocus_on_mode_switch_var,
        )
        self.autofocus_on_mode_switch_check.grid(row=0, column=0, columnspan=2)
        self.autofocus_on_mode_switch_var.trace_add("write", set_autofocus_on_mode_switch)

        def set_realtime_detection(*_):
            event_dispatcher.realtime_detection = self.realtime_detection_var.get()

        self.realtime_detection_var = BooleanVar(value=enable_detection)
        self.realtime_detection_check = ttk.Checkbutton(
            self.frame,
            text="Detect alignment markers in real time",
            variable=self.realtime_detection_var,
            state="disabled" if event_dispatcher.model is None else "normal",
        )
        self.realtime_detection_check.grid(row=1, column=0, columnspan=2)
        self.realtime_detection_var.trace_add("write", set_realtime_detection)

        def do_align():
            h, w, _ = event_dispatcher.camera_image.shape
            markers, _ = detect_alignment_markers(
                event_dispatcher.model, event_dispatcher.camera_image
            )
            dx, dy = 0, 0
            if len(markers) == 0:
                return

            # Get alignment parameters from config
            alignment = event_dispatcher.config.alignment

            for m in markers:
                xy0, xy1 = m
                x0, y0 = xy0
                x1, y1 = xy1
                # compute normalized centers of the bounding box
                x = (x0 + x1) / 2 / w
                y = (y0 + y1) / 2 / h

                if x > 0.5:
                    dx += alignment.x_scale_factor * (alignment.right_marker_x / w - x)
                else:
                    dx += alignment.x_scale_factor * (alignment.left_marker_x / w - x)
                if y > 0.5:
                    dy += alignment.y_scale_factor * (alignment.bottom_marker_y / h - y)
                else:
                    dy += alignment.y_scale_factor * (alignment.top_marker_y / h - y)

            dx /= len(markers)
            dy /= len(markers)
            event_dispatcher.move_relative({'x': dx, 'y': dy})

            print(markers)

        self.alignbutton = ttk.Button(
            self.frame,
            text="Align :)",
            command=do_align,
            state="disabled" if event_dispatcher.model is None else "normal",
        )
        self.alignbutton.grid(row=2, column=1)

        # TODO: Fix this
        self.autofocus_button = ttk.Button(
            self.frame,
            text="Autofocus",
            command=lambda: event_dispatcher.autofocus(blue_only=event_dispatcher.in_uv()),
        )
        self.autofocus_button.grid(row=2, column=0, sticky="ew")

        # Maybe this should have a scale?
        # Or, even further, maybe this should just be the same as the interface for posterization strength?
        self.border_size_var = IntVar()
        self.border_label = ttk.Label(self.frame, text="Border Size (%)")
        self.border_label.grid(row=3, column=0)
        self.border_entry = IntEntry(
            self.frame, var=self.border_size_var, default=0, min_value=0, max_value=100
        )
        self.border_entry.widget.grid(row=3, column=1, sticky="nesw")

        def on_border_size_change(*_):
            event_dispatcher.set_border_size(self.border_size_var.get())

        self.border_size_var.trace_add("write", on_border_size_change)

        self.placeholder_photo = image_to_tk_image(Image.new("RGB", THUMBNAIL_SIZE, "black"))
        self.photo = None

        self.current_image = ttk.Label(self.frame, image=self.placeholder_photo)
        self.current_image.grid(row=4, column=0, columnspan=2)

        # Disable the autofocus button if autofocus is already running
        def movement_lock_changed():
            if event_dispatcher.movement_lock == MovementLock.LOCKED:
                self.autofocus_button.configure(state="disabled")
            else:
                self.autofocus_button.configure(state="normal")

        event_dispatcher.add_event_listener(Event.MOVEMENT_LOCK_CHANGED, movement_lock_changed)

        def shown_image_changed():
            img = event_dispatcher.current_image
            if img is None:
                self.current_image.configure(image=self.placeholder_photo)
            else:
                photo = image_to_tk_image(img.resize(THUMBNAIL_SIZE, Image.Resampling.NEAREST))
                self.current_image.configure(image=photo)
                self.photo = photo

        event_dispatcher.add_event_listener(Event.SHOWN_IMAGE_CHANGED, shown_image_changed)

        self.snapshot_frame = ttk.LabelFrame(self.frame, text="Snapshot Settings")
        self.snapshot_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)

        self.auto_snapshot_var = BooleanVar(value=event_dispatcher.auto_snapshot_on_uv)
        self.auto_snapshot_check = ttk.Checkbutton(
            self.snapshot_frame,
            text="Auto-save snapshot on UV mode entry",
            variable=self.auto_snapshot_var,
        )
        self.auto_snapshot_check.grid(row=0, column=0, columnspan=2)

        def on_auto_snapshot_change(*_):
            event_dispatcher.auto_snapshot_on_uv = self.auto_snapshot_var.get()

        self.auto_snapshot_var.trace_add("write", on_auto_snapshot_change)

        # Directory selection
        ttk.Label(self.snapshot_frame, text="Save Directory:").grid(row=1, column=0)
        self.directory_var = StringVar(value=str(event_dispatcher.snapshot_directory))
        self.directory_entry = ttk.Entry(
            self.snapshot_frame, textvariable=self.directory_var, state="readonly"
        )
        self.directory_entry.grid(row=1, column=1, sticky="ew")

        def choose_directory():
            dir_path = filedialog.askdirectory(
                initialdir=self.directory_var.get(),
                title="Select Snapshot Save Directory",
            )
            if dir_path:  # User didn't cancel
                event_dispatcher.set_snapshot_directory(Path(dir_path))
                self.directory_var.set(dir_path)

        self.choose_dir_button = ttk.Button(
            self.snapshot_frame, text="Choose Directory", command=choose_directory
        )
        self.choose_dir_button.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Configure grid weights for proper expansion
        self.snapshot_frame.columnconfigure(1, weight=1)


class ExposureHistoryFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.LabelFrame(parent, text="Exposure History")
        self.text = tkinter.Text(self.frame, width=40, height=10, wrap="none", state="disabled")
        self.h_scroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self.text.xview)
        self.text.configure(xscrollcommand=self.h_scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.frame.grid_columnconfigure(0, weight=1)
        self.event_dispatcher = event_dispatcher
        event_dispatcher.add_event_listener(Event.PATTERNING_BUSY_CHANGED, lambda: self._refresh())

    def _refresh(self):
        self.text["state"] = "normal"
        self.text.delete("1.0", "end")
        for exp_log in self.event_dispatcher.exposure_history[-10:]:
            t = exp_log.time.strftime("%H:%M:%S")
            line = f"{t} {exp_log.path} {int(exp_log.duration)}ms X:{exp_log.coords[0]} Y:{exp_log.coords[1]} Z:{exp_log.coords[2]}\n"

            if self.text.index("end-1c") != 1.0:
                self.text.insert("end", "\n")
            self.text.insert("end", line)
        self.text["state"] = "disabled"


class OffsetAmountFrame:
    def __init__(self, parent, label, default_offset):
        self.frame = ttk.LabelFrame(parent, text=label)

        offset_label = ttk.Label(self.frame, text="Offset (µm)")
        offset_label.grid(row=0, column=0)
        self.offset_var = StringVar(value=str(default_offset))
        self.offset_entry = ttk.Entry(self.frame, textvariable=self.offset_var, width=5)
        self.offset_entry.grid(row=0, column=1)
        amount_label = ttk.Label(self.frame, text="Amount")
        amount_label.grid(row=0, column=2)
        self.amount_var = StringVar(value="1")
        self.amount_spinbox = ttk.Spinbox(
            self.frame, from_=-20, to=20, textvariable=self.amount_var, width=3
        )
        self.amount_spinbox.grid(row=0, column=3)

# Will to changes
class TilingFrame:
    def __init__(self, parent, model: EventDispatcher):
        self.frame = ttk.LabelFrame(parent, text="Tiling")
        self.model = model

        self.red_to_uv_offset = -40

        self.overall_pattern_size_w = 0
        self.overall_pattern_size_h = 0

        # Defaults set based on DLP471TP and a 10x objective
        # 5.4 um Pixel Pitch
        # Width 10.368 mm
        # Height 5.832 mm
        # Move in X = 10.368mm / 10 = 1037um
        # Move in Y = 5.832 mm / 10 = 538.2 um ~ 539 um
        # Subtraction amounts are there to tune the offset for the alignment markers
        self.x_settings = OffsetAmountFrame(
            self.frame, "X", 1037 - 54
        )  # Move amount between exposures in X
        self.y_settings = OffsetAmountFrame(
            self.frame, "Y", 539 - 27
        )  # Move amount between exposures in y

        # Tiling verisons of alignment
        def detect_alignment_markers_tiling(
            yolo_model, image, draw_rectangle=False, edge=None, edge_fraction=0.25
        ):
            # Detects alignment markers and optionally filters detections by image edge(s).
            # yolo_model: YOLO model
            # image: image to detect on
            # draw_rectangle: If True, draw rectangles
            # edge: 'left', 'right', 'top', or a list like ['left', 'right'] where markers are expected
            # none means that markers are expect on all edges
            # edge_fraction: Fraction of width/height considered as edge region

            detections = []
            display_image = image.copy()
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_height, original_width = image_rgb.shape[:2]
                resized = cv2.resize(image_rgb, (640, 640))
                results = yolo_model(resized)
                boxes = results[0].boxes

                if isinstance(edge, str):
                    edge = [edge]  # allow single string or list

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 * original_width / 640)
                    x2 = int(x2 * original_width / 640)
                    y1 = int(y1 * original_height / 640)
                    y2 = int(y2 * original_height / 640)
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    # If edge filtering is enabled
                    if edge is not None:
                        if 'left' in edge and x_center > original_width * edge_fraction:
                            continue
                        if 'right' in edge and x_center < original_width * (1 - edge_fraction):
                            continue
                        if 'top' in edge and y_center > original_height * edge_fraction:
                            continue

                    detections.append(((x1, y1), (x2, y2)))
                    if draw_rectangle:
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                print(f"Detected {len(detections)} marker(s)")
            except Exception as e:
                print(f"Detection failed: {e}")

            return detections, display_image

        def do_align_tiling(edge):
            # edge = ['left', 'right', 'top']
            h, w, _ = model.camera_image.shape

            # Detect markers on the left, right, and top edges
            markers, _ = detect_alignment_markers_tiling(model.model, model.camera_image, edge)
            if len(markers) == 0:
                print("No markers detected.")
                return

            alignment = model.config.alignment
            dx, dy = 0.0, 0.0
            count_x, count_y = 0, 0

            for m in markers:
                xy0, xy1 = m
                x0, y0 = xy0
                x1, y1 = xy1
                x = (x0 + x1) / 2 / w
                y = (y0 + y1) / 2 / h

                # Horizontal alignment (left/right markers)
                if x < 0.5:
                    dx += alignment.x_scale_factor * (alignment.left_marker_x / w - x)
                    count_x += 1
                elif x > 0.5:
                    dx += alignment.x_scale_factor * (alignment.right_marker_x / w - x)
                    count_x += 1

                # Vertical alignment (top markers only)
                if y < 0.3:  # top region
                    dy += alignment.y_scale_factor * (alignment.top_marker_y / h - y)
                    count_y += 1

            # Average corrections based on detected edges
            if count_x > 0:
                dx /= count_x
            if count_y > 0:
                dy /= count_y

            # Move accordingly (if no top markers, dy=0)
            # If a small amount of alignment is needed move the image otherwise move the stage since we have far more percision in moving the image than the stage
            # The con of this is that large movements of the image result in cropping of the image
            # TODO calibrate the stage move threshold
            if dx < 10 or dy < 10:
                # move the image instead of the stage
                model.set_image_position(dx, dy, t=0)
            else:
                model.move_relative({'x': dx, 'y': dy})
                print(
                    f"Alignment correction: dx={dx:.5f}, dy={dy:.5f} using {len(markers)} markers."
                )

        # function that takes in an arbitrary sized image composed of 3840x2160 tiles
        # with shared alignment marks that are 200 pixels from the edge
        def split_image_with_overlap(
            image_path,
            tile_width=3840,
            tile_height=2160,
            overlap_x=200,
            overlap_y=200,
            output_dir="tiles",
        ):
            img = Image.open(image_path)
            img_w, img_h = img.size
            self.overall_pattern_size_w = img_w
            self.overall_pattern_size_h = img_h
            os.makedirs(output_dir, exist_ok=True)

            stride_x = tile_width - overlap_x
            stride_y = tile_height - overlap_y

            # Compute all top-left coordinates
            x_positions = []
            y_positions = []

            # Horizontal positions
            x = 0
            while True:
                if x + tile_width >= img_w:
                    x = max(0, img_w - tile_width)
                    x_positions.append(x)
                    break
                x_positions.append(x)
                x += stride_x

            # Vertical positions
            y = 0
            while True:
                if y + tile_height >= img_h:
                    y = max(0, img_h - tile_height)
                    y_positions.append(y)
                    break
                y_positions.append(y)
                y += stride_y

            tile_count = 0
            # Set amount of tiles for later use when exposing
            self.x_settings.amount_var.set(str(len(x_positions)))
            self.y_settings.amount_var.set(str(len(y_positions)))
            # Crop and Save the tile images
            for tile_id_y, top in enumerate(y_positions):
                for tile_id_x, left in enumerate(x_positions):
                    right = left + tile_width
                    bottom = top + tile_height

                    box = (left, top, right, bottom)
                    tile = img.crop(box)
                    tile.save(os.path.join(output_dir, f"tile_{tile_id_y},{tile_id_x}.png"))
                    tile_count += 1

            print("X amount = " + str(self.x_settings.amount_var))
            print("Y amount = " + str(self.y_settings.amount_var))
            print(f"Saved {tile_count} tiles to {output_dir}")

        # function that patterns a single tile
        def pattern_for_tile(
            self,
            model,
            x_start,
            x_dir,
            x_idx,
            x_offset,
            y_start,
            y_dir,
            y_idx,
            y_offset,
            y_idx_max,
            x_idx_max,
            tile_dir="tiles",
        ):
            # change image
            image_path = tile_dir + "/tile_" + str(y_idx) + "," + str(x_idx) + ".png"
            current_tile = Image.open(image_path)
            model.set_pattern_image(current_tile, image_path)
            # move to the next position if not the first tile
            # the first tile is exposed where the operator(user of the stepper) places it
            if ~(x_idx == 0 & y_idx == 0):
                self.model.move_absolute(
                    {
                        "x": x_start + x_dir * x_idx * x_offset,
                        "y": y_start + y_dir * y_idx * y_offset,
                    }
                )
            # Red autofocus
            self.model.autofocus(blue_only=False)

            # align to previous alignment marks if not first tile
            if ~(x_idx == 0 & y_idx == 0):
                if x_idx != 0 & x_idx != x_idx_max:
                    if y_idx % 2 == 0:
                        do_align_tiling('left')
                    else:
                        do_align_tiling('right')
                else:
                    do_align_tiling('top')

            # UV offset + autofocus, expose, back to red
            self.model.expose_pattern(self.red_to_uv_offset)

            # TODO Add second exposure of the alignment markers
            # I tried doing this with a non blocking delay but didnt have success
            # I think that loading a pattern of the alignment marks that is hardcoded into the software might be the best bet

        def segment():
            # create tile directory and segment images
            split_image_with_overlap(model.pattern_image_path)
            # load the first tile for operator placement
            model.set_red_focus_source(RedFocusSource.PATTERN)
            image_path = "tiles/tile_" + str(0) + "," + str(0) + ".png"
            current_tile = Image.open(image_path)
            model.set_pattern_image(current_tile, image_path)

        def on_begin():
            model.set_red_focus_source(RedFocusSource.PATTERN)

            x_amount = int(self.x_settings.amount_var.get())
            x_offset = int(self.x_settings.offset_var.get())
            x_dir = 1 if x_amount > 0 else -1
            x_amount = abs(x_amount)

            y_amount = int(self.y_settings.amount_var.get())
            y_offset = int(self.y_settings.offset_var.get())
            y_dir = 1 if y_amount > 0 else -1
            y_amount = abs(y_amount)

            x_start, y_start = (
                self.model.stage_setpoint[0],
                self.model.stage_setpoint[1],
            )
            print(f"x_start {x_start}, y_start = {y_start}")

            # Move in Snake pattern with left to right on even rows and right to left on odd rows
            tiles_done = 0
            tiling_t0 = time.time()
            print(f"[Tiling] patterning {x_amount} x {y_amount} tiles")
            for y_idx in range(y_amount):
                if y_idx % 2 == 0:
                    for x_idx in range(x_amount):
                        pattern_for_tile(
                            self,
                            model,
                            x_start,
                            -x_dir,
                            x_idx,
                            x_offset,
                            y_start,
                            -y_dir,
                            y_idx,
                            y_offset,
                            y_idx_max=y_amount,
                            x_idx_max=x_amount,
                        )
                        tiles_done += 1
                        print(
                            f"[Tiling] patterned tile {tiles_done}/{x_amount * y_amount} "
                            f"(row {y_idx + 1}, col {x_idx + 1}), {time.time() - tiling_t0:.0f} s elapsed"
                        )
                else:
                    for x_idx in range(x_amount - 1, -1, -1):
                        pattern_for_tile(
                            self,
                            model,
                            x_start,
                            -x_dir,
                            x_idx,
                            x_offset,
                            y_start,
                            -y_dir,
                            y_idx,
                            y_offset,
                            y_idx_max=y_amount,
                            x_idx_max=x_amount,
                        )
                        tiles_done += 1
                        print(
                            f"[Tiling] patterned tile {tiles_done}/{x_amount * y_amount} "
                            f"(row {y_idx + 1}, col {x_idx + 1}), {time.time() - tiling_t0:.0f} s elapsed"
                        )

        # TODO IMPLEMENT ABORT
        def on_abort():
            pass

        # Segment Images must be done before begining tiling
        # TODO enforce above
        # Tiling check must be done before segment images if needed
        self.tiling_check_button = TilingCheckFrame(self.frame, model)
        self.tiling_check_button.frame.grid(row=0, column=0)
        self.segment_images_button = ttk.Button(
            self.frame, text="Segement Images", command=segment, state="enabled"
        )
        self.segment_images_button.grid(row=1, column=0)
        self.begin_tiling_button = ttk.Button(
            self.frame, text="Begin Tiling", command=on_begin, state="enabled"
        )
        self.begin_tiling_button.grid(row=2, column=0)
        self.abort_tiling_button = ttk.Button(
            self.frame, text="Abort Tiling", command=on_abort, state="disabled"
        )
        self.abort_tiling_button.grid(row=3, column=0)

        def open_map():
            # the tile grid is whatever the tiling panel already holds (set when a pattern is segmented)
            run_map_and_placement(
                self.frame,
                model,
                tile_counts=lambda: (self.y_settings.amount_var.get(), self.x_settings.amount_var.get()),
                red_to_uv_offset=self.red_to_uv_offset,
            )

        self.map_button = ttk.Button(
            self.frame, text="Map + Placement", command=open_map, state="enabled"
        )
        self.map_button.grid(row=4, column=0)

        self.finished_piece_button = ttk.Button(
            self.frame,
            text="Map finished piece",
            command=lambda: run_finished_piece(self.frame, model),
            state="enabled",
        )
        self.finished_piece_button.grid(row=5, column=0)


# --------------------------------------------------------------------------------------
# Camera map -> pattern placement -> finished-piece scan (the two map buttons of the Tiling panel)
# --------------------------------------------------------------------------------------
def slot_name(r: int, c: int, rows: int, cols: int) -> str:
    if rows == 2 and cols == 2:
        return ("top" if r == 0 else "bottom") + "-" + ("left" if c == 0 else "right")
    return f"row {r + 1}, column {c + 1}"


# --------------------------------------------------------------------------------------
# Pattern picker
# --------------------------------------------------------------------------------------
class PatternPickerDialog:
    CELL_W, CELL_H, GAP = 168, 95, 14  # 16:9 cells like the frames themselves

    def __init__(
        self,
        parent: tk.Misc,
        rows: int = 2,
        cols: int = 2,
        sample_paths: Optional[Sequence[str]] = None,
        title: str = "Select pattern images",
    ):
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.sample_paths = list(sample_paths) if sample_paths else None
        self.paths: List[Optional[str]] = [None] * self.n
        self.geos: List[Optional[CrossGeometry]] = [None] * self.n
        self.thumbs: List[Optional[ImageTk.PhotoImage]] = [None] * self.n
        self.current = 0
        self.result: Optional[List[str]] = None

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.resizable(False, False)
        try:
            self.win.transient(parent)
        except tk.TclError:
            pass

        self.header = ttk.Label(self.win, font=("TkDefaultFont", 12, "bold"), wraplength=520)
        self.header.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 4), sticky="w")
        w = self.cols * (self.CELL_W + self.GAP) + self.GAP
        h = self.rows * (self.CELL_H + self.GAP) + self.GAP
        self.canvas = tk.Canvas(self.win, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=1, column=0, padx=14, pady=6, sticky="n")
        side = ttk.Frame(self.win)
        side.grid(row=1, column=1, padx=(0, 14), pady=6, sticky="nsew")
        side.columnconfigure(0, minsize=300)
        self.path_var = tk.StringVar(value="(no file chosen)")
        ttk.Label(side, textvariable=self.path_var, wraplength=300, justify="left").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar()
        self.status = tk.Label(side, textvariable=self.status_var, wraplength=300, justify="left", fg="#444")
        self.status.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.browse_btn = ttk.Button(side, text="Browse...", command=self.browse)
        self.browse_btn.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.sample_btn = ttk.Button(side, text="Use sample set", command=self.use_samples)
        self.sample_btn.grid(row=3, column=0, sticky="w", pady=(6, 0))
        if not self.sample_paths:
            self.sample_btn.state(["disabled"])

        bar = ttk.Frame(self.win)
        bar.grid(row=2, column=0, columnspan=2, padx=14, pady=(6, 12), sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.back_btn = ttk.Button(bar, text="< Back", command=self.back)
        self.back_btn.grid(row=0, column=1, padx=4)
        self.next_btn = ttk.Button(bar, text="Next >", command=self.next)
        self.next_btn.grid(row=0, column=2, padx=4)
        self._refresh()

    # ---- logic (also callable from tests) ----
    def try_set(self, path: str) -> Optional[str]:
        """Accept ``path`` for the current slot.  Returns an error message, or None on success."""
        try:
            img = load_image(path)
            geo = measure_cross_geometry(img)
        except Exception as e:  # noqa: BLE001 - shown to the user
            return str(e)
        for k, g in enumerate(self.geos):
            if g is not None and k != self.current and not same_layout(g, geo):
                return (
                    f"Cross layout differs from pattern {k + 1} (frame {geo.frame_w}x{geo.frame_h}, "
                    f"crosses {geo.cross_w}x{geo.cross_h}). All tiles must share one layout."
                )
        tw, th = self.CELL_W - 8, self.CELL_H - 8
        small = cv2.resize(img[..., :3] if img.ndim == 3 else img, (tw, th), interpolation=cv2.INTER_AREA)
        small = np.clip(small.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)  # keep thin lines visible
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB) if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        self.thumbs[self.current] = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.paths[self.current] = path
        self.geos[self.current] = geo
        self._ok = (
            f"OK: {geo.frame_w}x{geo.frame_h} px, crosses {geo.cross_w}x{geo.cross_h} px, "
            f"cross pitch {geo.pitch_x} x {geo.pitch_y} px"
        )
        return None

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.win,
            title=f"Image for pattern {self.current + 1}",
            initialdir=SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else None,
            filetypes=[("Pattern images", "*.png *.bmp *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.choose(path)

    def choose(self, path: str) -> bool:
        err = self.try_set(path)
        if err:
            self.status.configure(fg="#B00020")
            self.status_var.set("Not accepted: " + err)
            return False
        self.status.configure(fg="#1B6E2D")
        self.status_var.set(self._ok)
        self._refresh()
        return True

    def use_samples(self) -> None:
        if not self.sample_paths:
            return
        for k, p in enumerate(self.sample_paths[: self.n]):
            self.current = k
            if not self.choose(p):
                break
        else:
            self.current = self.n - 1
        self._refresh()

    def next(self) -> None:
        if self.paths[self.current] is None:
            return
        if self.current < self.n - 1:
            self.current += 1
            self.status_var.set("")
            self._refresh()
        else:
            self.result = [p for p in self.paths if p is not None]
            self.win.destroy()

    def back(self) -> None:
        if self.current > 0:
            self.current -= 1
            self.status_var.set("")
            self._refresh()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> Optional[List[str]]:
        self.win.grab_set()
        self.win.wait_window()
        return self.result

    # ---- drawing ----
    def _refresh(self) -> None:
        r, c = divmod(self.current, self.cols)
        self.header.configure(
            text=f"Pattern {self.current + 1} of {self.n}: choose the image for the "
            f"{slot_name(r, c, self.rows, self.cols)} tile (row {r + 1}, column {c + 1})"
        )
        p = self.paths[self.current]
        self.path_var.set(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" if p else "(no file chosen)")
        self.next_btn.configure(text="Finish" if self.current == self.n - 1 else "Next >")
        self.next_btn.state(["!disabled"] if p else ["disabled"])
        self.back_btn.state(["!disabled"] if self.current > 0 else ["disabled"])
        cv = self.canvas
        cv.delete("all")
        for s in range(self.n):
            rr, cc = divmod(s, self.cols)
            x0 = self.GAP + cc * (self.CELL_W + self.GAP)
            y0 = self.GAP + rr * (self.CELL_H + self.GAP)
            x1, y1 = x0 + self.CELL_W, y0 + self.CELL_H
            if s == self.current:
                cv.create_rectangle(x0 - 5, y0 - 5, x1 + 5, y1 + 5, fill="#FFE2B0", outline="#E07B00", width=4)
            fill = "#F4F4F4" if self.thumbs[s] is None else "#000000"
            cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#888" if s != self.current else "#E07B00", width=2)
            if self.thumbs[s] is not None:
                cv.create_image(x0 + 4, y0 + 4, image=self.thumbs[s], anchor="nw")
                cv.create_rectangle(x0, y0, x0 + 26, y0 + 22, fill="#FFFFFF", outline="")
                cv.create_text(x0 + 13, y0 + 11, text=str(s + 1), font=("TkDefaultFont", 11, "bold"), fill="#1B6E2D")
            else:
                cv.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2 - (8 if s == self.current else 0), text=str(s + 1),
                    font=("TkDefaultFont", 26, "bold"), fill="#E07B00" if s == self.current else "#999",
                )
                if s == self.current:
                    cv.create_text((x0 + x1) / 2, y1 - 14, text="picking this one", fill="#9A5400", font=("TkDefaultFont", 9))


# --------------------------------------------------------------------------------------
# Area dialog: how many tiles -> how much area gets mapped
# --------------------------------------------------------------------------------------
class AreaDialog:
    """Asks for the number of tiles (wide x high) and shows the area that will be mapped: one more complete
    frame than tiles in each direction, centred on the current stage position.  Confirm starts the map."""

    CANVAS_W, CANVAS_H = 400, 240

    def __init__(
        self,
        parent: tk.Misc,
        cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
        cam_size: Tuple[int, int] = (1920, 1080),
        rows: int = 2,
        cols: int = 2,
    ):
        self.cam_um_per_px = cam_um_per_px
        self.cam_size = cam_size
        self.result: Optional[Tuple[int, int]] = None  # (tile rows, tile cols)
        self.win = tk.Toplevel(parent)
        self.win.title("Map: area to map")
        self.win.protocol("WM_DELETE_WINDOW", self.cancel)
        self.win.resizable(False, False)
        ttk.Label(self.win, text="Map the area around the current stage position.\nHow many tiles will the pattern have?").grid(
            row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w"
        )
        self.cols_var = tk.IntVar(value=cols)
        self.rows_var = tk.IntVar(value=rows)
        form = ttk.Frame(self.win)
        form.grid(row=1, column=0, columnspan=2, padx=14, sticky="w")
        ttk.Label(form, text="tiles wide").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=8, width=4, textvariable=self.cols_var, command=self.refresh).grid(row=0, column=1, padx=8)
        ttk.Label(form, text="tiles high").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=8, width=4, textvariable=self.rows_var, command=self.refresh).grid(row=1, column=1, padx=8)
        for v in (self.cols_var, self.rows_var):
            v.trace_add("write", lambda *_: self.refresh())
        self.canvas = tk.Canvas(self.win, width=self.CANVAS_W, height=self.CANVAS_H, bg="#e8e8e8", highlightthickness=0)
        self.canvas.grid(row=2, column=0, padx=14, pady=10)
        self.info = tk.Text(self.win, width=58, height=15, font=("Courier", 9), wrap="word", state="disabled")
        self.info.grid(row=2, column=1, padx=(0, 14), pady=10, sticky="n")
        bar = ttk.Frame(self.win)
        bar.grid(row=3, column=0, columnspan=2, pady=(0, 14), padx=14, sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.confirm_btn = ttk.Button(bar, text="Confirm and map", command=self.confirm)
        self.confirm_btn.grid(row=0, column=1, padx=4)
        self.refresh()

    def _counts(self) -> Optional[Tuple[int, int]]:
        try:
            r, c = int(self.rows_var.get()), int(self.cols_var.get())
        except (tk.TclError, ValueError):
            return None
        return (r, c) if 1 <= r <= 8 and 1 <= c <= 8 else None

    def info_lines(self) -> List[str]:
        counts = self._counts()
        if counts is None:
            return ["enter 1 to 8 tiles in each direction"]
        r, c = counts
        pm = area_for_tiles(r, c)
        w_um, h_um = pm.area_size_um
        cfg = MapConfig(um_per_px=self.cam_um_per_px, cam_size=self.cam_size)
        mw, mh = scan_area_um(pm, cfg)
        sc = scan_summary((mw, mh), cfg)
        x0 = min(t.dest[0] for t in pm.tiles)
        y0 = min(t.dest[1] for t in pm.tiles)
        x1 = max(t.dest[2] for t in pm.tiles)
        y1 = max(t.dest[3] for t in pm.tiles)
        bw, bh = (x1 - x0) * pm.um_per_px[0], (y1 - y0) * pm.um_per_px[1]
        lines = [
            f"pattern: {c} wide x {r} high = {c * r} tiles",
            f"area to map: {c + 1} x {r + 1} complete frames",
            f"space occupied: {w_um:.0f} x {h_um:.0f} um  ({w_um / 1000:.2f} x {h_um / 1000:.2f} mm)",
            f"pattern block inside: {bw:.0f} x {bh:.0f} um, centred",
            f"mapped: {mw:.0f} x {mh:.0f} um (every tile's alignment view)",
            "",
            f"map photos: {sc['cols']} x {sc['rows']} ({sc['photos']}), step "
            f"{sc['step_fraction'][0] * 100:.0f}% x / {sc['step_fraction'][1] * 100:.0f}% y of a frame (max 75%)",
            f"estimated mapping time: {sc['est_s']:.0f} s (limit 300 s), with autofocus after each move",
            "",
            "The current stage position becomes the centre of the area.",
            "Tiles can be inserted once the map is done.",
        ]
        if sc["est_s"] > 300:
            lines.insert(8, "WARNING: this is over the 5 minute mapping target")
        return lines

    def refresh(self) -> None:
        lines = self.info_lines()
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        self.canvas.delete("all")
        counts = self._counts()
        self.confirm_btn.state(["!disabled"] if counts else ["disabled"])
        if counts is None:
            return
        r, c = counts
        fr, fc = r + 1, c + 1
        pad = 22
        cell = min((self.CANVAS_W - 2 * pad) / fc, (self.CANVAS_H - 2 * pad) / fr * 16 / 9)  # cell width; frames are 16:9
        cell_h = cell * 9 / 16
        x0 = (self.CANVAS_W - cell * fc) / 2
        y0 = (self.CANVAS_H - cell_h * fr) / 2
        self.canvas.create_rectangle(x0, y0, x0 + cell * fc, y0 + cell_h * fr, fill="white", outline="#888")
        for i in range(1, fc):
            self.canvas.create_line(x0 + i * cell, y0, x0 + i * cell, y0 + cell_h * fr, fill="#d0d0d0")
        for j in range(1, fr):
            self.canvas.create_line(x0, y0 + j * cell_h, x0 + cell * fc, y0 + j * cell_h, fill="#d0d0d0")
        # the pattern block sits half a frame in from every edge, one tile per cell
        for j in range(r):
            for i in range(c):
                tx, ty = x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell_h
                self.canvas.create_rectangle(tx, ty, tx + cell, ty + cell_h, fill="#cfe3ff", outline="#2b6cb0")
                self.canvas.create_text(tx + cell / 2, ty + cell_h / 2, text=str(j * c + i + 1), fill="#2b6cb0", font=("TkDefaultFont", 11, "bold"))
        self.canvas.create_text(self.CANVAS_W / 2, y0 + cell_h * fr + 12, text=f"{fc} x {fr} frames", fill="#555")

    def confirm(self) -> None:
        counts = self._counts()
        if counts is None:
            return
        self.result = counts
        self.win.destroy()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> Optional[Tuple[int, int]]:
        self.win.grab_set()
        self.win.wait_window()
        return self.result


# --------------------------------------------------------------------------------------
# Placement approval
# --------------------------------------------------------------------------------------
@dataclass
class ApprovedPlacement:
    placement: PlacementMap
    stage_targets: List[Tuple[float, float]]
    area_center_um: Tuple[float, float]
    seams: List[dict]
    images: List[np.ndarray] = field(default_factory=list, repr=False)  # the patterns, for the run folder
    paths: List[str] = field(default_factory=list)
    map_result: Optional[MapResult] = field(default=None, repr=False)

    def lines(self) -> List[str]:
        out = self.placement.report()
        for k, (x, y) in enumerate(self.stage_targets):
            out.append(f"stage target tile {k + 1}: x {x:.1f} um, y {y:.1f} um")
        return out


class PlacementApprovalWindow:
    CANVAS_W, CANVAS_H = 880, 500

    def __init__(
        self,
        parent: tk.Misc,
        images: Sequence[np.ndarray],
        paths: Optional[Sequence[str]] = None,
        rows: int = 2,
        cols: int = 2,
        area_rows: int = 3,
        area_cols: int = 3,
        area_center_um: Tuple[float, float] = (0.0, 0.0),
        map_result: Optional[MapResult] = None,
        um_per_px: Tuple[float, float] = PROJ_UM_PER_PX,
        on_approve: Optional[Callable[[ApprovedPlacement], None]] = None,
        sign: Tuple[int, int] = (1, -1),
        mapped_area_um: Optional[Tuple[float, float]] = None,
    ):
        self.mapped_area_um = mapped_area_um
        self.images = list(images)
        self.masks = [exposure_mask(i) for i in self.images]  # cached; used by every redraw
        self.paths = list(paths) if paths else []
        self.rows, self.cols, self.area_rows, self.area_cols = rows, cols, area_rows, area_cols
        self.area_center = area_center_um
        self.map_result = map_result
        self.um_per_px = um_per_px
        self.on_approve = on_approve
        self.sign = sign
        self.geo = measure_cross_geometry(self.images[0])
        self.seams = check_seams(self.images, rows, cols, self.geo)
        self.spec_notes = spec_mismatches(self.geo)  # differences from the CROSS_* inputs in lib/tiling.py
        self.nudge = [0.0, 0.0]
        self.result: Optional[ApprovedPlacement] = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        self.win = tk.Toplevel(parent)
        self.win.title("Placement map: review and approve")
        self.win.protocol("WM_DELETE_WINDOW", self.cancel)
        self.canvas = tk.Canvas(self.win, width=self.CANVAS_W, height=self.CANVAS_H, bg="#e8e8e8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        side = ttk.Frame(self.win)
        side.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        self.mode_var = tk.StringVar(value="trimmed")
        box = ttk.LabelFrame(side, text="Exposure mode")
        box.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(box, text="Split at the crosses (footprints)", value="trimmed", variable=self.mode_var, command=self.refresh).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="Whole frames at half-frame offset", value="whole", variable=self.mode_var, command=self.refresh).grid(row=1, column=0, sticky="w")
        self.bg_var = tk.BooleanVar(value=False)
        self.bg_check = ttk.Checkbutton(side, text="Show camera map underneath", variable=self.bg_var, command=self.refresh)
        self.bg_check.grid(row=1, column=0, sticky="w", pady=(6, 0))
        if map_result is None or map_result.mosaic is None:
            self.bg_check.state(["disabled"])

        nb = ttk.LabelFrame(side, text="Nudge placement (um)")
        nb.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.step_var = tk.StringVar(value="10")
        ttk.Entry(nb, textvariable=self.step_var, width=6).grid(row=0, column=1)
        ttk.Button(nb, text="^", width=3, command=lambda: self.move(0, -1)).grid(row=0, column=3)
        ttk.Button(nb, text="<", width=3, command=lambda: self.move(-1, 0)).grid(row=1, column=2)
        ttk.Button(nb, text="Reset", width=6, command=self.reset_nudge).grid(row=1, column=3)
        ttk.Button(nb, text=">", width=3, command=lambda: self.move(1, 0)).grid(row=1, column=4)
        ttk.Button(nb, text="v", width=3, command=lambda: self.move(0, 1)).grid(row=2, column=3)
        ttk.Label(nb, text="step").grid(row=0, column=0)

        self.info = tk.Text(side, width=64, height=24, font=("Courier", 9), wrap="word", state="disabled")
        self.info.grid(row=3, column=0, pady=(8, 0), sticky="nsew")
        side.rowconfigure(3, weight=1)
        bar = ttk.Frame(side)
        bar.grid(row=4, column=0, pady=(10, 0), sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.approve_btn = ttk.Button(bar, text="Approve placement", command=self.approve)
        self.approve_btn.grid(row=0, column=1, padx=4)
        self.refresh()

    # ---- state ----
    def placement(self) -> PlacementMap:
        return build_placement_map(
            self.geo, self.rows, self.cols, self.area_rows, self.area_cols,
            mode=self.mode_var.get(), um_per_px=self.um_per_px, nudge_um=(self.nudge[0], self.nudge[1]),
        )

    def move(self, dx: int, dy: int) -> None:
        try:
            step = float(self.step_var.get())
        except ValueError:
            step = 10.0
        self.nudge[0] += dx * step
        self.nudge[1] += dy * step
        self.refresh()

    def reset_nudge(self) -> None:
        self.nudge = [0.0, 0.0]
        self.refresh()

    # ---- drawing ----
    def refresh(self) -> None:
        pm = self.placement()
        pad = 600
        total_w = pm.area_size_px[0] + 2 * pad
        scale = min(self.CANVAS_W / total_w, self.CANVAS_H / (pm.area_size_px[1] + 2 * pad))
        bg = None
        if self.bg_var.get() and self.map_result is not None and self.map_result.mosaic is not None:
            bg = background_from_map(self.map_result, pm, scale)
        img, _ = render_placement(pm, self.masks, scale=scale, background=bg, pad_px=pad)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.CANVAS_W // 2, self.CANVAS_H // 2, image=self._photo)

        lines = pm.report()
        lines.append(f"nudge: {self.nudge[0]:+.0f} um x, {self.nudge[1]:+.0f} um y")
        lines.append("")
        lines.append("crosses that must coincide between neighbours:")
        for s in self.seams:
            a, b = s["a"] + 1, s["b"] + 1
            where = "left|right" if s["dir"] == "h" else "top|bottom"
            lines.append(f"  tile {a} / tile {b} ({where}): " + ("match" if s["ok"] else f"MISMATCH, {s['mismatch_px']} px differ - wrong order?"))
        lines.append("")
        if self.spec_notes:
            lines.append("cross inputs (CROSS_* in lib/tiling.py) DIFFER from this pattern:")
            lines += [f"  {n}" for n in self.spec_notes]
            lines.append("  keep cross size and distances identical on every tile!")
        else:
            lines.append("cross size/distances match the CROSS_* inputs in lib/tiling.py")
        self.too_big = False
        if self.mapped_area_um is not None:
            need = scan_area_um(pm, self.map_result.cfg) if self.map_result is not None else pm.area_size_um
            over = (need[0] - self.mapped_area_um[0], need[1] - self.mapped_area_um[1])
            if max(over) > 25.0:
                self.too_big = True
                lines.append(
                    f"WARNING: this placement needs {need[0]:.0f} x {need[1]:.0f} um mapped but only "
                    f"{self.mapped_area_um[0]:.0f} x {self.mapped_area_um[1]:.0f} um was mapped (cross layout differs from CROSS_*)"
                )
        targets = pm.stage_targets(self.area_center, self.sign)
        lines.append("")
        lines.append("stage targets (um):")
        for k, (x, y) in enumerate(targets):
            lines.append(f"  tile {k + 1}: x {x:9.1f}   y {y:9.1f}")
        if self.map_result is not None:
            rep = budget_report(self.map_result.timing.get("total_s", 0.0), n_tiles=len(pm.tiles), model=TilingTimeModel())
            lines.append("")
            lines.append(f"map took {rep['map_s']:.0f} s (limit {rep['map_limit_s']:.0f} s)")
            lines.append(
                f"tiling overhead ~{rep['tile_overhead_s']:.0f} s (assumed), leaves up to "
                f"{rep['max_exposure_per_tile_s']:.0f} s exposure per tile in {rep['tile_limit_s']:.0f} s"
            )
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        bad = any(not s["ok"] for s in self.seams)
        self.approve_btn.configure(
            text="Approve placement (outside the mapped area!)" if self.too_big
            else "Approve placement (seam mismatch!)" if bad else "Approve placement"
        )

    # ---- actions ----
    def approve(self) -> None:
        pm = self.placement()
        self.result = ApprovedPlacement(
            pm, pm.stage_targets(self.area_center, self.sign), self.area_center, self.seams,
            self.images, self.paths, self.map_result,
        )
        if self.on_approve:
            self.on_approve(self.result)
        self.win.destroy()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()


# --------------------------------------------------------------------------------------
# Whole flow
# --------------------------------------------------------------------------------------
class ModelRig:
    """Rig for the old UI: wraps the EventDispatcher ('model') the way TilingCheckFrame uses it.

    Every photo it hands out is cropped to the part of the camera frame the projector lights
    (CAMERA_LIT_FRACTION), so the dark edge never reaches the map, the matching or the focus score."""

    def __init__(self, model, z: Optional[float] = None, red_to_uv_offset: float = -40.0):
        self.model = model
        self.z = model.stage_setpoint[2] if z is None else z
        self.red_to_uv_offset = red_to_uv_offset  # z move from red focus to UV focus (TilingFrame's value)
        self.pattern_dir: Optional[str] = None  # where expose() saves each tile's exposed pattern (run folder)
        self._focused_once = False  # the first autofocus does the coarse stage, later ones only the fine one

    @contextmanager
    def solid_red(self):
        """Light the chip with plain red while photos are taken (map, alignment, finished piece).  Red does not
        expose the resist; a projected red pattern would sit still in every photo and pull the matching."""
        m = self.model
        if not getattr(m, "camera", None) or not hasattr(m, "red_focus_source"):
            yield
            return
        prev_source, prev_shown = m.red_focus_source, m.shown_image
        self.project_and_confirm(
            lambda: (m.set_red_focus_source(RedFocusSource.SOLID), m.enter_red_mode(mode_switch_autofocus=False)),
            "solid red on",
        )
        try:
            yield
        finally:
            m.set_red_focus_source(prev_source)
            m.set_shown_image(prev_shown)

    def expose(
        self, pattern: np.ndarray, proj_um_per_px: Tuple[float, float], crop: Tuple[int, int, int, int], name: str = "tile"
    ) -> None:
        """Expose one tile here: its pattern with everything outside the footprint black, then the same UV
        offset / UV autofocus / exposure / back-to-red steps as Begin Tiling.

        The exposed pattern is saved as a PNG first: the exposure log keeps its path, and the Chip panel
        opens that file for the exposure's thumbnail (a path that is not a file failed with errno 2)."""
        m = self.model
        img = cv2_to_pil(tile_pattern(pattern, crop))
        folder = self.pattern_dir or tempfile.mkdtemp(prefix="exposed_patterns_")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}_exposed_pattern.png")
        img.save(path)
        m.set_pattern_image(img, path)
        m.expose_pattern(self.red_to_uv_offset)
        self.z = m.stage_setpoint[2]

    def move_to(self, x_um: float, y_um: float) -> None:
        # move_absolute fills missing axes with 0, so always pass z
        if self.model.move_absolute({"x": x_um, "y": y_um, "z": self.z}) is False:
            raise RuntimeError(f"stage refused the move to ({x_um:.0f}, {y_um:.0f}) um (out of bounds?)")

    def settle(self, seconds: float) -> None:
        self.model.non_blocking_delay(seconds)

    def autofocus(self) -> None:
        """Toby's autofocus, ported to the old UI: project a red ArUco grid and golden-section search z for the
        sharpest image (coarse 2x2 grid +-250 um, then fine 4x4 grid +-10 um).

        The first call does both stages.  Later calls (the next photo, a few um of tilt away) run the fine stage
        only, and fall back to the coarse one when the fine search ends on the edge of its range.  The projector
        is put back to what it showed before, so the map photos are not taken with the grid on the chip."""
        m = self.model
        if not getattr(m, "camera", None):
            return
        prev_source, prev_image, prev_shown = m.red_focus_source, m.red_focus_image, m.shown_image
        sp = m.stage_setpoint
        z_start = float(sp[2])
        n_moves = 0
        try:
            m.set_autofocus_busy(True)
            proj_size = tuple(m.hardware.projector.size())

            def run_stage(grid_n: int, half: float, thr: float, centre: float):
                def show_grid() -> None:
                    m.set_red_focus_image(Image.fromarray(aruco_grid_image(grid_n, proj_size)).convert("RGBA"))
                    m.set_red_focus_source(RedFocusSource.IMAGE)
                    m.enter_red_mode(mode_switch_autofocus=False)

                # the camera confirms the grid is on (no fixed wait for it to appear)
                self.project_and_confirm(show_grid, f"autofocus {grid_n}x{grid_n} grid on")

                def measure(z: float) -> Optional[float]:
                    nonlocal n_moves
                    if m.move_absolute({"x": sp[0], "y": sp[1], "z": z}) is False:
                        return None
                    n_moves += 1
                    self.settle(FOCUS_SETTLE_S)
                    return sobel_focus_score(self.grab())

                return golden_section_focus(measure, centre, half, thr)

            centre = z_start
            res = None
            stages = FOCUS_STAGES[1:] if self._focused_once else FOCUS_STAGES
            for k, (grid_n, half, thr) in enumerate(stages):
                res = run_stage(grid_n, half, thr, centre)
                if res is None:
                    break
                centre = res[0]
                # a fine search that ends on the edge of its range has not found the peak: do the coarse stage too
                if self._focused_once and k == 0 and abs(centre - z_start) >= half - 2 * thr:
                    res = run_stage(*FOCUS_STAGES[0], centre)
                    if res is not None:
                        res = run_stage(*FOCUS_STAGES[1], res[0])
                    break
            if res is None:
                status("autofocus: a z move was refused (limit?), staying at the starting height")
                m.move_absolute({"x": sp[0], "y": sp[1], "z": z_start})
            else:
                m.move_absolute({"x": sp[0], "y": sp[1], "z": res[0]})
                self._focused_once = True
                status(f"autofocus: z {z_start:.1f} -> {res[0]:.1f} um, {n_moves} z moves")
            self.settle(FOCUS_SETTLE_S)  # the stage settling after the last z move (not the projector)
        finally:

            def restore() -> None:
                m.set_red_focus_image(prev_image)
                m.set_red_focus_source(prev_source)
                m.set_shown_image(prev_shown)

            try:
                # Don't return until the camera shows the grid gone: the map photo is grab()bed straight after
                # this, and before this fix nothing ran the Tk loop in between, so the projector window had not
                # even been repainted and camera_image was still the last frame of the focus search (grid on).
                self.project_and_confirm(restore, "autofocus grid off")
            finally:
                m.set_autofocus_busy(False)
        self.z = m.stage_setpoint[2]  # autofocus moved z; keep later moves at the focused height

    def _next_frame(self) -> np.ndarray:
        """Run the Tk loop until the camera delivers a frame that was not here yet, and return a copy of it.

        camera_image is only refreshed by CameraFrame._on_new_frame, which runs from the Tk loop every 66 ms, so
        reading it without running the loop returns whatever frame happened to be there last."""
        m = self.model
        seq0 = m.camera_image_seq
        t0 = time.monotonic()
        while m.camera_image_seq == seq0:
            m.root.update()
            if time.monotonic() - t0 > CAMERA_DEAD_S:
                raise RuntimeError(f"camera delivered no new frame in {CAMERA_DEAD_S:.0f} s (stopped?)")
        return np.array(m.camera_image)

    def project_and_confirm(self, change: Callable[[], None], what: str) -> Optional[np.ndarray]:
        """Make a projector change and return once the camera shows it has finished, with that frame.

        Done: a new frame differs from the old picture by more than the camera noise, and matches the frame
        before it, so the projector is no longer changing (no half-faded frame).  The time this takes is whatever
        the projector window, the display and the camera need, measured each time rather than assumed."""
        m = self.model
        if not getattr(m, "camera", None):
            change()
            return None
        try:
            old_a, old = self._next_frame(), self._next_frame()  # two frames of the old picture: camera noise
        except Exception:
            change()  # still make the change (it may be the grid going off) before reporting the camera problem
            raise
        change()
        m.root.update_idletasks()  # paint the projector window now: Tk only repaints a Label when the loop runs
        noise = frame_difference(old_a, old)
        limit = max(PROJ_CHANGE_NOISE_FACTOR * noise, PROJ_CHANGE_MIN_DN)
        prev: Optional[np.ndarray] = None
        for n in range(1, PROJ_CHANGE_MAX_FRAMES + 1):
            frame = self._next_frame()
            if prev is not None and frame_difference(frame, old) > limit and frame_difference(frame, prev) <= limit:
                if self.verbose_projector:
                    status(f"{what}: confirmed by the camera after {n} new frames")
                return frame
            prev = frame
        status(
            f"{what}: the camera picture did not settle within {PROJ_CHANGE_MAX_FRAMES} frames "
            f"(noise {noise:.2f}, limit {limit:.2f}); carrying on"
        )
        return prev

    verbose_projector = True  # print how many frames each projector change took (to see the real latency)

    def grab(self) -> np.ndarray:
        """The lit middle of the next frame the camera delivers after this call (never the one already there)."""
        if getattr(self.model, "camera", None):
            return crop_center(self._next_frame(), CAMERA_LIT_FRACTION)
        img = self.model.camera_image
        if img is None:
            raise RuntimeError("no camera image available")
        return crop_center(np.array(img), CAMERA_LIT_FRACTION)

    def cam_size(self) -> Tuple[int, int]:
        """(width, height) of the photos grab() hands out."""
        f = self.grab()
        return (f.shape[1], f.shape[0])


class ProgressWindow:
    def __init__(self, parent: tk.Misc, title: str = "Camera map"):
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.resizable(False, False)
        self.label = ttk.Label(self.win, text="starting...", width=60)
        self.label.grid(row=0, column=0, padx=14, pady=(14, 6))
        self.bar = ttk.Progressbar(self.win, length=420, maximum=1.0)
        self.bar.grid(row=1, column=0, padx=14, pady=6)
        self.aborted = False
        ttk.Button(self.win, text="Abort", command=self._abort).grid(row=2, column=0, pady=(6, 14))

    def _abort(self) -> None:
        self.aborted = True

    def update(self, frac: float, msg: str) -> None:
        self.bar["value"] = frac
        self.label.configure(text=msg)
        self.win.update()

    def close(self) -> None:
        self.win.destroy()


DEFAULT_TILE_GRID = (2, 2)  # (rows, cols) used when the tiling panel holds no grid yet (the 2x2 proof of concept)
MAX_TILE_GRID = 8


def default_tile_counts(get_panel_counts: Optional[Callable[[], Tuple[object, object]]] = None) -> Tuple[Tuple[int, int], str]:
    """Tile grid (rows, cols) for the map, taken from the tiling panel instead of being typed in again.

    The tiling panel's X / Y "Amount" fields hold the grid (segmenting a big pattern fills them in).  Negative
    amounts only mean 'go the other way', so the absolute value is used.  If the panel still has its untouched
    1 x 1 default (or cannot be read) the 2 x 2 proof-of-concept grid is used.  Returns (rows, cols), why.
    """
    if get_panel_counts is not None:
        try:
            rows_raw, cols_raw = get_panel_counts()
            rows, cols = abs(int(float(str(rows_raw)))), abs(int(float(str(cols_raw))))
        except (TypeError, ValueError):
            rows = cols = 0
        if rows * cols > 1:
            clamped = (min(rows, MAX_TILE_GRID), min(cols, MAX_TILE_GRID))
            note = "" if clamped == (rows, cols) else f" (limited to {MAX_TILE_GRID} per direction)"
            return clamped, f"{cols} x {rows} tiles from the tiling panel{note}"
    r, c = DEFAULT_TILE_GRID
    return (r, c), f"{c} x {r} tiles (default, the tiling panel holds no grid yet)"


def run_map_and_placement(
    parent: tk.Misc,
    model,
    cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
    on_approved: Optional[Callable[[ApprovedPlacement], None]] = None,
    tile_counts: Optional[Callable[[], Tuple[object, object]]] = None,
    red_to_uv_offset: float = -40.0,
) -> Optional[ApprovedPlacement]:
    """Tile grid (taken from the tiling panel, shown in the area dialog to confirm) -> camera map around the
    current stage position -> ONLY THEN pick the pattern for each tile -> placement approval -> (asked)
    pattern the tiles with closed-loop alignment -> re-map the finished piece to check the alignment."""
    rig = ModelRig(model, red_to_uv_offset=red_to_uv_offset)
    cam_size = rig.cam_size()
    (rows0, cols0), why = default_tile_counts(tile_counts)
    status(f"tile grid: {why}")
    t_dialog = time.perf_counter()
    counts = AreaDialog(parent, cam_um_per_px, cam_size, rows=rows0, cols=cols0).show()
    dialog_s = time.perf_counter() - t_dialog
    if not counts:
        return None
    tile_rows, tile_cols = counts
    area_pm = area_for_tiles(tile_rows, tile_cols)
    cfg = MapConfig(um_per_px=cam_um_per_px, cam_size=cam_size)
    map_um = scan_area_um(area_pm, cfg)
    center = (float(model.stage_setpoint[0]), float(model.stage_setpoint[1]))
    status(
        f"mapping {map_um[0]:.0f} x {map_um[1]:.0f} um (every tile's alignment view of a {tile_cols} x {tile_rows} "
        f"pattern) around stage ({center[0]:.0f}, {center[1]:.0f}) um"
    )
    recorder = RunRecorder(label="map")  # one folder per run, next to this file
    recorder.add_timing("area dialog", dialog_s, "confirm the tile grid", "user")
    try:
        model.map_run = recorder  # the finished-piece scan writes into the same folder
    except Exception:  # noqa: BLE001
        pass
    prog = ProgressWindow(parent)
    t_map = time.perf_counter()
    try:
        with rig.solid_red():
            result = map_area(rig, cfg, map_um, center, prog.update, lambda: prog.aborted)
    except Exception as e:  # noqa: BLE001
        prog.close()
        recorder.log(f"mapping stopped: {e}")
        messagebox.showerror("Camera map", f"Mapping stopped: {e}")
        return None
    prog.close()
    print("\n".join(result.report()))
    recorder.add_timing("map", time.perf_counter() - t_map, f"{result.captured} photos, {'ok' if result.ok else 'FAILED'}", "machine")
    t_save = time.perf_counter()
    recorder.save_map(result)  # photos + combined map are kept even when the map fails (useful for debugging)
    recorder.add_timing("saving map photos and files", time.perf_counter() - t_save, "", "detail")
    if not result.ok:
        messagebox.showwarning("Camera map", "\n".join(result.report()) + f"\n\nPhotos saved in:\n{recorder.dir}")
        return None

    # the map is good: now (and only now) the tiles can be inserted
    sample_paths = [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]
    sample_paths = sample_paths if (tile_rows, tile_cols) == (2, 2) and all(os.path.exists(p) for p in sample_paths) else None
    t_pick = time.perf_counter()
    paths = PatternPickerDialog(parent, tile_rows, tile_cols, sample_paths).show()
    recorder.add_timing("pattern picking", time.perf_counter() - t_pick, "choose the pattern for each tile", "user")
    if not paths:
        recorder.log("no patterns chosen")
        return None
    images = [load_image(p) for p in paths]
    t_approve = time.perf_counter()
    win = PlacementApprovalWindow(
        parent, images, paths, rows=tile_rows, cols=tile_cols, area_rows=tile_rows + 1, area_cols=tile_cols + 1,
        area_center_um=center, map_result=result, on_approve=on_approved, mapped_area_um=map_um,
    )
    win.win.grab_set()
    win.win.wait_window()
    recorder.add_timing("placement approval", time.perf_counter() - t_approve, "check and approve the placement", "user")
    if win.result is not None:
        print("\n".join(win.result.lines()))
        t_place = time.perf_counter()
        recorder.save_placement(win.result.placement, images, paths, center, result)
        recorder.add_timing("saving placement preview", time.perf_counter() - t_place, "", "detail")
        try:
            model.map_placement = win.result  # kept for the exposure step that comes next
        except Exception:  # noqa: BLE001
            pass
        n = len(win.result.stage_targets)
        if messagebox.askyesno(
            "Pattern tiles",
            f"Expose the {n} tiles now?\n\nEach tile: move to its target, align under red light (crosses already "
            "printed by earlier tiles, else the chip's own features against the map), then UV exposure with "
            "that tile's pattern.  Afterwards the finished piece is re-mapped to check the alignment.",
        ):
            run_tiles(parent, model, win.result, recorder, rig)
            run_finished_piece(parent, model, cam_um_per_px, rig)
    else:
        recorder.log("placement not approved")
    return win.result


def run_tiles(parent: tk.Misc, model, placement: ApprovedPlacement, recorder: RunRecorder, rig: ModelRig) -> Optional[dict]:
    """Pattern every tile of an approved placement: closed-loop alignment, then UV exposure (pattern_tiles)."""
    mr = placement.map_result
    if mr is None:
        messagebox.showerror("Pattern tiles", "This placement has no camera map to align against.")
        return None
    yolo = getattr(model, "model", None)
    rig.pattern_dir = os.path.join(recorder.dir, "3_align")  # each tile's exposed pattern, next to its alignment report
    prog = ProgressWindow(parent, "Pattern tiles")
    t0 = time.perf_counter()
    try:
        with rig.solid_red():
            res = pattern_tiles(
                rig, mr, placement.placement, placement.images, mr.cfg, placement.area_center_um,
                closed_loop=True, use_crosses=True, recorder=recorder,
                progress=lambda msg: prog.update(0.5, msg),
                ml_detector=yolo_detector(yolo) if yolo is not None else None,
                should_abort=lambda: prog.aborted,
            )
    except Exception as e:  # noqa: BLE001
        prog.close()
        recorder.log(f"patterning stopped: {e}")
        messagebox.showerror("Pattern tiles", f"Patterning stopped: {e}")
        return None
    prog.close()
    done = len(res["exposed"])
    recorder.add_timing("patterning", time.perf_counter() - t0, f"{done} tiles", "machine")
    recorder.log(f"patterned {done} of {len(placement.stage_targets)} tiles, alignment driven by {res['driven_by']}")
    return res


def run_finished_piece(
    parent: tk.Misc, model, cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX, rig: Optional[ModelRig] = None
) -> Optional[MapResult]:
    """After patterning: quick re-map of the same area, saved as separate photos + one stitched image with the
    expected tile outlines drawn on it, in the run folder of the approved placement (4_finished_piece)."""
    placement: Optional[ApprovedPlacement] = getattr(model, "map_placement", None)
    recorder: Optional[RunRecorder] = getattr(model, "map_run", None)
    if placement is None or recorder is None:
        messagebox.showinfo("Camera map", "Run Map + Placement and approve a placement first.")
        return None
    rig = rig or ModelRig(model)
    cfg = MapConfig(um_per_px=cam_um_per_px, cam_size=rig.cam_size())
    prog = ProgressWindow(parent, "Finished piece")
    try:
        with rig.solid_red():
            res = map_finished_piece(
                rig, cfg, scan_area_um(placement.placement, cfg), placement.area_center_um, recorder,
                placement.placement, prog.update, lambda: prog.aborted,
            )
    except Exception as e:  # noqa: BLE001
        prog.close()
        recorder.log(f"finished-piece scan stopped: {e}")
        messagebox.showerror("Camera map", f"Finished-piece scan stopped: {e}")
        return None
    prog.close()
    messagebox.showinfo("Camera map", "\n".join(res.report()) + f"\n\nSaved in:\n{recorder.dir}")
    return res


class ProjectorDisplayFrame:
    """Frame to display what the projector is currently showing"""

    def __init__(
        self,
        parent,
        event_dispatcher: EventDispatcher,
        display_size: tuple[int, int] = (320, 180),
    ):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        # Main label frame
        self.display_frame = ttk.LabelFrame(self.frame, text="Projector Output")
        self.display_frame.grid(row=0, column=0)

        # Create placeholder image
        self.display_size = display_size
        placeholder = Image.new("RGB", self.display_size, "black")
        self.photo = image_to_tk_image(placeholder)

        # Display label
        self.label = ttk.Label(self.display_frame, image=self.photo, relief="solid", borderwidth=2)
        self.label.grid(row=0, column=0, padx=5, pady=5)

        # Status label showing current mode
        self.status_var = StringVar(value="Status: Clear")
        self.status_label = ttk.Label(self.display_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, padx=5, pady=5)

        # Listen for projector changes
        event_dispatcher.add_event_listener(Event.SHOWN_IMAGE_CHANGED, self._update_display)
        event_dispatcher.add_event_listener(Event.PATTERN_IMAGE_CHANGED, self._update_display)
        event_dispatcher.add_event_listener(Event.IMAGE_ADJUST_CHANGED, self._update_display)
        event_dispatcher.add_event_listener(Event.PATTERNING_BUSY_CHANGED, self._update_display)

        # Force initial update
        # self.event_dispatcher.root.after(100, self._update_display)

    def _update_display(self):
        """Update the display when projector content changes"""
        shown_image = self.event_dispatcher.shown_image

        # Update status text
        status_map = {
            ShownImage.CLEAR: "Status: Clear (No Output)",
            ShownImage.PATTERN: "Status: Pattern (UV Exposure)",
            ShownImage.FLATFIELD: "Status: Flatfield Correction",
            ShownImage.RED_FOCUS: "Status: Red Focus Mode",
            ShownImage.UV_FOCUS: "Status: UV Focus Pattern",
        }
        self.status_var.set(status_map.get(shown_image, "Status: Unknown"))

        # Get the appropriate processed image based on mode
        # Note: When patterning, we check patterning_busy flag as well
        img = None
        if shown_image == ShownImage.RED_FOCUS:
            img = self.event_dispatcher.red_focus.processed()
        # pattern case above uv focus case: when set_patterning_busy(True) is called,
        # shown_image is never changed to PATTERN during patterning - it stays as UV_FOCUS
        elif shown_image == ShownImage.PATTERN or self.event_dispatcher.patterning_busy:
            img = self.event_dispatcher.pattern.processed()
        elif shown_image == ShownImage.UV_FOCUS:
            img = self.event_dispatcher.uv_focus.processed()
        elif shown_image == ShownImage.FLATFIELD:
            # Flatfield might not be implemented, use pattern as fallback
            img = self.event_dispatcher.pattern.processed()

        # Update image
        if img is None or (
            shown_image == ShownImage.CLEAR and not self.event_dispatcher.patterning_busy
        ):
            # Show black placeholder when clear
            placeholder = Image.new("RGB", self.display_size, "black")
            self.photo = image_to_tk_image(placeholder)
        else:
            display_img = img.copy()
            display_img.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            self.photo = image_to_tk_image(display_img)

        self.label.configure(image=self.photo)

    def resize(self, display_size: tuple[int, int]):
        self.display_size = display_size
        self._update_display()


class TilingCheckFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        self.capture_button = ttk.Button(
            self.frame,
            text="Capture & Stitch Chip Imges",
            command=self.capture_and_stitch,
        )
        self.capture_button.grid(row=0, column=0)

        self.preview_label = ttk.Label(self.frame)
        self.preview_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)

    def capture_and_stitch(self):
        self.img = self.event_dispatcher.pattern_image
        self.img_w, self.img_h = self.img.size  # 3840*2, 2160*2 # in pixels
        print("self.img_w, self.img_h: ", self.img_w, self.img_h)

        self.tile_width, self.tile_height = (
            3840,
            2160,
        )  # in pixels, defined in TilingFrame

        # the distance that the stage will move in um
        stride_x, stride_y = 798, 448  # 1037 / 1.3 and 583 / 1.3
        # crop image pixels
        # each snapshot captured is 1920 x 1080 pixels
        crop_x, crop_y = round(1920 * CAMERA_LIT_FRACTION), round(1080 * CAMERA_LIT_FRACTION)  # 1477, 831
        # Disable button during capture
        self.capture_button.config(state='disabled', text="Capturing...")
        self.frame.update()

        stitched_image = self.takeAndStitchMapImages(stride_x, stride_y, crop_x, crop_y)
        if stitched_image:
            self.display_image(stitched_image, crop_x, crop_y)
            print("Stitching complete!")
        else:
            print("Failed to stitch images")

        self.capture_button.config(state='normal', text="Capture & Stitch Chip Imges")

    def display_image(self, pil_image, crop_x, crop_y):
        display_img = pil_image.copy()
        display_img = display_img.resize((crop_x // 4, crop_y // 4), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        self.preview_label.config(image=photo)
        self.preview_label.image = photo  # type: ignore

    def capture_current_image(self):
        # Get the camera view from the event dispatcher
        if (
            hasattr(self.event_dispatcher, 'camera_image')
            and self.event_dispatcher.camera_image is not None
        ):
            camera_image = self.event_dispatcher.camera_image
            pil_image = Image.fromarray(camera_image)
            return pil_image
        else:
            print("No camera image available")
            return None

    def takeAndStitchMapImages(self, stride_x, stride_y, crop_x, crop_y):
        """
        Take snapshots num_cols * num_rows times, move in snake pattern
        Move in stride_x and y um in distance
        Crop all snapshots to crop_x, crop_y pixels in the center
        Paste them to the blank canvas
        Overlay the pattern image in the center
        """
        # calculate how many times the stage will move: num cols & rows
        total_x_um = self.img_w * 1037 / 3840
        total_y_um = self.img_h * 583 / 2160
        num_cols = int(total_x_um // stride_x)
        num_rows = int(total_y_um // stride_y)
        if num_cols * stride_x < total_x_um:
            num_cols += 1
        if num_rows * stride_y < total_y_um:
            num_rows += 1
        print("num_cols, num_rows: ", num_cols, num_rows)

        # large blank canvas
        stitched_width = num_cols * crop_x
        stitched_height = num_rows * crop_y
        stitched_image = Image.new('RGB', (stitched_width, stitched_height), color='black')

        # get starting position: center the chip
        orig_x, orig_y, orig_z = self.event_dispatcher.stage_setpoint
        delta_x_um = (num_cols * stride_x - total_x_um) / 2
        delta_y_um = (num_rows * stride_y - total_y_um) / 2
        start_x = orig_x - delta_x_um
        start_y = orig_y - delta_y_um

        # move in snake pattern with left to right on even rows
        # and right to left on odd rows
        for row in range(num_rows):
            current_y = start_y + row * stride_y

            if row % 2 == 0:
                col_range = range(num_cols)
                first_x = start_x
            else:
                col_range = range(num_cols - 1, -1, -1)
                first_x = start_x + (num_cols - 1) * stride_x

            # move to the next row
            self.event_dispatcher.move_absolute({"x": first_x, "y": current_y, "z": orig_z})
            self.event_dispatcher.non_blocking_delay(2)

            # columns in this row
            for idx, col in enumerate(col_range):
                current_x = start_x + col * stride_x
                if idx > 0:  # idx = 0 first one don't need to move
                    self.event_dispatcher.move_absolute(
                        {"x": current_x, "y": current_y, "z": orig_z}
                    )
                    self.event_dispatcher.non_blocking_delay(2.5)

                captured_image = self.capture_current_image()
                # crop
                orig_width, orig_height = captured_image.size
                # each snapshot captured is 1920 x 1080 pixels
                left = (orig_width - crop_x) // 2
                top = (orig_height - crop_y) // 2
                right = left + crop_x
                bottom = top + crop_y
                cropped_img = captured_image.crop((left, top, right, bottom))
                # paste
                x_pos = col * crop_x
                y_pos = (num_rows - 1 - row) * crop_y
                stitched_image.paste(cropped_img, (x_pos, y_pos))

                self.event_dispatcher.non_blocking_delay(0.5)
                self.frame.update()

        # Return to starting position
        self.event_dispatcher.move_absolute({"x": orig_x, "y": orig_y, "z": orig_z})

        # Overlay the pattern image at center with 50% transparency
        pattern_img = self.img.copy().convert('RGBA')  # pattern image
        # resize pattern image to match the snapshot's pixel
        # each snapshot has half of each tile's w and h
        pattern_img = pattern_img.resize(
            (self.img_w // 2, self.img_h // 2), Image.Resampling.LANCZOS
        )
        pattern_img.putalpha(int(255 * 0.5))
        center_x = (stitched_width - self.img_w // 2) // 2
        center_y = (stitched_height - self.img_h // 2) // 2
        stitched_image = stitched_image.convert('RGBA')
        stitched_image.paste(pattern_img, (center_x, center_y), pattern_img)
        stitched_image = stitched_image.convert('RGB')

        return stitched_image


class MapFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher, canvas_size: int = 350):
        self.frame = ttk.LabelFrame(parent)
        self.event_dispatcher = event_dispatcher

        # Map dimensions in micrometers
        self.map_size_um = 10000.0  # 1 cm * 1 cm

        # Canvas size in pixels
        self.canvas_size = canvas_size

        # Pattern dimensions: from DLP projector datasheet
        self.pattern_w = 1037
        self.pattern_h = 583

        # canvas with plain background
        self.canvas = tkinter.Canvas(
            self.frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='#3F9490',
        )
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # coordinates of exposed patterns (list of tuples: (x, y))
        self.pattern_markers = []

        event_dispatcher.add_event_listener(Event.STAGE_POSITION_CHANGED, self._on_position_changed)
        event_dispatcher.add_event_listener(Event.PATTERNING_FINISHED, self._on_pattern_exposed)
        event_dispatcher.add_event_listener(Event.CHIP_CHANGED, self._on_chip_changed)

        self._redraw_all()

    def _um_to_pixels(self, um_x, um_y):
        """
        Convert micrometer coordinates to canvas pixel coordinates.
        (0, 0) in micrometers is at the center of the canvas.
        """
        scale = self.canvas_size / self.map_size_um

        # Add half map size to shift origin to center
        pixel_x = (um_x + self.map_size_um / 2) * scale
        pixel_y = (um_y + self.map_size_um / 2) * scale

        return pixel_x, pixel_y

    def _um_size_to_pixels(self, um_width, um_height):
        """Convert micrometer dimensions to pixel dimensions"""
        scale = self.canvas_size / self.map_size_um
        return um_width * scale, um_height * scale

    def _draw_pattern_marker(self, x_um, y_um):
        """Draw a blue rectangle given x and y"""
        x1_px, y1_px = self._um_to_pixels(x_um, y_um)
        w_px, h_px = self._um_size_to_pixels(self.pattern_w, self.pattern_h)

        x2_px = x1_px + w_px
        y2_px = y1_px + h_px

        # shift from bottom down to bottom up
        y1_px = self.canvas_size - y1_px
        y2_px = self.canvas_size - y2_px

        marker = self.canvas.create_rectangle(x1_px, y1_px, x2_px, y2_px, fill='#7BB7B7', width=0)

        return marker

    def _draw_current_position(self):
        """Draw the red rectangle for current position."""
        x_um, y_um, z_um = self.event_dispatcher.stage_setpoint

        # Convert top-left corner to pixel coordinates
        x1_px, y1_px = self._um_to_pixels(x_um, y_um)

        # Get pattern dimensions in pixels
        w_px, h_px = self._um_size_to_pixels(self.pattern_w, self.pattern_h)

        # Calculate bottom-right corner
        x2_px = x1_px + w_px
        y2_px = y1_px + h_px

        y1_px = self.canvas_size - y1_px
        y2_px = self.canvas_size - y2_px

        marker = self.canvas.create_rectangle(
            x1_px,
            y1_px,
            x2_px,
            y2_px,
            fill='',  # No fill
            outline='#E86E7F',  # Red outline
            width=2,
        )

        return marker  # marker ID

    def _load_patterns_from_chip(self):
        """Load all exposed pattern coordinates from the chip
        into the self.pattern_markers list"""

        self.pattern_markers.clear()

        chip = self.event_dispatcher.chip
        for layer in chip.layers:
            for exposure in layer.exposures:
                if (
                    not exposure.aborted and exposure.coords is not None
                ):  # Only include successful exposures
                    x, y, z = exposure.coords
                    self.pattern_markers.append((x, y))

    def _redraw_all(self):
        """Redraw all exposed patterns from the chip"""
        # Clear existing pattern markers
        self.canvas.delete("all")

        # Draw all exposures from all layers
        for x_um, y_um in self.pattern_markers:
            self._draw_pattern_marker(x_um, y_um)

        self._draw_current_position()

    def _on_position_changed(self):
        self._redraw_all()  # TODO: only update current_position?

    def _on_pattern_exposed(self):
        """Get the most recent exposure from current layer"""
        chip = self.event_dispatcher.chip
        if chip.layers and chip.layers[-1].exposures:
            latest_exposure = chip.layers[-1].exposures[-1]
            if not latest_exposure.aborted:
                x, y, z = latest_exposure.coords
                self.pattern_markers.append((x, y))
                self._redraw_all()

    def _on_chip_changed(self):
        """Handle chip changes (load, new chip, etc.) - reload and redraw"""
        self._load_patterns_from_chip()
        self._redraw_all()

    def resize(self, canvas_size: int):
        self.canvas_size = canvas_size
        self.canvas.config(width=canvas_size, height=canvas_size)
        self._redraw_all()


class LithographerGui:
    root: Tk
    event_dispatcher: EventDispatcher

    def __init__(self, config: LithographerConfig):
        self.root = Tk()

        # Scale all fixed-pixel UI elements to the actual screen size.
        # Cap at 1.0 so we don't upscale on large monitors; floor at 0.5 to stay usable.
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        ui_scale = max(0.5, min(screen_w / 1920, screen_h / 1080, 1.0))

        # All three main rows share vertical space equally so they all shrink/grow together.
        # Row 1 (progress bar) stays fixed since it's tiny.
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.minsize(900, 640)

        self.event_dispatcher = EventDispatcher(
            config.stage,
            TkProjector(self.root),
            self.root,
            config.camera,
            config.red_exposure,
            config.uv_exposure,
            dlpc=config.dlpc,
        )
        self.event_dispatcher.initialize_alignment(config)

        self.shown_image = ShownImage.CLEAR

        self.top_panel = ttk.Frame(self.root)
        self.top_panel.grid(row=0, column=0, sticky='nsew')

        # Map (top)
        self.map = MapFrame(self.top_panel, self.event_dispatcher, canvas_size=int(350 * ui_scale))
        self.map.frame.grid(row=0, column=0, padx=5, pady=5)
        # Camera frame (top)
        self.camera = CameraFrame(
            self.top_panel, self.event_dispatcher, config.camera, config.camera_scale
        )
        self.camera.frame.grid(row=0, column=1, padx=5, pady=5)

        # Projector display (top)
        self.projector_display = ProjectorDisplayFrame(
            self.top_panel,
            self.event_dispatcher,
            display_size=(int(320 * ui_scale), int(180 * ui_scale)),
        )
        self.projector_display.frame.grid(row=0, column=2, padx=5, pady=5)

        # uniform="top" forces all 3 columns to always share width equally
        self.top_panel.grid_columnconfigure(0, weight=1, uniform="top")
        self.top_panel.grid_columnconfigure(1, weight=1, uniform="top")
        self.top_panel.grid_columnconfigure(2, weight=1, uniform="top")

        # Progress bar
        self.pattern_progress = Progressbar(self.root, orient="horizontal", mode="determinate")
        self.pattern_progress.grid(row=1, column=0, sticky="ew")

        # Main tab interface (replaces middle_panel)
        self.mode_select_frame = ModeSelectFrame(
            self.root, self.event_dispatcher, ui_scale=ui_scale
        )
        self.mode_select_frame.notebook.grid(row=2, column=0, sticky="nsew")

        # Bottom panel (chip log and image adjustment and tiling)
        self.bottom_panel = ttk.Frame(self.root)
        self.bottom_panel.grid(row=3, column=0, sticky="ew")

        # Chip management
        self.chip_frame = ChipFrame(self.bottom_panel, self.event_dispatcher)
        self.chip_frame.frame.grid(row=0, column=0, sticky="nsew")

        # Image adjustment controls
        self.image_adjust_frame = ImageAdjustFrame(self.bottom_panel, self.event_dispatcher)
        self.image_adjust_frame.frame.grid(row=0, column=1, sticky="nsew")

        # Global settings
        self.global_settings_frame = GlobalSettingsFrame(
            self.bottom_panel, self.event_dispatcher, config.alignment.enabled
        )
        self.global_settings_frame.frame.grid(row=0, column=2, sticky="nsew")

        # Tiling controls
        self.tiling_frame = TilingFrame(self.bottom_panel, self.event_dispatcher)
        self.tiling_frame.frame.grid(row=0, column=3, sticky="nsew")

        # uniform="bottom" forces all 4 columns to always share width equally
        for i in range(4):
            self.bottom_panel.grid_columnconfigure(i, weight=1, uniform="bottom")

        # Legacy references for compatibility (if needed elsewhere in code)
        self.exposure_frame = self.mode_select_frame.uv_mode_frame.exposure_frame
        self.patterning_frame = self.mode_select_frame.uv_mode_frame.patterning_frame

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())

        self._pending_resize: Optional[str] = None
        self.root.bind('<Configure>', self._on_resize)

        # Things that have to after the main loop begins
        def on_start():
            self.camera.start()
            self.event_dispatcher.enter_red_mode(
                mode_switch_autofocus=False
            )  # ensure exposure settings are correctly set
            self.event_dispatcher.query_config()
            if self.event_dispatcher.hardware.stage.has_homing():
                self.event_dispatcher.home_stage()

            self.event_dispatcher.stage_setpoint = (
                self.event_dispatcher.hardware.stage.get_position()
            )
            print(f"Current GUI Location: {self.event_dispatcher.stage_setpoint}")

            messagebox.showinfo(
                message="BEFORE CONTINUING: Ensure that you move the projector window to the correct display! Click on the fullscreen, completely black window, then press Windows Key + Shift + Left Arrow until it no longer is visible!"
            )

        self.root.after(0, on_start)

    def _on_resize(self, event):
        if event.widget is not self.root:
            return
        if self._pending_resize is not None:
            self.root.after_cancel(self._pending_resize)
        self._pending_resize = self.root.after(
            150, lambda: self._apply_resize(event.width, event.height)
        )

    def _apply_resize(self, window_w: int, window_h: int):
        self._pending_resize = None
        ui_scale = max(0.3, min(window_w / 1920, window_h / 1080, 1.0))

        self.map.resize(int(350 * ui_scale))
        self.projector_display.resize((int(320 * ui_scale), int(180 * ui_scale)))
        self.camera.resize(ui_scale)

        red_stage = self.mode_select_frame.red_mode_frame.stage_position_frame
        uv_stage = self.mode_select_frame.uv_mode_frame.stage_position_frame
        red_stage.resize(ui_scale)
        uv_stage.resize(ui_scale)

    def cleanup(self):
        print("Patterning GUI closed.")
        if self.event_dispatcher.dlpc is not None:
            try:
                self.event_dispatcher.dlpc.set_illumination_enable(0)
                self.event_dispatcher.dlpc.close()
            except Exception as e:
                print(f"DLPC cleanup error: {e}")
        self.root.destroy()
        self.camera.cleanup()
        # if RUN_WITH_STAGE:
        # serial_port.close()


def main():
    # Open a file selector window
    config_win = tkinter.Tk()
    config_win.withdraw()
    ftypes = [('all files', '.*'), ('toml files', '.toml')]

    # Window to prompt user to select a toml configuration file
    # Note: this assumes that the current working directory has all the possible toml files
    config_path = filedialog.askopenfilename(
        title="Select a config file (default is default.toml)", filetypes=ftypes
    )
    print(f"Selected configuration: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
    except FileNotFoundError:
        print("config.toml does not exist, copying settings from default.toml")
        shutil.copy("default.toml", config_path)
        with open(config_path, "r") as f:
            config = toml.load(f)

    # STAGE CONFIG
    config_win.destroy()
    stage_config = config["stage"]
    if stage_config["enabled"]:
        if stage_config.get("type") == "omm":
            stage = OMMStage(stage_config["autofocus"], stage_config["omm"]["z-max"])
            stage.connect(stage_config["port"], stage_config["baud-rate"])
        else:
            serial_port = serial.Serial(stage_config["port"], stage_config["baud-rate"])
            print(f"Using serial port {serial_port.name}")

            # default features to False if they aren't specified -> supports legacy config.toml files
            if stage_config.get("tiling", 0) == 0:
                stage_config['tiling'] = False
            if stage_config.get('autofocus', 0) == 0:
                stage_config['autofocus'] = 0
            stage = GrblStage(
                serial_port,
                stage_config["homing"],
                stage_config["tiling"],
                stage_config["autofocus"],
            )
    else:
        stage = StageController()

    # CAMERA CONFIG

    camera_config = config["camera"]

    if camera_config["type"] == "webcam":
        try:
            index = int(camera_config["index"])
        except Exception:
            index = 0
        camera = Webcam(index)
    elif camera_config["type"] == "flir":
        import importlib

        flir = importlib.import_module("camera.flir.flir_camera")  # type: ignore[unresolved-import]
        camera = flir.FlirCamera()
    elif camera_config["type"] in ("basler", "pylon"):
        from camera.pylon import BaslerPylon

        try:
            index = int(camera_config["index"])
        except Exception:
            index = 0
        camera = BaslerPylon(index)
    elif camera_config["type"] == "none":
        camera = None
    else:
        print(f"config.toml specifies invalid camera type {camera_config['type']}")
        return 1

    camera_scale = float(camera_config.get("gui-scale", 1.0))
    red_exposure = float(camera_config.get("red-exposure", DEFAULT_RED_EXPOSURE))
    uv_exposure = float(camera_config.get("uv-exposure", DEFAULT_UV_EXPOSURE))

    # ALIGNMENT CONFIG

    alignment_config = config["alignment"]
    alignment_enabled = alignment_config.get("enabled", False)
    alignment_model = alignment_config.get("model_path", "ckpts/best.pt")

    # Get alignment marker reference coordinates with defaults
    right_marker_x = float(alignment_config.get("right_marker_x", 1820.0))
    left_marker_x = float(alignment_config.get("left_marker_x", 280.0))
    top_marker_y = float(alignment_config.get("top_marker_y", 269.0))
    bottom_marker_y = float(alignment_config.get("bottom_marker_y", 1075.0))

    # Get scaling factors with defaults
    x_scale_factor = float(alignment_config.get("x_scale_factor", -1100))
    y_scale_factor = float(alignment_config.get("y_scale_factor", 800))

    alignment_config = AlignmentConfig(
        enabled=alignment_enabled,
        model_path=alignment_model,
        right_marker_x=right_marker_x,
        left_marker_x=left_marker_x,
        top_marker_y=top_marker_y,
        bottom_marker_y=bottom_marker_y,
        x_scale_factor=x_scale_factor,
        y_scale_factor=y_scale_factor,
    )

    # DLPC CONFIG
    dlpc = None
    projector_config = config.get("projector", {})
    if projector_config.get("dlpc_enabled", False):
        try:
            from dlpc import connect as dlpc_connect

            pid = projector_config.get("dlpc_pid", None)
            dlpc = dlpc_connect(pid)
            if dlpc is None:
                raise RuntimeError(
                    "No TI USB projector controller was found. Check that the "
                    "projector USB connection is enabled and that the correct "
                    "device is connected."
                )

            print("Connected to DLPC6540 projector controller")
            SAFE_DEFAULT_LEVEL = 150
            SAFE_MAX_LEVEL = 400
            uv_level = projector_config.get("uv_led_drive_level", SAFE_DEFAULT_LEVEL)
            try:
                uv_level = int(uv_level)
                if not 0 <= uv_level <= SAFE_MAX_LEVEL:
                    uv_level = SAFE_DEFAULT_LEVEL
            except (TypeError, ValueError):
                uv_level = SAFE_DEFAULT_LEVEL
            try:
                dlpc.set_led_drive_level(SAFE_DEFAULT_LEVEL, SAFE_DEFAULT_LEVEL, uv_level)
            except Exception as e:
                raise RuntimeError(f"Failed to configure the DLPC LED drive levels: {e}") from e
            print(
                f"Set LED drive levels: red={SAFE_DEFAULT_LEVEL}, "
                f"green={SAFE_DEFAULT_LEVEL}, blue(UV)={uv_level}"
            )
        except Exception as e:
            print(f"ERROR: DLPC USB startup failed: {e}")
            if dlpc is not None:
                try:
                    dlpc.close()
                except Exception as close_error:
                    print(f"ERROR: failed to close DLPC after startup failure: {close_error}")
            raise

    assert camera is not None, "Camera must be initialized"

    lithographer_config = LithographerConfig(
        stage,
        camera,
        camera_scale,
        red_exposure,
        uv_exposure,
        alignment_config,
        dlpc,
    )

    lithographer = LithographerGui(lithographer_config)
    lithographer.root.mainloop()


if __name__ == "__main__":
    main()

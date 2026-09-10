import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from camera import CameraModule
from hardware import ImageProcessSettings, Lithographer, ProcessedImage
from stage_control import StageController

from .alignment import load_alignment_model
from .events import Event, EventBus, MovementLock, RedFocusSource, ShownImage
from .focus import execute_autofocus
from .lithography import Chip, ChipLayer, ExposureLog, LithographerConfig


class StepperEngine:
    """Core headless process controller for the lithography stepper.

    Encapsulates all stage motion, illumination/DLP synchronization, image
    processing, autofocus routines, exposure sequencing, and data logging
    without dependencies on any GUI toolkit.
    """

    def __init__(
        self,
        stage: StageController,
        projector: Any,
        camera: Optional[CameraModule] = None,
        red_exposure: float = 4167.0,
        uv_exposure: float = 25000.0,
        dlpc: Optional[Any] = None,
        delay_func: Optional[Callable[[float], None]] = None,
        warning_callback: Optional[Callable[[str], None]] = None,
    ):
        self.hardware = Lithographer(stage, projector)
        self.camera = camera
        self.dlpc = dlpc
        self.delay_func = delay_func or time.sleep
        self.warning_callback = warning_callback or self._default_warning

        # Detection model and config
        self.model = None
        self.config: Optional[LithographerConfig] = None

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
        self.camera_image: Optional[np.ndarray] = None

        # Image settings
        self.image_adjust_position = (0.0, 0.0, 0.0)
        self.border_size = 0.0
        self.posterize_strength: Optional[int] = None
        self.red_focus_source = RedFocusSource.IMAGE
        self.use_solid_red = False

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
        self.exposure_time = 8000  # ms
        self.patterning_progress = 0.0
        self.exposure_progress = 0.0
        self.red_exposure_time = red_exposure
        self.uv_exposure_time = uv_exposure

        # History and logging
        self.exposure_history: List[ExposureLog] = []
        self.chip = Chip([ChipLayer([])])

        # Snapshot settings
        self.auto_snapshot_on_uv = True
        self.snapshot_directory = Path("stepper_captures")
        self.snapshot_directory.mkdir(exist_ok=True)

        # Event management
        self.events = EventBus()
        self.events.add_listener(Event.SHOWN_IMAGE_CHANGED, lambda: self._update_projector())

    def _default_warning(self, msg: str):
        print(f"Warning: {msg}")

    def create_warning(self, msg: str):
        self.warning_callback(msg)

    def sleep(self, t: float):
        self.delay_func(t)

    # -------------------------------------------------------------------------
    # Chip Management
    # -------------------------------------------------------------------------
    def load_chip(self, path: str):
        print(f"Loading chip at {path!r}")
        with open(path, "r") as f:
            d = json.load(f)
        self.chip = Chip.from_disk(d)
        self.events.emit(Event.CHIP_CHANGED)

    def new_chip(self):
        self.chip = Chip([ChipLayer([])])
        self.events.emit(Event.CHIP_CHANGED)

    def add_chip_layer(self):
        self.chip.layers.append(ChipLayer([]))
        self.events.emit(Event.CHIP_CHANGED)

    def save_chip(self, path: str):
        with open(path, "w") as f:
            json.dump(self.chip.to_disk(), f)

    def delete_chip_exposure(self, layer: int, ex: int):
        self.chip.layers[layer].exposures.pop(ex)
        print(f"Deleted exposure {layer} {ex}")
        self.events.emit(Event.CHIP_CHANGED)

    # -------------------------------------------------------------------------
    # Projector & Illumination
    # -------------------------------------------------------------------------
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
            return 0b111
        r, g, b = settings.color_channels
        return (0b001 if r else 0) | (0b010 if g else 0) | (0b100 if b else 0)

    def _sync_led_enable(self, mask: int) -> None:
        """Sync the Image channel mask to the DLPC chip's LED enable mask."""
        if self.dlpc is None:
            return
        try:
            self.dlpc.set_illumination_enable(mask)
        except Exception as e:
            print(f"DLPC LED sync failed: {e}")

    def set_shown_image(self, shown_image: ShownImage):
        print(f"set_shown_image({shown_image})")
        self.shown_image = shown_image
        self.events.emit(Event.SHOWN_IMAGE_CHANGED)

    # -------------------------------------------------------------------------
    # Image Adjustments & Pipelines
    # -------------------------------------------------------------------------
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

        if self.red_focus_source in (RedFocusSource.PATTERN, RedFocusSource.INV_PATTERN):
            self._refresh_red_focus()

        self.events.emit(Event.PATTERN_IMAGE_CHANGED)

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
            self.events.emit(Event.SHOWN_IMAGE_CHANGED)

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
            self.events.emit(Event.SHOWN_IMAGE_CHANGED)

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

    def set_use_solid_red(self, use: bool):
        self.use_solid_red = use
        self.set_shown_image(ShownImage.RED_FOCUS)
        self._refresh_red_focus()

    def set_latest_image(self, camera_image: Optional[np.ndarray]):
        self.camera_image = camera_image

    def set_snapshot_directory(self, directory: Path):
        self.snapshot_directory = directory
        self.snapshot_directory.mkdir(exist_ok=True)

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

    def set_image_position(self, x: float, y: float, t: float):
        self.image_adjust_position = (x, y, t)
        self._refresh_red_focus()
        self._refresh_uv_focus()
        self._refresh_pattern()
        self.events.emit(Event.IMAGE_ADJUST_CHANGED)

    @property
    def image_position(self) -> Tuple[float, float, float]:
        return self.image_adjust_position

    # -------------------------------------------------------------------------
    # Stage Motion & Bounds
    # -------------------------------------------------------------------------
    def _check_bounds(self, set_point: Tuple[float, float, float]) -> Tuple[bool, Optional[str]]:
        bounds = self.hardware.stage.get_bounds()
        if bounds is None:
            return True, None

        axes = [("x", 0), ("y", 1), ("z", 2)]
        for name, i in axes:
            lo, hi = bounds[name]
            val = set_point[i]
            if not (lo <= val <= hi):
                return False, (
                    f"Moving {name.upper()} to {val} prohibited. Boundaries are [{lo}, {hi}]"
                )
        return True, None

    def move_absolute(self, coords: dict[str, float]) -> bool:
        if self.hardware.stage.has_homing():
            print(f"Moving to position: {coords}")
            print(
                f"Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}"
            )

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
            self.events.emit(Event.STAGE_POSITION_CHANGED)
            return True
        except RuntimeError as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            return False
        except Exception as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            self.stage_setpoint = self.hardware.stage.get_position()
            self.events.emit(Event.STAGE_POSITION_CHANGED)
            return False

    def move_relative(self, coords: dict[str, float]) -> bool:
        if self.hardware.stage.has_homing():
            print(f"Moving: {coords}")
            print(
                f"Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}"
            )

        x = self.stage_setpoint[0] + coords.get("x", 0)
        y = self.stage_setpoint[1] + coords.get("y", 0)
        z = self.stage_setpoint[2] + coords.get("z", 0)
        set_point = (x, y, z)

        if self.hardware.stage.has_homing():
            ok, msg = self._check_bounds(set_point)
            if not ok:
                self.create_warning(msg)
                return False

        try:
            self.hardware.stage.move_relative(coords)
            self.stage_setpoint = set_point
            self.events.emit(Event.STAGE_POSITION_CHANGED)
            return True
        except RuntimeError as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            return False
        except Exception as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            self.stage_setpoint = self.hardware.stage.get_position()
            self.events.emit(Event.STAGE_POSITION_CHANGED)
            return False

    def home_stage(self):
        self.hardware.stage.home()
        self.hardware.stage.set_on_start_location()
        print(f"Post Homing Location: {self.hardware.stage.get_on_start_location()}")
        print("Homing Complete.")
        self.events.emit(Event.STAGE_POSITION_CHANGED)

    def query_config(self):
        self.hardware.stage.get_position()
        print("Query Config Complete.")

    @property
    def movement_lock(self) -> MovementLock:
        if self.patterning_busy or self.autofocus_busy:
            return MovementLock.LOCKED
        return MovementLock.UNLOCKED

    def in_uv(self) -> bool:
        return self.shown_image in (ShownImage.PATTERN, ShownImage.UV_FOCUS)

    # -------------------------------------------------------------------------
    # Mode Transitions
    # -------------------------------------------------------------------------
    def enter_red_mode(self, mode_switch_autofocus=True):
        print("enter_red_mode")
        self.set_shown_image(ShownImage.RED_FOCUS)
        if self.camera:
            self.camera.setExposureTime(self.red_exposure_time)
        if mode_switch_autofocus and self.autofocus_on_mode_switch:
            self.autofocus(blue_only=False)
        self.events.emit(Event.MOVEMENT_LOCK_CHANGED)

    def enter_uv_mode(self, mode_switch_autofocus=True):
        if self.auto_snapshot_on_uv:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = self.snapshot_directory / f"uv_mode_entry_{timestamp}.png"
            self.events.emit(Event.SNAPSHOT, str(filename))

        if self.camera:
            self.camera.setExposureTime(self.uv_exposure_time)

        self.set_shown_image(ShownImage.CLEAR)

        if mode_switch_autofocus and self.autofocus_on_mode_switch:
            self.sleep(2.0)
            self.autofocus(blue_only=True)

        self.events.emit(Event.MOVEMENT_LOCK_CHANGED)

    # -------------------------------------------------------------------------
    # Autofocus
    # -------------------------------------------------------------------------
    def set_autofocus_busy(self, busy: bool):
        self.autofocus_busy = busy
        self.events.emit(Event.MOVEMENT_LOCK_CHANGED)

    def autofocus(self, blue_only: bool = False, log: bool = False):
        if not self.camera:
            print("No camera connected, skipping autofocus")
            return

        if self.first_autofocus:
            self.first_autofocus = False
            return

        if self.autofocus_busy:
            print("Skipping nested autofocus!")
            return

        self.set_autofocus_busy(True)
        try:
            execute_autofocus(
                has_homing=self.hardware.stage.has_homing(),
                get_autofocus_base=lambda: self.hardware.stage.get_autofocus(),
                get_current_z=lambda: self.stage_setpoint[2],
                move_absolute=self.move_absolute,
                move_relative=self.move_relative,
                get_camera_image=lambda: self.camera_image,
                delay=self.sleep,
                on_warning=self.create_warning,
                blue_only=blue_only,
                log=log,
            )
        finally:
            self.set_autofocus_busy(False)

    # -------------------------------------------------------------------------
    # Exposure & Patterning
    # -------------------------------------------------------------------------
    def set_patterning_busy(self, busy: bool):
        self.patterning_busy = busy
        self.events.emit(Event.MOVEMENT_LOCK_CHANGED)
        self.events.emit(Event.PATTERNING_BUSY_CHANGED)

    def set_progress(self, pattern_progress: float, exposure_progress: float):
        self.patterning_progress = pattern_progress
        self.exposure_progress = exposure_progress
        self.events.emit(Event.EXPOSURE_PATTERN_PROGRESS_CHANGED)

    def abort_patterning(self):
        self.should_abort = True
        print("Aborting patterning")

    def begin_patterning(self):
        print("Patterning at ", self.stage_setpoint)
        duration = self.exposure_time
        print(f"Patterning 1 tiles for {duration}ms\nTotal time: {str(round(duration / 1000))}s")

        self.set_patterning_busy(True)
        self.set_shown_image(ShownImage.PATTERN)
        end_time = time.time() + duration / 1000.0

        while time.time() < end_time:
            progress = 1.0 - ((end_time - time.time()) * 1000.0 / duration)
            self.set_progress(0.0, progress)
            self.sleep(0.02)
            if self.should_abort:
                break

        self.set_shown_image(ShownImage.CLEAR)
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

        self.events.emit(Event.CHIP_CHANGED)
        self.set_patterning_busy(False)

        if self.should_abort:
            print("Patterning aborted")
            self.should_abort = False

    def initialize_alignment(self, config: LithographerConfig):
        self.config = config
        self.realtime_detection = config.alignment.enabled
        self.model = None
        if config.alignment.enabled:
            self.model = load_alignment_model(config.alignment.model_path)

import json
import os
import queue
import shutil
import time
import toml

import tkinter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import partial
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.ttk import Progressbar
from typing import Callable, List, Optional

import cv2
import numpy as np
import serial
from PIL import Image, ImageOps
from ultralytics import YOLO

from camera.camera_module import CameraModule
from camera.webcam import Webcam
from hardware import ImageProcessSettings, Lithographer, ProcessedImage
from lib.gui import IntEntry, Thumbnail
from lib.img import image_to_tk_image
from projector import TkProjector
from stage_control.grbl_stage import GrblStage
from stage_control.stage_controller import StageController

# TODO: Don't hardcode
THUMBNAIL_SIZE: tuple[int, int] = (160, 90)
DEFAULT_RED_EXPOSURE: float = 4167.0
DEFAULT_UV_EXPOSURE: float = 25000.0

def compute_focus_score(camera_image, blue_only, save=False):
    camera_image = camera_image.copy()
    camera_image[:, :, 1] = 0  # green should never be used for focus
    #if blue_only:
    #   camera_image[:, :, 0] = 0  # disable red
    img = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    mean = np.sum(img) / (img.shape[0] * img.shape[1])
    img_lapl = (np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1))) / mean
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
    patterning_progress: float # ranges from 0.0 to 1.0
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
    ):
        # Hardware components
        self.hardware = Lithographer(stage, proj)
        self.camera = camera
        self.root = root

        # Detection model
        self.model = None

        # Image processing objects
        self.red_focus = ProcessedImage()
        self.uv_focus = ProcessedImage()
        self.pattern = ProcessedImage()

        # Source images
        self.pattern_image = Image.new("RGB", (1, 1), "black")
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
        else:
            self.hardware.projector.show(img)

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

    def move_absolute(self, coords: dict[str, float]):
        self.hardware.stage.move_to(coords)
        x = coords.get("x", self.stage_setpoint[0])
        y = coords.get("y", self.stage_setpoint[1])
        z = coords.get("z", self.stage_setpoint[2])
        self.stage_setpoint = (x, y, z)
        self.on_event(Event.STAGE_POSITION_CHANGED)

    def move_relative(self, coords: dict[str, float]):
        x = coords.get("x", 0) + self.stage_setpoint[0]
        y = coords.get("y", 0) + self.stage_setpoint[1]
        z = coords.get("z", 0) + self.stage_setpoint[2]
        self.stage_setpoint = (x, y, z)
        self.hardware.stage.move_to({k: self.stage_setpoint[i] for k, i in (("x", 0), ("y", 1), ("z", 2))})
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
    
    def set_latest_image(self, camera_image):
        self.camera_image = camera_image

    def set_autofocus_busy(self, busy):
        self.autofocus_busy = busy
        self.on_event(Event.MOVEMENT_LOCK_CHANGED)

    def abort_patterning(self):
        self.should_abort = True
        print("Aborting patterning")

    def in_uv(self):
        return self.shown_image in (ShownImage.PATTERN, ShownImage.UV_FOCUS)

    def home_stage(self):
        self.hardware.stage.home()
        self.non_blocking_delay(1.0)
        while True:
            self.non_blocking_delay(1.0)
            idle, pos = self.hardware.stage._query_state()
            if idle:
                break

        self.stage_setpoint = (pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0)
        print(f"Homed stage to {self.stage_setpoint}")
        self.on_event(Event.STAGE_POSITION_CHANGED)

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
        elif (self.shown_image == ShownImage.UV_FOCUS or self.shown_image == ShownImage.PATTERN):
            return MovementLock.XY_LOCKED
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
        img = self.pattern.processed()

        self.set_patterning_busy(True)
        self.hardware.projector.show(img)
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

    def non_blocking_delay(self, t: float):
        start = time.time()
        while time.time() - start < t:
            self.root.update()

    def enter_red_mode(self, mode_switch_autofocus=True):
        print("enter_red_mode")
        self.set_shown_image(ShownImage.RED_FOCUS)
        self.camera.setExposureTime(self.red_exposure_time)
        if mode_switch_autofocus and self.autofocus_on_mode_switch:
            self.autofocus(blue_only=False)
        self.on_event(Event.MOVEMENT_LOCK_CHANGED)

    def enter_uv_mode(self, mode_switch_autofocus=True):
        if self.auto_snapshot_on_uv:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = self.snapshot_directory / f"uv_mode_entry_{timestamp}.png"
            self.on_event(Event.SNAPSHOT, str(filename))

        self.camera.setExposureTime(self.uv_exposure_time)
        if (
            mode_switch_autofocus
            and not self.autofocus_busy
            and self.autofocus_on_mode_switch
        ):
            # UV mode usually needs about -70 to be in focus compared to red mode
            #self.move_relative({"z": -85.0})
            pass

        self.set_shown_image(ShownImage.UV_FOCUS)

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
            

        print("Starting autofocus")

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

        self.button = ttk.Button(self.frame, text="Take Snapshot", command=on_snapshot_button, state=state)
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
        self.gui_camera_scale = camera_scale

        self.focus_score_label = ttk.Label(self.frame, text="Focus Score: N/A")
        self.focus_score_label.grid(row=0, column=1)

        self.snapshot = SnapshotFrame(self.frame, c is not None, event_dispatcher)
        self.snapshot.frame.grid(row=1, column=0)

        self.event_dispatcher = event_dispatcher

        self.snapshots_pending = queue.Queue()
        self.event_dispatcher.add_event_listener(Event.SNAPSHOT, lambda filename: self.snapshots_pending.put(filename))

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
            image, dimensions, format = self.pending_frame
            red_score = compute_focus_score(image, blue_only=False)
            blue_score = compute_focus_score(image, blue_only=True)
            self.focus_score_label.configure(text=f"Focus Score: {red_score:.3e} {blue_score:.3e}")

            try:
                filename = self.snapshots_pending.get_nowait()
                print(f"Saving image {filename}")
                compute_focus_score(image, blue_only=False, save='focusred.png')
                compute_focus_score(image, blue_only=False, save='focusblue.png')
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(filename, img)
            except queue.Empty:
                pass

            self.gui_camera_preview(image, dimensions)
        finally:
            self.event_dispatcher.root.after(66, lambda: self._on_new_frame())

    def start(self):
        if not self.camera:
            print("No camera available")
            return

        # self.event_dispatcher.root.bind('<<NewFrame>>', lambda x: self._on_new_frame(x))

        def cameraCallback(image, dimensions, format):
            self.pending_frame = (image, dimensions, format)
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

    def gui_camera_preview(self, camera_image, dimensions):
        model = self.event_dispatcher.model
        if model and self.event_dispatcher.realtime_detection:
            _, camera_image = detect_alignment_markers(model, camera_image, draw_rectangle=True)
        self.event_dispatcher.set_latest_image(camera_image)
        resized_img = cv2.resize(camera_image, (0, 0), fx=self.gui_camera_scale, fy=self.gui_camera_scale)
        gui_img = image_to_tk_image(Image.fromarray(resized_img, mode="RGB"))
        self.label.configure(image=gui_img)  # type:ignore
        self.gui_img = gui_img


class StagePositionFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)

        self.position_intputs = []
        self.step_size_intputs = []

        self.xy_widgets = []
        self.z_widgets = []

        # Absolute

        self.absolute_frame = ttk.LabelFrame(self.frame, text="Stage Position")
        self.absolute_frame.grid(row=0, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "z")):
            self.position_intputs.append(IntEntry(parent=self.absolute_frame, default=0))
            self.position_intputs[-1].widget.grid(row=0, column=i)

        def callback_set():
            x, y, z = self._position()
            event_dispatcher.move_absolute({"x": x, "y": y, "z": z})

        self.set_position_button = ttk.Button(self.absolute_frame, text="Set Stage Position", command=callback_set)
        self.set_position_button.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Relative

        self.relative_frame = ttk.LabelFrame(self.frame, text="Adjustment")
        self.relative_frame.grid(row=1, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "z")):
            self.step_size_intputs.append(
                IntEntry(
                    parent=self.relative_frame,
                    default=10,
                    min_value=-1000,
                    max_value=1000,
                )
            )
            self.step_size_intputs[-1].widget.grid(row=3, column=i)

            def callback_pos(index, c):
                event_dispatcher.move_relative({c: self.step_sizes()[index]})

            def callback_neg(index, c):
                event_dispatcher.move_relative({c: -self.step_sizes()[index]})

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

            if i in (0, 1):
                self.xy_widgets.append(coord_inc_button)
                self.xy_widgets.append(coord_dec_button)
                self.xy_widgets.append(self.position_intputs[i].widget)
                self.xy_widgets.append(self.step_size_intputs[i].widget)
            else:
                self.z_widgets.append(coord_inc_button)
                self.z_widgets.append(coord_dec_button)
                self.z_widgets.append(self.position_intputs[i].widget)
                self.z_widgets.append(self.step_size_intputs[i].widget)

        ttk.Label(
            self.relative_frame, text="Step Size (microns)", anchor="center"
        ).grid(row=2, column=0, columnspan=3, sticky="ew")

        self.all_widgets = self.xy_widgets + self.z_widgets + [self.set_position_button]

        def on_lock_change():
            lock = event_dispatcher.movement_lock
            match lock:
                case MovementLock.UNLOCKED:
                    for w in self.all_widgets:
                        w.configure(state="normal")
                case MovementLock.XY_LOCKED:
                    for w in self.xy_widgets:
                        w.configure(state="disabled")
                    for w in self.z_widgets:
                        w.configure(state="normal")
                    self.set_position_button.configure(state="normal")
                case MovementLock.LOCKED:
                    for w in self.all_widgets:
                        w.configure(state="disabled")

        event_dispatcher.add_event_listener(Event.MOVEMENT_LOCK_CHANGED, on_lock_change)

        def on_position_change():
            pos = event_dispatcher.stage_setpoint
            for i in range(3):
                self.position_intputs[i].set(pos[i])

        event_dispatcher.add_event_listener(Event.STAGE_POSITION_CHANGED, on_position_change)

        self.shortcut_frame = ttk.LabelFrame(self.frame, text="Shortcuts")
        self.shortcut_frame.grid(row=2, column=0)

        def on_chip_origin():
            event_dispatcher.move_absolute({ "x": -14500.0, "y": -13500.0, "z": -13844.0 })
        def on_chip_unload():
            event_dispatcher.move_absolute({ "x": -14500.0, "y": -14500.0, "z": -14500.0 })


        btn_state = "normal" if event_dispatcher.hardware.stage.has_homing() else "disabled"
        self.chip_origin_button = ttk.Button(self.shortcut_frame, text="Chip origin", command=on_chip_origin, state=btn_state)
        self.chip_origin_button.grid(row=0, column=0)
        self.chip_unload_button = ttk.Button(self.shortcut_frame, text="Load/unload", command=on_chip_unload, state=btn_state)
        self.chip_unload_button.grid(row=0, column=1)


    def _position(self) -> tuple[int, int, int]:
        return tuple(intput.get() for intput in self.position_intputs)

    def _set_position(self, pos: tuple[int, int, int]):
        for i in range(3):
            self.position_intputs[i].set(pos[i])

    def step_sizes(self) -> tuple[int, int, int]:
        return tuple(intput.get() for intput in self.step_size_intputs)


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
            self.position_intputs.append(
                IntEntry(parent=self.absolute_frame, default=0)
            )
            self.position_intputs[-1].widget.grid(row=0, column=i)

        def callback_set():
            x, y, t = self._position()
            event_dispatcher.set_image_position(x, y, t)

        self.set_position_button = ttk.Button(self.absolute_frame, text="Set Image Position", command=callback_set)
        self.set_position_button.grid(row=1, column=0, columnspan=3, sticky="ew")

        # Relative
        self.relative_frame = ttk.LabelFrame(self.frame, text="Adjustment")
        self.relative_frame.grid(row=1, column=0)

        for i, coord in ((0, "x"), (1, "y"), (2, "ϴ")):
            self.step_size_intputs.append(
                IntEntry(
                    parent=self.relative_frame,
                    default=10,
                    min_value=-1000,
                    max_value=1000,
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
                self.position_intputs[i].set(pos[i])

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
        return tuple(intput.get() for intput in self.position_intputs)

    def _set_position(self, pos: tuple[int, int, int]):
        for i in range(3):
            self.position_intputs[i].set(pos[i])

    def step_sizes(self) -> tuple[int, int, int]:
        return tuple(intput.get() for intput in self.step_size_intputs)


class ImageSelectFrame:
    def __init__(self, parent, button_text, import_command):
        self.frame = ttk.Frame(parent)

        self.thumb = Thumbnail(self.frame, THUMBNAIL_SIZE, on_import=import_command)
        self.thumb.widget.grid(row=0, column=0)

        self.label = ttk.Label(self.frame, text=button_text)
        self.label.grid(row=1, column=0)


class MultiImageSelectFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)

        def get_name(path):
          return "" if path == "" else Path(path).name

        self.pattern_frame = ImageSelectFrame(
            self.frame,
            "Pattern",
            lambda t: event_dispatcher.set_pattern_image(self.pattern_image(), self.pattern_frame.thumb.path),
        )
        self.uv_focus_frame = ImageSelectFrame(
            self.frame,
            "UV Focus",
            lambda t: event_dispatcher.set_uv_focus_image(self.uv_focus_image()),
        )

        self.pattern_frame.frame.grid(row=0, column=0)
        self.uv_focus_frame.frame.grid(row=1, column=0)

        event_dispatcher.add_event_listener(Event.SHOWN_IMAGE_CHANGED, lambda: self.highlight_button(event_dispatcher.shown_image))

    def pattern_image(self):
        return self.pattern_frame.thumb.image

    def uv_focus_image(self):
        return self.uv_focus_frame.thumb.image

    def highlight_button(self, which: ShownImage):
        # TODO:
        pass


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
            yes = messagebox.askyesno(title="Delete Exposure", message="Are you sure you want to delete the selected exposure?")
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
        self.finish_layer_button = ttk.Button(self.chip_select_frame, text="Finish Layer", command=on_finish_layer)
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
        self.tree_view_style.configure("Treeview", rowheight=50)

        self.prev_layer_view = ttk.Treeview(self.prev_layer_frame, selectmode="browse", columns=("XYZ",), height=5)
        self.prev_layer_view.grid(row=0, column=0)
        self.prev_layer_view.bind("<<TreeviewSelect>>", lambda e: on_select(e, cur=False))
        self.prev_layer_view.bind("<Double-1>", lambda e: on_double_click(cur=False))
        self.cur_layer_view = ttk.Treeview(self.cur_layer_frame, selectmode="browse", columns=("XYZ",), height=5)
        self.cur_layer_view.grid(row=0, column=0)
        self.cur_layer_view.bind("<<TreeviewSelect>>", lambda e: on_select(e, cur=True))
        self.cur_layer_view.bind("<Double-1>", lambda e: on_double_click(cur=True))

        select_frame = ttk.Frame(self.prev_layer_frame)
        select_frame.grid(row=1, column=0)

        prev_layer_select_label = ttk.Label(select_frame, text="Select previous layer:")
        prev_layer_select_label.grid(row=0, column=0)

        self.prev_layer_select_var = StringVar()
        self.prev_layer_select_var.trace_add("write", prev_layer_index_changed)
        self.prev_layer_select = ttk.Spinbox(select_frame, from_=0, to=0, textvariable=self.prev_layer_select_var)
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
            print(f"Leaving previous layer empty because select var is {self.prev_layer_select_var.get()!r}")
            return

        if len(chip.layers) < 2:
            return

        for i, ex in enumerate(chip.layers[idx].exposures):
            ex_id = f"{idx}_{i}"
            pos = f"{ex.coords[0]},{ex.coords[1]},{ex.coords[2]}"
            self.prev_layer_view.insert("", "end", ex_id, image=self._get_thumbnail(ex.path), values=(pos,))

    def refresh_cur_layer(self):
        chip = self.model.chip

        for item in self.cur_layer_view.get_children():
            self.cur_layer_view.delete(item)

        for i, ex in enumerate(chip.layers[-1].exposures):
            ex_id = f"{len(chip.layers) - 1}_{i}"
            pos = f"{ex.coords[0]},{ex.coords[1]},{ex.coords[2]}"
            self.cur_layer_view.insert("", "end", ex_id, image=self._get_thumbnail(ex.path), values=(pos,))


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
        self.posterize_scale = ttk.Scale(self.frame, variable=self.posterize_strength_var, from_=0.0, to=100.0)
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

        self.preview_tile = ttk.Label(self.frame, text="Next Pattern Tile", compound="top")  # type:ignore
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
        self.exposure_progress = Progressbar(self.frame, orient="horizontal", mode="determinate", maximum=1000)
        self.exposure_progress.grid(row=4, column=0, sticky="ew")

        self.set_image(Image.new("RGB", (1, 1)))

        def on_change_patterning_status():
            if event_dispatcher.patterning_busy:
                self.begin_patterning_button["state"] = "disabled"
                self.abort_patterning_button["state"] = "normal"
            else:
                self.begin_patterning_button["state"] = "normal"
                self.abort_patterning_button["state"] = "disabled"

        event_dispatcher.add_event_listener(Event.PATTERNING_BUSY_CHANGED, on_change_patterning_status)
        event_dispatcher.add_event_listener(Event.PATTERN_IMAGE_CHANGED, lambda: self.set_image(event_dispatcher.pattern.processed()))

        def on_progress_changed():
            self.exposure_progress["value"] = (event_dispatcher.exposure_progress * 1000.0)

        event_dispatcher.add_event_listener(Event.EXPOSURE_PATTERN_PROGRESS_CHANGED, on_progress_changed)

    def set_image(self, img: Image.Image):
        # TODO: What is the correct size?
        self.thumb_image = image_to_tk_image(img.resize(THUMBNAIL_SIZE))
        self.preview_tile.configure(image=self.thumb_image)  # type:ignore


class RedModeFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent, name="redmodeframe")

        self.red_select_var = StringVar(value="focus")
        self.red_focus_rb = ttk.Radiobutton(
            self.frame,
            variable=self.red_select_var,
            text="Red Focus",
            value=RedFocusSource.IMAGE.value,
        )
        self.solid_red_rb = ttk.Radiobutton(
            self.frame,
            variable=self.red_select_var,
            text="Solid Red",
            value=RedFocusSource.SOLID.value,
        )
        self.pattern_rb = ttk.Radiobutton(
            self.frame,
            variable=self.red_select_var,
            text="Same as Pattern",
            value=RedFocusSource.PATTERN.value,
        )
        self.inv_pattern_rb = ttk.Radiobutton(
            self.frame,
            variable=self.red_select_var,
            text="Inverse of Pattern",
            value=RedFocusSource.INV_PATTERN.value,
        )

        self.red_focus_rb.grid(row=0, column=0)
        self.solid_red_rb.grid(row=1, column=0)
        self.pattern_rb.grid(row=2, column=0)
        self.inv_pattern_rb.grid(row=3, column=0)

        self.red_focus_frame = ImageSelectFrame(
            self.frame,
            "Red Focus",
            lambda t: event_dispatcher.set_red_focus_image(self.red_focus_image()),
        )
        self.red_focus_frame.frame.grid(row=0, column=1, rowspan=4)

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
    def __init__(self, parent, event_dispatcher):
        self.frame = ttk.Frame(parent, name="uvmodeframe")
        self.exposure_frame = ExposureFrame(self.frame, event_dispatcher)
        self.exposure_frame.frame.grid(row=0, column=0)
        self.patterning_frame = PatterningFrame(self.frame, event_dispatcher)
        self.patterning_frame.frame.grid(row=0, column=1)


class ModeSelectFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.notebook = ttk.Notebook(parent)

        self.red_mode_frame = RedModeFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.red_mode_frame.frame, text="Red Mode")
        self.uv_mode_frame = UvModeFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.uv_mode_frame.frame, text="UV Mode")

        def on_tab_change():
            if self._current_tab() == "uv":
                event_dispatcher.enter_uv_mode()
            else:
                event_dispatcher.enter_red_mode()

        self.notebook.bind("<<NotebookTabChanged>>", lambda _: on_tab_change())

        # def on_tab_event(evt):
        #  self.notebook.select(1 if evt == Event.EnterUvMode else 0)

        # event_dispatcher.add_event_listener(Event.EnterRedMode, lambda: on_tab_event(Event.EnterRedMode))
        # event_dispatcher.add_event_listener(Event.EnterUvMode, lambda: on_tab_event(Event.EnterUvMode))

    def _current_tab(self):
      return "red" if "redmode" in self.notebook.select() else "uv"


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
            markers, _ = detect_alignment_markers(event_dispatcher.model, event_dispatcher.camera_image)
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
            event_dispatcher.move_relative({ 'x': dx, 'y': dy })

            print(markers)
        
        self.alignbutton = ttk.Button(
            self.frame, 
            text="Align :)", 
            command=do_align, 
            state="disabled" if event_dispatcher.model is None else "normal",
        )
        self.alignbutton.grid(row=2, column=1)

        # TODO: Fix this
        self.autofocus_button = ttk.Button(self.frame, text="Autofocus", command=lambda: event_dispatcher.autofocus(blue_only=event_dispatcher.in_uv()))
        self.autofocus_button.grid(row=2, column=0, sticky="ew")

        # Maybe this should have a scale?
        # Or, even further, maybe this should just be the same as the interface for posterization strength?
        self.border_size_var = IntVar()
        self.border_label = ttk.Label(self.frame, text="Border Size (%)")
        self.border_label.grid(row=3, column=0)
        self.border_entry = IntEntry(self.frame, var=self.border_size_var, default=0, min_value=0, max_value=100)
        self.border_entry.widget.grid(row=3, column=1, sticky="nesw")

        def on_border_size_change(*_):
            event_dispatcher.set_border_size(self.border_size_var.get())

        self.border_size_var.trace_add("write", on_border_size_change)

        self.placeholder_photo = image_to_tk_image(Image.new("RGB", THUMBNAIL_SIZE, "black"))
        self.photo = None

        self.current_image = ttk.Label(self.frame, image=self.placeholder_photo)  # type:ignore
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
                self.current_image.configure(image=self.placeholder_photo)  # type:ignore
            else:
                photo = image_to_tk_image(img.resize(THUMBNAIL_SIZE, Image.Resampling.NEAREST))
                self.current_image.configure(image=photo)  # type:ignore
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
        self.directory_entry = ttk.Entry(self.snapshot_frame, textvariable=self.directory_var, state="readonly")
        self.directory_entry.grid(row=1, column=1, sticky="ew")

        def choose_directory():
            dir_path = filedialog.askdirectory(
                initialdir=self.directory_var.get(),
                title="Select Snapshot Save Directory",
            )
            if dir_path:  # User didn't cancel
                event_dispatcher.set_snapshot_directory(Path(dir_path))
                self.directory_var.set(dir_path)

        self.choose_dir_button = ttk.Button(self.snapshot_frame, text="Choose Directory", command=choose_directory)
        self.choose_dir_button.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Configure grid weights for proper expansion
        self.snapshot_frame.columnconfigure(1, weight=1)


class ExposureHistoryFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.LabelFrame(parent, text="Exposure History")
        self.text = tkinter.Text(self.frame, width=80, height=10, wrap="none", state="disabled")
        self.text.grid(row=0, column=0)
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
        self.amount_spinbox = ttk.Spinbox(self.frame, from_=-20, to=20, textvariable=self.amount_var, width=3)
        self.amount_spinbox.grid(row=0, column=3)


class TilingFrame:
    def __init__(self, parent, model: EventDispatcher):
        self.frame = ttk.LabelFrame(parent, text="Tiling")
        self.model = model

        # TODO: Tune default offsets
        self.x_settings = OffsetAmountFrame(self.frame, "X", 1050)
        self.x_settings.frame.grid(row=0, column=0)
        self.y_settings = OffsetAmountFrame(self.frame, "Y", 900)
        self.y_settings.frame.grid(row=1, column=0)

        def on_begin():
            x_amount = int(self.x_settings.amount_var.get())
            x_offset = int(self.x_settings.offset_var.get())
            x_dir = 1 if x_amount > 0 else -1
            x_amount = abs(x_amount)

            y_amount = int(self.y_settings.amount_var.get())
            y_offset = int(self.y_settings.offset_var.get())
            y_dir = 1 if y_amount > 0 else -1
            y_amount = abs(y_amount)

            x_start, y_start = self.model.stage_setpoint[0], self.model.stage_setpoint[1]

            for x_idx in range(x_amount):
                for y_idx in range(y_amount):
                    self.model.move_absolute(
                        {
                            "x": x_start + x_dir * x_idx * x_offset,
                            "y": y_start + y_dir * y_idx * y_offset,
                        }
                    )
                    self.model.autofocus(blue_only=False)

                    self.model.move_relative({"z": -85.0})
                    self.model.non_blocking_delay(0.5)
                    self.model.enter_uv_mode(mode_switch_autofocus=False)
                    self.model.autofocus(blue_only=True)

                    self.model.begin_patterning()
                    self.model.enter_red_mode(mode_switch_autofocus=False)

        def on_abort():
            pass

        self.begin_tiling_button = ttk.Button(self.frame, text="Begin Tiling", command=on_begin, state="enabled")
        self.begin_tiling_button.grid(row=2, column=0)
        self.abort_tiling_button = ttk.Button(self.frame, text="Abort Tiling", command=on_abort, state="disabled")
        self.abort_tiling_button.grid(row=3, column=0)


class LithographerGui:
    root: Tk
    event_dispatcher: EventDispatcher

    def __init__(self, config: LithographerConfig):
        self.root = Tk()
        self.event_dispatcher = EventDispatcher(
            config.stage, 
            TkProjector(self.root), 
            self.root, 
            config.camera,
            config.red_exposure,
            config.uv_exposure,
        )
        self.event_dispatcher.initialize_alignment(config)

        self.shown_image = ShownImage.CLEAR

        self.camera = CameraFrame(self.root, self.event_dispatcher, config.camera, config.camera_scale)
        self.camera.frame.grid(row=0, column=0)

        self.middle_panel = ttk.Frame(self.root)
        self.middle_panel.grid(row=2, column=0)

        self.bottom_panel = ttk.Frame(self.root)
        self.bottom_panel.grid(row=3, column=0)

        # self.exposure_history_frame = ExposureHistoryFrame(self.bottom_panel, self.event_dispatcher)
        # self.exposure_history_frame.frame.grid(row=0, column=4)

        self.chip_frame = ChipFrame(self.bottom_panel, self.event_dispatcher)
        self.chip_frame.frame.grid(row=0, column=0)

        self.global_settings_frame = GlobalSettingsFrame(self.bottom_panel, self.event_dispatcher, config.alignment.enabled)
        self.global_settings_frame.frame.grid(row=0, column=2)

        self.stage_position_frame = StagePositionFrame(self.middle_panel, self.event_dispatcher)
        self.stage_position_frame.frame.grid(row=0, column=1)

        self.image_adjust_frame = ImageAdjustFrame(self.bottom_panel, self.event_dispatcher)
        self.image_adjust_frame.frame.grid(row=0, column=1)

        self.mode_select_frame = ModeSelectFrame(self.middle_panel, self.event_dispatcher)
        self.mode_select_frame.notebook.grid(row=0, column=2)

        self.tiling_frame = TilingFrame(self.middle_panel, self.event_dispatcher)
        self.tiling_frame.frame.grid(row=0, column=3)

        self.exposure_frame = self.mode_select_frame.uv_mode_frame.exposure_frame
        self.patterning_frame = self.mode_select_frame.uv_mode_frame.patterning_frame

        self.pattern_progress = Progressbar(self.root, orient="horizontal", mode="determinate")
        self.pattern_progress.grid(row=1, column=0, sticky="ew")

        self.multi_image_select_frame = MultiImageSelectFrame(self.middle_panel, self.event_dispatcher)
        self.multi_image_select_frame.frame.grid(row=0, column=0)

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())
        # self.debug.info("Debug info will appear here")

        # Things that have to after the main loop begins
        def on_start():
            self.camera.start()
            self.event_dispatcher.enter_red_mode(mode_switch_autofocus=False) # ensure exposure settings are correctly set
            if self.event_dispatcher.hardware.stage.has_homing():
                self.event_dispatcher.home_stage()
                #self.event_dispatcher.move_relative({"x": 5000.0, "y": 3500.0, "z": 1900.0})
            messagebox.showinfo(
                message="BEFORE CONTINUING: Ensure that you move the projector window to the correct display! Click on the fullscreen, completely black window, then press Windows Key + Shift + Left Arrow until it no longer is visible!"
            )

        self.root.after(0, on_start)
    

    def cleanup(self):
        print("Patterning GUI closed.")
        print("TODO: Cleanup")
        self.root.destroy()
        self.camera.cleanup()
        # if RUN_WITH_STAGE:
        # serial_port.close()


def main():
    try:
        with open("config.toml", "r") as f:
            config = toml.load(f)
    except FileNotFoundError:
        print("config.toml does not exist, copying settings from default.toml")
        shutil.copy("default.toml", "config.toml")
        with open("config.toml", "r") as f:
            config = toml.load(f)

    # STAGE CONFIG

    stage_config = config["stage"]
    if stage_config["enabled"]:
        serial_port = serial.Serial(stage_config["port"], stage_config["baud-rate"])
        print(f"Using serial port {serial_port.name}")
        stage = GrblStage(serial_port, stage_config["homing"])
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
        import camera.flir.flir_camera as flir
        camera = flir.FlirCamera()
    elif camera_config["type"] in ("basler", "pylon"):
        from camera.pylon import BaslerPylon
        camera = BaslerPylon()
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
    alignment_model = alignment_config.get("model_path", "best.pt")
    
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
    
    lithographer_config = LithographerConfig(
        stage,
        camera,
        camera_scale,
        red_exposure,
        uv_exposure,
        alignment_config,
    )

    lithographer = LithographerGui(lithographer_config)
    lithographer.root.mainloop()


if __name__ == "__main__":
    main()

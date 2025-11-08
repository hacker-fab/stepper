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
import math
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

        # self.set_shown_image(ShownImage.UV_FOCUS)
        self.set_shown_image(ShownImage.CLEAR) # enter uv mode: don't project uv

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

'''
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
'''

class StagePositionFrame:
    def __init__(self, parent, event_dispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher
        
        # Position display at top
        self.position_frame = ttk.LabelFrame(self.frame, text="Current Position (µm)")
        self.position_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Track all interactive widgets for locking
        self.xy_widgets = []
        self.z_widgets = []
        self.all_widgets = []
        
        # self.position_labels = []
        self.position_inputs = []
        for i, coord in enumerate(["X", "Y", "Z"]):
            label = ttk.Label(self.position_frame, text=f"{coord}:")
            label.grid(row=0, column=i, padx=5) # X: 0.0 label
            # self.position_labels.append(label)

            # Entry widget for position
            position_input = IntEntry(parent=self.position_frame, default=0)
            position_input.widget.grid(row=1, column=i, padx=(0, 5))
            position_input.widget.config(width=10)
            self.position_inputs.append(position_input)
            
            # Add to appropriate widget list for locking
            if i < 2:  # X and Y
                self.xy_widgets.append(position_input.widget)
            else:  # Z
                self.z_widgets.append(position_input.widget)
        
        # Add "Set Position" button
        self.set_position_button = ttk.Button(
            self.position_frame, 
            text="Set Stage Position", 
            command=self._on_set_position
        )
        self.set_position_button.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(5, 0))
        self.all_widgets.append(self.set_position_button)
        
        # Control panel container
        control_frame = ttk.Frame(self.frame)
        control_frame.grid(row=1, column=0, columnspan=2)
        
        # XY circular control (left)
        xy_frame = ttk.Frame(control_frame)
        xy_frame.grid(row=0, column=0, padx=10)
        self.create_xy_control(xy_frame)
        
        # Z vertical control (right)
        z_frame = ttk.Frame(control_frame)
        z_frame.grid(row=0, column=1, padx=10)
        self.create_z_control(z_frame)
        
        # Shortcuts at bottom
        self.shortcut_frame = ttk.LabelFrame(self.frame, text="Shortcuts")
        self.shortcut_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        def on_chip_origin():
            event_dispatcher.move_absolute({"x": -14500.0, "y": -13500.0, "z": -13844.0})
        
        def on_chip_unload():
            event_dispatcher.move_absolute({"x": -14500.0, "y": -14500.0, "z": -14500.0})
        
        btn_state = "normal" if event_dispatcher.hardware.stage.has_homing() else "disabled"
        self.chip_origin_button = ttk.Button(self.shortcut_frame, text="Chip origin", 
                                            command=on_chip_origin, state=btn_state)
        self.chip_origin_button.grid(row=0, column=0, padx=5)
        
        self.chip_unload_button = ttk.Button(self.shortcut_frame, text="Load/unload", 
                                            command=on_chip_unload, state=btn_state)
        self.chip_unload_button.grid(row=0, column=1, padx=5)
        
        # Listen to events
        event_dispatcher.add_event_listener("MOVEMENT_LOCK_CHANGED", self._on_lock_change)
        event_dispatcher.add_event_listener("STAGE_POSITION_CHANGED", self._on_position_change)
    
    def _on_set_position(self):
        """Handle the Set Position button click"""
        x = self.position_inputs[0].get()
        y = self.position_inputs[1].get()
        z = self.position_inputs[2].get()
        self.event_dispatcher.move_absolute({"x": x, "y": y, "z": z})

    def create_xy_control(self, parent):
        """Create circular XY control with 4 quadrants and 4 layers each"""
        canvas_size = 300
        center = canvas_size // 2
        
        self.xy_canvas = tkinter.Canvas(parent, width=canvas_size, height=canvas_size, 
                                   bg='#34495e', highlightthickness=0)
        self.xy_canvas.pack()
        
        # Step sizes for each layer (inner to outer)
        step_sizes = [10, 50, 100, 250]
        
        # Radii for the 4 layers (inner to outer)
        radii = [40, 75, 110, 145]
        
        # Colors for each layer (lighter as you go out)
        colors = ['#52796f', '#5a9a8b', '#7ab8a8', '#95c9be']
        
        # Draw concentric circles for layers
        for i, radius in enumerate(radii[::-1]):
            self.xy_canvas.create_oval(
                center - radius, center - radius,
                center + radius, center + radius,
                fill=colors[i], outline='#2c3e50', width=2
            )
        
        # Draw cross lines to divide into quadrants
        cx, cy = center, center
        half_L = canvas_size / 2

        for angle_deg, color in [(-45, "#2c3e50"), (45, "#2c3e50")]:
            angle_rad = math.radians(angle_deg)

            # Adjust for Tkinter’s downward Y-axis
            x0 = cx - half_L * math.sin(angle_rad)
            y0 = cy - half_L * math.cos(angle_rad)
            x1 = cx + half_L * math.sin(angle_rad)
            y1 = cy + half_L * math.cos(angle_rad)

            self.xy_canvas.create_line(x0, y0, x1, y1, fill=color, width=3)
        
        
        # Labels for directions
        label_offset = 160
        labels = [
            (center, 10, "+Y", '#85c1e9'),
            (center, canvas_size - 10, "-Y", '#85c1e9'),
            (canvas_size - 10, center, "+X", '#f4d03f'),
            (10, center, "-X", '#f4d03f'),
        ]
        
        for x, y, text, color in labels:
            self.xy_canvas.create_text(x, y, text=text, fill=color, 
                                      font=('Arial', 12, 'bold'))
        
        # Add step size labels on each layer
        for i, (radius, step) in enumerate(zip(radii, step_sizes)):
            # Show on top quadrant
            y_pos = center - radius + 20
            self.xy_canvas.create_text(center, y_pos, text=str(step), 
                                      fill='white', font=('Arial', 9, 'bold'))
        
        # Bind click events
        self.xy_canvas.bind('<Button-1>', self._on_xy_click)
        
        # Store reference for locking
        self.xy_widgets.append(self.xy_canvas)
    
    def _on_xy_click(self, event):
        """Handle clicks on the XY canvas"""
        canvas_size = 300
        center = canvas_size // 2
        
        # Calculate distance from center and angle
        dx = event.x - center
        dy = event.y - center
        distance = math.sqrt(dx**2 + dy**2)
        
        # Ignore clicks in the center circle
        # if distance < 15:
        #     return
        
        # Determine which layer (step size)
        radii = [40, 75, 110, 145]
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
            # Horizontal movement (X direction)
            if dx > 0:
                # +X (right)
                self.event_dispatcher.move_relative({"x": step_size})
            else:
                # -X (left)
                self.event_dispatcher.move_relative({"x": -step_size})
        else:
            # Vertical movement (Y direction)
            if dy < 0:
                # +Y (up)
                self.event_dispatcher.move_relative({"y": step_size})
            else:
                # -Y (down)
                self.event_dispatcher.move_relative({"y": -step_size})
    
    def create_z_control(self, parent):
        """Create vertical Z control bar with 8 sections"""
        bar_width = 80
        section_height = 35
        total_height = section_height * 8
        
        self.z_canvas = tkinter.Canvas(parent, width=bar_width, height=total_height,
                                 bg='#34495e', highlightthickness=0)
        self.z_canvas.pack()
        
        # Step sizes for Z (top 4 are +Z, bottom 4 are -Z)
        step_sizes = [10, 50, 100, 250]
        
        # Colors (gradient from light to dark for +Z, then dark to light for -Z)
        colors_plus = ['#a8dadc', '#87bdbf', '#66a0a3', '#458486']
        colors_minus = ['#458486', '#66a0a3', '#87bdbf', '#a8dadc']
        
        # Create sections
        for i in range(8):
            y_start = i * section_height
            y_end = (i + 1) * section_height
            
            if i < 4:
                # Top half: +Z movement
                color = colors_plus[i]
                step = step_sizes[3 - i]
                label = f"+Z\n{step}"
                direction = "+"
            else:
                # Bottom half: -Z movement
                color = colors_minus[i - 4]
                step = step_sizes[i - 4]
                label = f"-Z\n{step}"
                direction = "-"
            
            # Draw rectangle
            rect_id = self.z_canvas.create_rectangle(
                0, y_start, bar_width, y_end,
                fill=color, outline='#2c3e50', width=2
            )
            
            # Draw label
            text_id = self.z_canvas.create_text(
                bar_width // 2, (y_start + y_end) // 2,
                text=label, fill='#1a1a1a', font=('Arial', 10, 'bold')
            )
            
            # Bind click events
            for item_id in [rect_id, text_id]:
                self.z_canvas.tag_bind(
                    item_id, '<Button-1>',
                    lambda e, d=direction, s=step: self._on_z_click(d, s)
                )
        
        # Add label at top
        ttk.Label(parent, text="Z Control", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        # Store reference for locking
        self.z_widgets.append(self.z_canvas)
    
    def _on_z_click(self, direction, step_size):
        """Handle clicks on Z control sections"""
        if direction == "+":
            self.event_dispatcher.move_relative({"z": step_size})
        else:
            self.event_dispatcher.move_relative({"z": -step_size})
    
    def _on_lock_change(self):
        """Handle movement lock state changes"""
        from enum import Enum
        
        # Assuming MovementLock enum exists in your code
        lock = self.event_dispatcher.movement_lock
        
        # Map lock enum to state (you may need to adjust based on your actual enum)
        if hasattr(lock, 'value'):
            lock_value = lock.value if hasattr(lock, 'value') else str(lock)
        else:
            lock_value = str(lock)
        
        if 'UNLOCKED' in lock_value.upper():
            # All movements allowed
            for w in self.xy_widgets + self.z_widgets:
                if isinstance(w, tkinter.Canvas):
                    w.configure(state='normal')
                else:
                    w.configure(state='normal')
        elif 'XY_LOCKED' in lock_value.upper():
            # Only Z movement allowed
            for w in self.xy_widgets:
                if isinstance(w, tkinter.Canvas):
                    w.configure(state='disabled')
                else:
                    w.configure(state='disabled')
            for w in self.z_widgets:
                if isinstance(w, tkinter.Canvas):
                    w.configure(state='normal')
                else:
                    w.configure(state='normal')
        else:  # LOCKED
            # No movements allowed
            for w in self.xy_widgets + self.z_widgets:
                if isinstance(w, tkinter.Canvas):
                    w.configure(state='disabled')
                else:
                    w.configure(state='disabled')
    
    def _on_position_change(self):
        """Update position display when stage moves"""
        pos = self.event_dispatcher.stage_setpoint
        for i, coord in enumerate(["X", "Y", "Z"]):
            self.position_labels[i].configure(text=f"{coord}: {pos[i]:.1f}")

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
            state="readonly"
        )
        self.image_dropdown.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.image_dropdown.bind("<<ComboboxSelected>>", self._on_selection_change)
        
        # Add a button to load the selected image
        self.load_button = ttk.Button(self.widget, text="Load Selected", command=self._load_selected)
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
                self.frame, 
                THUMBNAIL_SIZE, 
                predefined_images,
                on_select=import_command
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


class PatternDisplayFrame: # read only pattern display in red and uv focusing mode
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
        
        ttk.Label(self.pattern_display_frame, text="(Upload in Pattern Upload tab)", 
                 font=("TkDefaultFont", 8), foreground="gray").grid(row=1, column=0)
        
        event_dispatcher.add_event_listener(Event.PATTERN_IMAGE_CHANGED, self._update_pattern_display)
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

class UvFocusFrame: # crosses
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
            predefined_images=uv_focus_predefined
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
        self.event_dispatcher = event_dispatcher

        # Create left and right sections
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

        # Stage position controls (left side)
        self.stage_position_frame = StagePositionFrame(self.middle_frame, event_dispatcher)
        self.stage_position_frame.frame.grid(row=0, column=0)
        
        # Pattern preview display (right side)
        self.pattern_display = PatternDisplayFrame(self.right_frame, event_dispatcher)
        self.pattern_display.frame.grid(row=0, column=0)

        # Red focus image selection
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
    def __init__(self, parent, event_dispatcher):
        self.frame = ttk.Frame(parent, name="uvmodeframe")
        # Create left and right sections
        self.left_frame = ttk.Frame(self.frame)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=(0,10))
        
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
        self.uv_focus_frame.frame.grid(row=0, column=0, pady=(10,0))

        # Stage position controls (middle)
        self.stage_position_frame = StagePositionFrame(self.middle_frame, event_dispatcher)
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
            self.upload_frame,
            "Select Pattern Image",
            self._on_pattern_upload
        )
        self.pattern_selector.frame.grid(row=0, column=0)
        
        # Pattern info display
        self.info_frame = ttk.LabelFrame(container, text="Pattern Information")
        self.info_frame.grid(row=1, column=0)
        
        self.pattern_path_var = StringVar(value="No pattern loaded")
        ttk.Label(self.info_frame, text="Current Pattern:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.info_frame, textvariable=self.pattern_path_var, 
                 foreground="blue").grid(row=0, column=1, sticky="w", padx=(10,0))
        
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
        instruction_text = ("Upload your pattern image using the selector above. "
                          "This pattern will be used in UV mode for lithography.")
        ttk.Label(self.upload_frame, text=instruction_text, 
                 wraplength=300).grid(row=1, column=0, padx=5, pady=5)

    def _on_pattern_upload(self, _):
        """Handle pattern upload"""
        if self.pattern_selector.thumb.image:
            # Update the event dispatcher with the new pattern
            self.event_dispatcher.set_pattern_image(
                self.pattern_selector.thumb.image, 
                self.pattern_selector.thumb.path
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
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.notebook = ttk.Notebook(parent)

        # Add Pattern Upload tab first
        self.pattern_upload_frame = PatternUploadFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.pattern_upload_frame.frame, text="Pattern Upload")
        self.red_mode_frame = RedModeFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.red_mode_frame.frame, text="Red Mode")
        self.uv_mode_frame = UvModeFrame(self.notebook, event_dispatcher)
        self.notebook.add(self.uv_mode_frame.frame, text="UV Mode")

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


class ProjectorDisplayFrame:
    """Frame to display what the projector is currently showing"""
    
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher
        
        # Main label frame
        self.display_frame = ttk.LabelFrame(self.frame, text="Projector Output")
        self.display_frame.grid(row=0, column=0)
        
        # Create placeholder image
        # Using a similar size to camera preview for consistency
        self.display_size = THUMBNAIL_SIZE
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
        self.event_dispatcher.root.after(100, self._update_display)
        
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
        if img is None or (shown_image == ShownImage.CLEAR and not self.event_dispatcher.patterning_busy):
            # Show black placeholder when clear
            placeholder = Image.new("RGB", self.display_size, "black")
            self.photo = image_to_tk_image(placeholder)
        else:
            # Resize the projector output to fit in display
            try:
                display_img = img.copy()
                # Check if image is all black
                import numpy as np
                img_array = np.array(display_img)
                is_black = np.all(img_array == 0)
                # print(f"Image all black: {is_black}, min: {img_array.min()}, max: {img_array.max()}")
                
                display_img.thumbnail(self.display_size, Image.Resampling.LANCZOS)
                self.photo = image_to_tk_image(display_img)
            except Exception as e:
                print(f"Error displaying projector image: {e}")
                import traceback
                traceback.print_exc()
                placeholder = Image.new("RGB", self.display_size, "red")
                self.photo = image_to_tk_image(placeholder)
        
        self.label.configure(image=self.photo)
        
        # Schedule periodic updates to catch any missed changes
        self.event_dispatcher.root.after(500, self._update_display)

import os
from pathlib import Path
import re
from PIL import Image
import numpy as np

class mapFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        self.capture_button = ttk.Button(
            self.frame, 
            text="Capture Map", 
            command=lambda: self.takeMapImage(5000, 4000, 500, 500) # hardcoded
        )
        self.capture_button.grid(row=0, column=0)

    def takeMapImages(self, img_w, img_h, stride_x, stride_y, output_dir="maps"):
        """
        Take snapshots to cover an area of img_w * img_h with Z-shaped movement.
        stride_x: Horizontal distance between snapshots (in microns)
        stride_y: Vertical distance
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        num_cols = int(img_w / stride_x) + 1
        num_rows = int(img_h / stride_y) + 1
        
        print(f"Starting map capture: {num_rows} rows x {num_cols} cols\
              = {num_rows * num_cols} images")
        
        # get starting position
        start_x, start_y, start_z = self.event_dispatcher.stage_setpoint
        
        for row in range(num_rows):
            current_y = start_y + row * stride_y
            
            # Move to start of row
            if row > 0: # Skip first row since we are already there
                self.event_dispatcher.move_absolute({
                    "x": start_x,
                    "y": current_y,
                    "z": start_z
                })
                self.event_dispatcher.non_blocking_delay(0.5)
            
            # Iterate through columns in this row
            for col in range(num_cols):
                current_x = start_x + col * stride_x
                if col > 0:
                    self.event_dispatcher.move_absolute({
                        "x": current_x,
                        "y": current_y,
                        "z": start_z
                    })
                    self.event_dispatcher.non_blocking_delay(0.5)
                
                # filename with row and col index
                filename = output_path / f"{row:02d}{col:02d}.png"
                
                self.event_dispatcher.on_event(Event.SNAPSHOT, str(filename))
                self.event_dispatcher.non_blocking_delay(0.3)
                
                print(f"Captured image {row:02d}{col:02d} at\
                      X:{current_x:.1f} Y:{current_y:.1f}")
        
        # Return to starting position
        self.event_dispatcher.move_absolute({
            "x": start_x,
            "y": start_y,
            "z": start_z
        })
        
        print(f"Map capture complete! {num_rows * num_cols}\
              images saved to {output_dir}/")

    def stitchMapImages(self, stride_x, stride_y, output_dir="maps",
                        stitched_filename="stitched_map.png"):
        """
        Stitch together all captured map images into one large image.
        Returns PIL.Image: The stitched image with stitched_filename,
        or None if stitching failed
        """
        
        output_path = Path(output_dir)
        
        if not output_path.exists():
            print(f"Error: Directory {output_dir} does not exist")
            return None
        
        # Find all image files with pattern XXYY.png
        image_files = {} # dict of images: key = (row, col), value = file path
        pattern = re.compile(r'(\d{2})(\d{2})\.png') # group1 = XX group2 = YY
        
        for file in output_path.glob('*.png'): # all files ends with .png
            match = pattern.match(file.name)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                image_files[(row, col)] = file
        
        if not image_files:
            print(f"Error: No images found in {output_dir}")
            return None
        
        print(f"Found {len(image_files)} images to stitch")
        
        rows = [coord[0] for coord in image_files.keys()]
        cols = [coord[1] for coord in image_files.keys()]
        num_rows = max(rows) + 1
        num_cols = max(cols) + 1
        
        print(f"Grid size: {num_rows} rows x {num_cols} cols")

        img_width, img_height = stride_x, stride_y
                
        # large blank canvas
        stitched_width = num_cols * img_width
        stitched_height = num_rows * img_height
        stitched_image = Image.new('RGB', (stitched_width, stitched_height), color='black')
        
        print(f"Stitched image size: {stitched_width}x{stitched_height}")
        
        # Stitch images together
        for row in range(num_rows):
            for col in range(num_cols):
                if (row, col) in image_files:
                    # Load image
                    img_path = image_files[(row, col)]
                    img = Image.open(img_path)
                    
                    x_pos = col * img_width
                    y_pos = row * img_height
                    
                    stitched_image.paste(img, (x_pos, y_pos))
                    
                    print(f"Stitched image {row:02d}{col:02d} at position ({x_pos}, {y_pos})")
                else:
                    print(f"Warning: Missing image at position {row:02d}{col:02d}")
        
        output_file = output_path / stitched_filename
        stitched_image.save(output_file)
        print(f"Stitched image saved to {output_file}")
        
        return stitched_image

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

        self.top_panel = ttk.Frame(self.root)
        self.top_panel.grid(row=0, column=0)
        
        # Camera frame (top)
        self.camera = CameraFrame(self.top_panel, self.event_dispatcher, config.camera, config.camera_scale)
        self.camera.frame.grid(row=0, column=1)

        # Projector display (top)
        self.projector_display = ProjectorDisplayFrame(self.top_panel, self.event_dispatcher)
        self.projector_display.frame.grid(row=0, column=0, padx=5, pady=5, sticky='')

        # Progress bar
        self.pattern_progress = Progressbar(self.root, orient="horizontal", mode="determinate")
        self.pattern_progress.grid(row=1, column=0, sticky="ew")

        # Main tab interface (replaces middle_panel)
        self.mode_select_frame = ModeSelectFrame(self.root, self.event_dispatcher)
        self.mode_select_frame.notebook.grid(row=2, column=0, sticky="nsew")

        # Bottom panel (chip log and image adjustment and tiling)
        self.bottom_panel = ttk.Frame(self.root)
        self.bottom_panel.grid(row=3, column=0, sticky="ew")

        # Chip management
        self.chip_frame = ChipFrame(self.bottom_panel, self.event_dispatcher)
        self.chip_frame.frame.grid(row=0, column=0)

        # Image adjustment controls
        self.image_adjust_frame = ImageAdjustFrame(self.bottom_panel, self.event_dispatcher)
        self.image_adjust_frame.frame.grid(row=0, column=1)

        # Global settings
        self.global_settings_frame = GlobalSettingsFrame(self.bottom_panel, self.event_dispatcher, config.alignment.enabled)
        self.global_settings_frame.frame.grid(row=0, column=2)

        # Tiling controls
        self.tiling_frame = TilingFrame(self.bottom_panel, self.event_dispatcher)
        self.tiling_frame.frame.grid(row=0, column=3)

        # Configure grid weights for proper expansion
        # self.root.columnconfigure(0, weight=1)
        # self.root.rowconfigure(2, weight=1)  # Let the tab area expand

        # Legacy references for compatibility (if needed elsewhere in code)
        self.exposure_frame = self.mode_select_frame.uv_mode_frame.exposure_frame
        self.patterning_frame = self.mode_select_frame.uv_mode_frame.patterning_frame

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
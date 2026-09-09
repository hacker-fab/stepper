
# TODO: Don't hardcode
import cv2
import numpy as np
import json
import os
import time

import onnxruntime as rt
from PIL import ImageOps
from ultralytics import YOLO
from typing import Callable, List
from hardware import ImageProcessSettings, Lithographer, ProcessedImage

from datetime import datetime
from pathlib import Path
from tkinter import Tk, messagebox
from typing import Optional

from camera.camera_module import CameraModule

from projector import TkProjector
from stage_control.stage_controller import StageController

# importing utilities
from lib.structs import *

THUMBNAIL_SIZE: tuple[int, int] = (160, 90)
#The values set here are not used and instead come from the config file
DEFAULT_RED_EXPOSURE: float = 4167.0
DEFAULT_UV_EXPOSURE: float = 25000.0

def fetch_focus_score(camera_image, blue_only, ddepth=cv2.CV_64F, kernel_size=5, log=False):
    """ fetch_focus_score: computes the laplacian focal score after some
    pre-processing of the camera image. The key is to detect the edges better
    than other parts of the image that might not be suitable to be focused on. """

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
    img_lapl = (np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1))) / mean
    if save:
        print('saved focus: ', np.min(img_lapl), np.max(img_lapl))
        cv2.imwrite(save, img_lapl * 255.0 / 5.0)
    return img_lapl.var() / mean

def detect_markers(model, image, draw_rectangle=False):
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


class EventDispatcher:
    hardware: Lithographer
    root: Tk
    model: Optional[YOLO | rt.InferenceSession]
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
        self.num_rows = None
        self.num_cols = None

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
        self.stage_setpoint = (0.0,0.0,0.0)

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
        print(f"_refresh_red_focus: size: {self.image_adjust_position}, posterization: {self.posterize_strength}, projector size: {self.hardware.projector.size()}, border size = {self.border_size}")
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
        if(self.hardware.stage.has_homing()): # debugging statements
            print(f"Moving to position: {coords}")
            print(f"Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}")
        
        # find new coordinates -> some nuance exists between work and gui positioning
        # in work position, the x moves in negative direction (away from home) and y moves in positive direction (away from home)
        x = coords.get("x", self.stage_setpoint[0])
        y = coords.get("y", self.stage_setpoint[1])
        z = coords.get("z", self.stage_setpoint[2])
        set_point = (x, y, z)

        if self.hardware.stage.has_homing():
            ok, msg = self._check_bounds(set_point)
            if not ok:
                self.create_warning(msg)
                return False

        try:
            self.hardware.stage.move_absolute(coords)
            self.stage_setpoint = set_point
            self.on_event(Event.STAGE_POSITION_CHANGED)
            return True
        
        except(RuntimeError) as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            return False

        except(Exception) as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")
            self.stage_setpoint = self.hardware.stage.get_position()
            self.on_event(Event.STAGE_POSITION_CHANGED) 
            return False

        
    def _check_bounds(self, set_point):
        bounds = self.hardware.stage.get_bounds()

        if bounds is None:
            return True  # no homing, no bounds enforced
        
        axes = [('x', 0), ('y', 1), ('z', 2)]
        for name, i in axes:
            lo, hi = bounds[name]
            val = set_point[i]
            if not (lo <= val <= hi):
                return False, (f"Moving {name.upper()} to {val} prohibited. "
                            f"Boundaries are [{lo}, {hi}]")
        return True, None

    def move_relative(self, coords: dict[str, float]):

        if(self.hardware.stage.has_homing()): # debugging statements
            print(f"Moving by: {coords} | Current position: {self.stage_setpoint[0]}, {self.stage_setpoint[1]}, {self.stage_setpoint[2]}")
        # find new coordinates -> some nuance exists between work and gui positioning
        # in work position, the x moves in negative direction (away from home) and y moves in positive direction (away from home
        x = self.stage_setpoint[0] + coords.get("x", 0)
        y = self.stage_setpoint[1] + coords.get("y", 0)
        z = self.stage_setpoint[2] + coords.get("z", 0)
        set_point = (x, y, z)
        
        # if soft limits and max travel set, then enforce boundaries
        if(self.hardware.stage.has_homing()):
            ok, msg = self._check_bounds(set_point)
            if not ok:
                self.create_warning(msg)
                return

        try:
            self.hardware.stage.move_relative(coords)
            self.stage_setpoint = set_point
            self.on_event(Event.STAGE_POSITION_CHANGED)

        except(RuntimeError) as e:
            self.create_warning(f"{str(e)}. Please remove your chip, restart the program.")

        except(Exception) as e:
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
    
    def set_prev_pattern_image(self, img: Image.Image, path: str):
        self.prev_pattern_image = img
        self.prev_pattern_image_path = path

    def set_stitched_image(self, img: Image.Image, path: str):
        self.stitched_image = img
        self.stitched_image_path = path

    def set_capture_folder(self, capture_folder: str):
        self.capture_folder = capture_folder

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
        print("invoked: set_image_position")
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

    def autofocus(self, blue_only, log=False, search=20, start=None):
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
                
        if self.hardware.stage.has_homing():

            counter = 0
            def sample():
                def one_sample():
                    return fetch_focus_score(self.camera_image, blue_only=blue_only, log=True)
                focus_score = sum([one_sample() for _ in range(3)])/3
                print("focus average:", focus_score)
                nonlocal counter
                if log:
                    log_file.write(f'{counter},{focus_score}\n')
                    cv2.imwrite(f'aftest/img{counter}.png', self.camera_image, log=True)
                counter += 1
                return focus_score
            
            print("Starting Autofocus...")
            best_score = -1.0
            best_z = 0
            if start == None:
                z_base = self.hardware.stage.get_autofocus()
            else:
                z_base = start

            # account for uv mode, where z-focus is different
            if blue_only == True:
                z_base -= 50.0
                if(self.move_absolute({"z": z_base})) == False:
                    self.create_warning("Failed autofocus, z-stage can't go past boundary limits")
                    self.set_autofocus_busy(False)
                    return
                self.non_blocking_delay(1.0)

            else:
                for i in range(-search, search, 2):
                    if not (self.move_absolute({"z": (z_base+i)})):
                        self.create_warning("Failed autofocus, z-stage can't go past boundary limits")
                        self.set_autofocus_busy(False)
                        return
                    self.non_blocking_delay(0.5)
                    new_score = sample()
                    # always check for optimal scores
                    if (new_score > best_score):
                        best_score = new_score
                        best_z = self.stage_setpoint[2]
                
                print(f"Fine grain sampling done, best focus is: {best_score}")
                self.move_absolute({"z":best_z})
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
    
    def get_model(self, path: str):
        """
        Based on which model we're using, we will feed the model
        a different session. 'best.pt' indicates YOLO while the onx 
        file indicates RF-DETR
        
        Note to developers: RF-DETR was trained on latent and developed patterns
        while YOLO model was trained on developed patterns only
        """
        if "best" in path:
            print("Using YOLO model")
            self.model = YOLO(path)
        else:
            print("Using RF_DETR model")
            session = rt.InferenceSession(path)
            self.model = session
        return True

    def initialize_alignment(self, config: LithographerConfig):
        self.config = config
        self.realtime_detection = config.alignment.enabled
        # Attempt loading the model even if detection is off by default
        try:
            print("loading model")
            model_path = config.alignment.model_path
            self.get_model(model_path)
            print("loaded model")
        except Exception as e:
            print(f"Failed to load alignment model: {e}")

    def set_snapshot_directory(self, directory: Path):
        self.snapshot_directory = directory
        self.snapshot_directory.mkdir(exist_ok=True)

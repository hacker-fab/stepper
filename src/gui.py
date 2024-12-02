import serial
#import tomllib
import cv2
import numpy as np
import time
import queue
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from hardware import Lithographer, ImageProcessSettings, ProcessedImage, StageWrapper
from typing import Callable, List, Optional
from PIL import Image
from camera.camera_module import CameraModule
from camera.webcam import Webcam
from stage_control.stage_controller import StageController
from stage_control.grbl_stage import GrblStage
from projector import ProjectorController, TkProjector
from enum import Enum
from lib.gui import IntEntry, Thumbnail
from lib.img import image_to_tk_image, fit_image
from tkinter.ttk import Progressbar
from tkinter import ttk, Tk, BooleanVar, IntVar, StringVar, messagebox
import toml # Need to use a package because we're stuck on Python 3.10
import tkinter

# TODO: Don't hardcode
THUMBNAIL_SIZE: tuple[int,int] = (160,90)

class ShownImage(Enum):
  Clear = 'clear'
  Pattern = 'pattern'
  Flatfield = 'flatfield'
  RedFocus = 'red_focus'
  UvFocus = 'uv_focus'

class PatterningStatus(Enum):
  Idle = 'idle'
  Patterning = 'patterning'
  Aborting = 'aborting'

OnShownImageChange = Callable[[ShownImage], None]

class Event(Enum):
  Snapshot = 'snapshot'

  ShownImageChanged = 'shownimagechanged'
  StagePositionChanged = 'stagepositionchanged'
  ImageAdjustChanged = 'imageadjustchanged'
  PatternImageChanged = 'patternimagechanged'
  MovementLockChanged = 'movementlockchanged'
  ExposurePatternProgressChanged = 'exposurepatternprogresschanged'
  PatterningBusyChanged = 'patterningbusychanged'

class MovementLock(Enum):
  '''Disables or enables moving the stage position manually'''
  Unlocked = 'unlocked'
  '''X, Y, and Z are free to move'''
  XYLocked = 'xylocked'
  '''Only Z (focus) is free to move to avoid smearing UV focus pattern'''
  Locked = 'locked'
  '''No positions can move to avoid disrupting patterning'''

class RedFocusSource(Enum):
  Image = 'image'
  Solid = 'solid'
  Pattern = 'pattern'

@dataclass
class ExposureLog:
  time: datetime
  name: str
  coords: tuple[float, float, float]
  duration: float # ms
  aborted: bool

class EventDispatcher:
  red_focus: ProcessedImage
  uv_focus: ProcessedImage
  pattern: ProcessedImage

  hardware: Lithographer

  has_camera: bool
  has_stage: bool

  listeners: dict[Event, List[Callable]]

  exposure_time: int
  posterize_strength: Optional[int]
  patterning_progress: float # ranges from 0.0 to 1.0

  autofocus_on_mode_switch: bool

  shown_image: ShownImage
  autofocus_busy: bool
  patterning_busy: bool

  pattern_image: Image.Image
  red_focus_image: Image.Image
  uv_focus_image: Image.Image

  solid_red_image: Image.Image
  red_focus_source: RedFocusSource

  image_adjust_position: tuple[float, float, float]
  border_size: float

  focus_score: float

  exposure_history: List[ExposureLog]

  def __init__(self, stage, proj, root, has_camera: bool, has_stage: bool):
    self.listeners = dict()

    self.hardware = Lithographer(stage, proj) # TODO:

    self.has_camera = has_camera
    self.has_stage = has_stage

    self.root = root

    self.red_focus = ProcessedImage()
    self.uv_focus = ProcessedImage()
    self.pattern = ProcessedImage()

    self.exposure_time = 8000
    self.posterize_strength = None

    self.patterning_progress = 0.0

    self.shown_image = ShownImage.Clear
    self.autofocus_busy = False
    self.patterning_busy = False

    self.autofocus_on_mode_switch = True

    self.pattern_image = Image.new('RGB', (1, 1), 'black')
    self.red_focus_image = Image.new('RGB', (1, 1), 'black')
    self.uv_focus_image = Image.new('RGB', (1, 1), 'black')

    self.solid_red_image = Image.new('RGB', (1, 1), 'red')
    self.red_focus_source = RedFocusSource.Image

    self.should_abort = False

    self.focus_score = 0.0

    self.image_adjust_position = (0.0, 0.0, 0.0)
    self.border_size = 0.0

    self.exposure_history = []

    self.add_event_listener(Event.ShownImageChanged, lambda: self._update_projector())
  
  def current_image(self):
    match self.shown_image:
      case ShownImage.Clear:
        return None
      case ShownImage.RedFocus:
        return self.red_focus.processed()
      case ShownImage.UvFocus:
        return self.uv_focus.processed()
      case ShownImage.Pattern:
        return self.pattern.processed()

  def _update_projector(self):
    img = self.current_image()
    if img is None:
      self.hardware.projector.clear()
    else:
      self.hardware.projector.show(img)

  def _refresh_pattern(self):
    self.pattern.update(image=self.pattern_image, settings=ImageProcessSettings(
      posterization=self.posterize_strength,
      color_channels=(False, False, True),
      flatfield=None,
      size=self.hardware.projector.size(),
      image_adjust=self.image_adjust_position,
      border_size=self.border_size,
      #size=self.pattern_image.size,
      #flatfield=None,
      #image_adjust=(0.0, 0.0, 0.0),
      #border_size=0.0,
    ))

    if self.red_focus_source == RedFocusSource.Pattern:
      self._refresh_red_focus()

    # TODO:
    # Image adjust, resizing, and flatfield correction are performed *AFTER SLICING*

    self.on_event(Event.PatternImageChanged)
  
  def set_red_focus_source(self, source):
    self.red_focus_source = source
    self._refresh_red_focus()
  
  def _red_focus_source(self):
    match self.red_focus_source:
      case RedFocusSource.Image:
        return self.red_focus_image
      case RedFocusSource.Solid:
        return self.solid_red_image
      case RedFocusSource.Pattern:
        return self.pattern_image.getchannel('B').convert('RGBA')

  def _refresh_red_focus(self):
    if self.hardware.projector.size() != self.solid_red_image.size:
      self.solid_red_image = Image.new('RGB', self.hardware.projector.size(), 'red')

    img = self._red_focus_source()

    self.red_focus.update(image=img, settings=ImageProcessSettings(
      posterization=self.posterize_strength,
      flatfield=None,
      color_channels=(True, False, False),
      size=self.hardware.projector.size(),
      image_adjust=self.image_adjust_position,
      border_size=self.border_size,
    ))

    if self.shown_image == ShownImage.RedFocus:
      self.on_event(Event.ShownImageChanged)
  
  def _refresh_uv_focus(self):
    self.uv_focus.update(image=self.uv_focus_image, settings=ImageProcessSettings(
      posterization=self.posterize_strength,
      flatfield=None,
      color_channels=(False, False, True),
      size=self.hardware.projector.size(),
      image_adjust=self.image_adjust_position,
      border_size=0.0,
    ))

    if self.shown_image == ShownImage.UvFocus:
      self.on_event(Event.ShownImageChanged)
  
  def set_posterize_strength(self, strength):
    self.posterize_strength = strength
    self._refresh_red_focus()
    self._refresh_uv_focus()
    self._refresh_pattern()
  
  def set_border_size(self, border_size):
    self.border_size = border_size
    self._refresh_red_focus()
    self._refresh_uv_focus()
    self._refresh_pattern()
  
  def set_shown_image(self, shown_image: ShownImage):
    print(f'set_shown_image({shown_image})')
    self.shown_image = shown_image
    self.on_event(Event.ShownImageChanged)
  
  def set_stage_position(self, x: float, y: float, z: float):
    self.hardware.stage.move_to({ 'x': x, 'y': y, 'z': z }, commit=True)
    self.on_event(Event.StagePositionChanged)
  
  def offset_stage_position(self, coords: dict[str, float]):
    self.hardware.stage.move_by(coords, commit=True)
    self.on_event(Event.StagePositionChanged)
 
  def set_use_solid_red(self, use: bool):
    self.use_solid_red = use
    self.set_shown_image(ShownImage.RedFocus)
    self._refresh_red_focus()
  
  def set_pattern_image(self, img: Image.Image, name: str):
    self.pattern_image = img
    self.pattern_image_name = name
    self._refresh_pattern()
  
  def set_red_focus_image(self, img: Image.Image):
    self.red_focus_image = img
    self._refresh_red_focus()
  
  def set_uv_focus_image(self, img: Image.Image):
    self.uv_focus_image = img
    self._refresh_uv_focus()
  
  def set_patterning_busy(self, busy: bool):
    self.patterning_busy = busy
    self.on_event(Event.MovementLockChanged)
    self.on_event(Event.PatterningBusyChanged)

  def set_progress(self, pattern_progress: float, exposure_progress: float):
    self.patterning_progress = pattern_progress
    self.exposure_progress = exposure_progress
    self.on_event(Event.ExposurePatternProgressChanged)
  
  def set_focus_score(self, focus_score: float):
    self.focus_score = focus_score
  
  def set_autofocus_busy(self, busy):
    self.autofocus_busy = busy
    self.on_event(Event.MovementLockChanged)
  
  def abort_patterning(self):
    self.should_abort = True
    print('Aborting patterning')
  
  def stage_position(self):
    pos = self.hardware.stage.stage_position
    return (pos['x'], pos['y'], pos['z'])
  
  def set_image_position(self, x, y, t):
    self.image_adjust_position = (x, y, t)
    self._refresh_red_focus()
    self._refresh_uv_focus()
    self._refresh_pattern()
    self.on_event(Event.ImageAdjustChanged)
  
  def image_position(self):
    return self.image_adjust_position
    
  def movement_lock(self):
    if self.patterning_busy or self.autofocus_busy:
      return MovementLock.Locked
    elif self.shown_image == ShownImage.UvFocus or self.shown_image == ShownImage.Pattern:
      return MovementLock.XYLocked
    else:
      return MovementLock.Unlocked

  def on_event(self, event: Event, *args, **kwargs):
    if event not in self.listeners:
      return

    for l in self.listeners[event]:
      l(*args, **kwargs)
  
  def on_event_cb(self, event: Event, *args, **kwargs):
    return lambda: self.on_event(event, *args, **kwargs) 
  
  def add_event_listener(self, event: Event, listener: Callable):
    if event not in self.listeners:
      self.listeners[event] = []
    self.listeners[event].append(listener)
  
  def begin_patterning(self):
    # TODO: Update patterning preview

    print('Patterning at ', self.hardware.stage.stage_position)

    duration = self.exposure_time
    
    print(f"Patterning 1 tiles for {duration}ms\nTotal time: {str(round((duration)/1000))}s")

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
    self.set_shown_image(ShownImage.Clear)
    self.root.update() # Force image to stop being displayed ASAP
    self.set_progress(1.0, 1.0)

    self.exposure_history.append(ExposureLog(
      datetime.now(),
      self.pattern_image_name,
      self.stage_position(),
      duration,
      self.should_abort,
    ))

    self.set_patterning_busy(False)

    if self.should_abort:
      print('Patterning aborted')
      self.should_abort = False
  
  def non_blocking_delay(self, t: float):
    start = time.time()
    while time.time() - start < t:
      self.root.update()
  
  def enter_red_mode(self):
    print('enter_red_mode')
    self.set_shown_image(ShownImage.RedFocus)
    if self.autofocus_on_mode_switch and self.has_camera and self.has_stage:
      self.autofocus()

  def enter_uv_mode(self):
    if not self.autofocus_busy and self.autofocus_on_mode_switch and self.has_camera and self.has_stage:
      # UV mode is usually needs about -60 to be in focus compared to red mode
      self.offset_stage_position({ 'z': -100.0 })

    self.set_shown_image(ShownImage.UvFocus)

    if self.autofocus_on_mode_switch and self.has_camera and self.has_stage:
      self.non_blocking_delay(2.0)
      self.autofocus()
  
  def autofocus(self):
    if self.autofocus_busy:
      print('Skipping nested autofocus!')
      return
    
    if not self.has_camera or not self.has_stage:
      print('Camera or stage disabled, skipping autofocus')
      return

    print('Starting autofocus')


    self.set_autofocus_busy(True)

    self.non_blocking_delay(2.0)

    last_focus = self.focus_score
    for i in range(30):
      self.offset_stage_position({ 'z': 10.0 })
      self.non_blocking_delay(0.5)
      if last_focus > self.focus_score:
        print(f'Successful coarse autofocus {i}')
        break
      last_focus = self.focus_score

    self.offset_stage_position({ 'z': -20.0 })
    self.non_blocking_delay(1.0)
    last_focus = self.focus_score
    for i in range(30):
      self.offset_stage_position({ 'z': 2.0 })
      self.non_blocking_delay(0.5)
      if last_focus > self.focus_score:
        print(f'Successful fine autofocus {i}')
        break
      last_focus = self.focus_score

    self.offset_stage_position({ 'z': -2.0 })

    self.set_autofocus_busy(False)

    print('Finished autofocus')





class SnapshotFrame:
  '''
  Presents a frame with a filename entry and a button to save screenshots of the current camera view.
  '''

  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)
    self.frame.grid(row=1, column=0)
    
    # TODO: Allow %X, %Y, %Z formats to save position on chip
    self.name_var = StringVar(value='output_%T.png')
    self.name_var.trace_add('write', lambda _a, _b, _c: self._refresh_name_preview())

    self.counter = 0

    self.name_entry = ttk.Entry(self.frame, textvariable=self.name_var)
    self.name_entry.grid(row=0, column=0)

    self.name_preview = ttk.Label(self.frame)
    self.name_preview.grid(row=0, column=1)

    def on_snapshot_button():
      event_dispatcher.on_event(Event.Snapshot, self._next_filename())
      self.counter += 1
      self._refresh_name_preview()

    self.button = ttk.Button(self.frame, text='Take Snapshot', command=on_snapshot_button)
    self.button.grid(row=0, column=2)

    self._refresh_name_preview()
  
  def _next_filename(self):
    counter_str = str(self.counter)
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    name = self.name_var.get()
    name = name.replace('%C', counter_str).replace('%c', counter_str)
    name = name.replace('%T', time_str).replace('%t', time_str)
    return name

  def _refresh_name_preview(self):
    self.name_preview.configure(text=f'Output File: {self._next_filename()}')


class CameraFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher, c: Optional[CameraModule]):
    self.frame = ttk.Frame(parent)
    self.label = ttk.Label(self.frame, text='No camera connected')
    self.label.grid(row=0, column=0, sticky='nesw')
    
    if c is not None:
      self.snapshot = SnapshotFrame(self.frame, event_dispatcher)
      self.snapshot.frame.grid(row=1, column=0)

    self.event_dispatcher = event_dispatcher

    self.snapshots_pending = queue.Queue()
    self.event_dispatcher.add_event_listener(Event.Snapshot, lambda filename: self.snapshots_pending.put(filename))

    self.gui_img = None
    self.camera = c

    def cameraCallback(image, dimensions, format):
      try:
        filename = self.snapshots_pending.get_nowait()
        print(f'Saving image {filename}')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)
      except queue.Empty:
        pass

      self.gui_camera_preview(image, dimensions)

    if self.camera is not None:
      if not self.camera.open():
        print('Camera failed to start')
      else:
        self.camera.setSetting('image_format', "rgb888")
        self.camera.setStreamCaptureCallback(cameraCallback)
        if not self.camera.startStreamCapture():
          print('Failed to start stream capture for camera')
    else:
      print('No camera enabled in config.toml; disabling camera-related features')

  def cleanup(self):
    if self.camera is not None:
      self.camera.close()

  def gui_camera_preview(self, camera_image, dimensions):
    start_time = time.time()
    resized_img = cv2.resize(camera_image, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    mean = np.sum(img) / (img.shape[0] * img.shape[1])
    img_lapl = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1) / mean
    self.event_dispatcher.set_focus_score(img_lapl.var() / mean)
    end_time = time.time()
    #print(f'Took {(end_time - start_time)*1000}ms to process')

    resized_img = cv2.resize(resized_img, (0, 0), fx=0.5, fy=0.5)
    gui_img = image_to_tk_image(Image.fromarray(resized_img, mode='RGB'))
    self.label.configure(image=gui_img) # type:ignore
    self.gui_img = gui_img

class StagePositionFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.position_intputs = []
    self.step_size_intputs = []

    self.xy_widgets = []
    self.z_widgets = []

    # Absolute

    self.absolute_frame = ttk.LabelFrame(self.frame, text='Stage Position')
    self.absolute_frame.grid(row=0, column=0)

    for i, coord in ((0, 'x'), (1, 'y'), (2, 'z')):
      self.position_intputs.append(IntEntry(parent=self.absolute_frame, default=0))
      self.position_intputs[-1].widget.grid(row=0,column=i)
    
    def callback_set():
      x, y, z = self._position()
      event_dispatcher.set_stage_position(x, y, z)

    self.set_position_button = ttk.Button(self.absolute_frame, text='Set Stage Position', command=callback_set)
    self.set_position_button.grid(row=1, column=0, columnspan=3, sticky='ew')

    # Relative 

    self.relative_frame = ttk.LabelFrame(self.frame, text='Adjustment')
    self.relative_frame.grid(row=1, column=0)

    for i, coord in ((0, 'x'), (1, 'y'), (2, 'z')):
      self.step_size_intputs.append(IntEntry(parent=self.relative_frame, default=10, min_value=-1000, max_value=1000))
      self.step_size_intputs[-1].widget.grid(row=3,column=i)

      def callback_pos(index, c):
        pos = list(event_dispatcher.stage_position())
        pos[index] += self.step_sizes()[index]
        event_dispatcher.set_stage_position(*pos)
      def callback_neg(index, c):
        pos = list(event_dispatcher.stage_position())
        pos[index] -= self.step_sizes()[index]
        event_dispatcher.set_stage_position(*pos)

      coord_inc_button = ttk.Button(self.relative_frame, text=f'+{coord.upper()}', command=partial(callback_pos, i, coord))
      coord_dec_button = ttk.Button(self.relative_frame, text=f'-{coord.upper()}', command=partial(callback_neg, i, coord))

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

    ttk.Label(self.relative_frame, text='Step Size (microns)', anchor='center').grid(row=2, column=0, columnspan=3, sticky='ew')

    self.all_widgets = self.xy_widgets + self.z_widgets + [self.set_position_button]

    def on_lock_change():
      lock = event_dispatcher.movement_lock()
      match lock:
        case MovementLock.Unlocked:
          for w in self.all_widgets:
            w.configure(state='normal')
        case MovementLock.XYLocked:
          for w in self.xy_widgets:
            w.configure(state='disabled')
          for w in self.z_widgets:
            w.configure(state='normal')
          self.set_position_button.configure(state='normal')
        case MovementLock.Locked:
          for w in self.all_widgets:
            w.configure(state='disabled')

    event_dispatcher.add_event_listener(Event.MovementLockChanged, on_lock_change)

    def on_position_change():
      pos = event_dispatcher.stage_position()
      for i in range(3):
        self.position_intputs[i].set(pos[i])

    event_dispatcher.add_event_listener(Event.StagePositionChanged, on_position_change)
  
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

    self.absolute_frame = ttk.LabelFrame(self.frame, text='Image Adjustment')
    self.absolute_frame.grid(row=0, column=0)

    for i, coord in ((0, 'x'), (1, 'y'), (2, 'ϴ')):
      self.position_intputs.append(IntEntry(parent=self.absolute_frame, default=0))
      self.position_intputs[-1].widget.grid(row=0,column=i)
    
    def callback_set():
      x, y, t = self._position()
      event_dispatcher.set_image_position(x, y, t)

    self.set_position_button = ttk.Button(self.absolute_frame, text='Set Image Position', command=callback_set)
    self.set_position_button.grid(row=1, column=0, columnspan=3, sticky='ew')

    # Relative 

    self.relative_frame = ttk.LabelFrame(self.frame, text='Adjustment')
    self.relative_frame.grid(row=1, column=0)

    for i, coord in ((0, 'x'), (1, 'y'), (2, 'ϴ')):
      self.step_size_intputs.append(IntEntry(parent=self.relative_frame, default=10, min_value=-1000, max_value=1000))
      self.step_size_intputs[-1].widget.grid(row=3,column=i)

      def callback_pos(index, c):
        pos = list(event_dispatcher.image_position())
        pos[index] += self.step_sizes()[index]
        event_dispatcher.set_image_position(*pos)
      def callback_neg(index, c):
        pos = list(event_dispatcher.image_position())
        pos[index] -= self.step_sizes()[index]
        event_dispatcher.set_image_position(*pos)

      coord_inc_button = ttk.Button(self.relative_frame, text=f'+{coord.upper()}', command=partial(callback_pos, i, coord))
      coord_dec_button = ttk.Button(self.relative_frame, text=f'-{coord.upper()}', command=partial(callback_neg, i, coord))

      coord_inc_button.grid(row=0, column=i)
      coord_dec_button.grid(row=1, column=i)

      self.lockable_widgets.append(coord_inc_button)
      self.lockable_widgets.append(coord_dec_button)
      self.lockable_widgets.append(self.position_intputs[i].widget)
      self.lockable_widgets.append(self.step_size_intputs[i].widget)
    self.lockable_widgets.append(self.set_position_button)
    
    ttk.Label(self.relative_frame, text='Step Size (pixels, pixels, degrees)', anchor='center').grid(row=2, column=0, columnspan=3, sticky='ew')

    def on_position_change():
      pos = event_dispatcher.image_adjust_position
      for i in range(3):
        self.position_intputs[i].set(pos[i])

    event_dispatcher.add_event_listener(Event.ImageAdjustChanged, on_position_change)

    def on_lock_change():
      lock = event_dispatcher.movement_lock()
      if event_dispatcher.movement_lock() == MovementLock.Unlocked:
        for w in self.lockable_widgets:
          w.configure(state='normal')
      else:
        for w in self.lockable_widgets:
          w.configure(state='disabled')
    event_dispatcher.add_event_listener(Event.MovementLockChanged, on_lock_change)
  
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
      if path == '':
        return ''
      else:
        return Path(path).name

    self.pattern_frame   = ImageSelectFrame(
      self.frame, 'Pattern',
      lambda t: event_dispatcher.set_pattern_image(self.pattern_image(), get_name(self.pattern_frame.thumb.path)),
    )
    self.red_focus_frame = ImageSelectFrame(
      self.frame, 'Red Focus',
      lambda t: event_dispatcher.set_red_focus_image(self.red_focus_image()),
    )
    self.uv_focus_frame = ImageSelectFrame(
      self.frame, 'UV Focus',
      lambda t: event_dispatcher.set_uv_focus_image(self.uv_focus_image()),
    )

    self.pattern_frame.frame.grid(row=0, column=0)
    self.red_focus_frame.frame.grid(row=1, column=0)
    self.uv_focus_frame.frame.grid(row=1, column=1)

    event_dispatcher.add_event_listener(Event.ShownImageChanged, lambda: self.highlight_button(event_dispatcher.shown_image))
  
  def pattern_image(self):
    return self.pattern_frame.thumb.image
  
  def red_focus_image(self):
    return self.red_focus_frame.thumb.image

  def uv_focus_image(self):
    return self.uv_focus_frame.thumb.image
  
  def highlight_button(self, which: ShownImage):
    # TODO:
    pass

class ExposureFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.frame.columnconfigure(0, weight=1)

    ttk.Label(self.frame, text='Exposure Time (ms)', anchor='w').grid(row=0, column=0)
    self.exposure_time_entry = IntEntry(self.frame, default=8000, min_value=0)
    self.exposure_time_entry.widget.grid(row=0, column=1, columnspan=2, sticky='nesw')

    def on_exposure_time_change(_a, _b, _c):
      event_dispatcher.exposure_time = self.exposure_time_entry._var.get()
    self.exposure_time_entry._var.trace_add('write', on_exposure_time_change)

    # Posterization

    def on_posterize_change(*args):
      event_dispatcher.set_posterize_strength(self._posterize_strength())

    def on_posterize_check():
      if self.posterize_enable_var.get():
        self.posterize_scale['state'] = 'normal'
        self.posterize_cutoff_entry.widget['state'] = 'normal'
      else:
        self.posterize_scale['state'] = 'disabled'
        self.posterize_cutoff_entry.widget['state'] = 'disabled'
      on_posterize_change()

    self.posterize_enable_var = BooleanVar()
    self.posterize_checkbutton = ttk.Checkbutton(
      self.frame, text='Posterize Cutoff (%)', command=on_posterize_check,
      variable=self.posterize_enable_var, onvalue=True, offvalue=False
    )
    self.posterize_checkbutton.grid(row=2, column=0)
    self.posterize_strength_var = IntVar()
    self.posterize_strength_var.trace_add('write', on_posterize_change)
    self.posterize_scale = ttk.Scale(self.frame, variable=self.posterize_strength_var, from_=0.0, to=100.0)
    self.posterize_scale.grid(row=2, column=1, sticky='nesw')
    self.posterize_scale['state'] = 'disabled'
    self.posterize_cutoff_entry = IntEntry(self.frame, var=self.posterize_strength_var, default=50, min_value=0, max_value=100)
    self.posterize_cutoff_entry.widget.grid(row=2, column=2, sticky='nesw')
    self.posterize_cutoff_entry.widget['state'] = 'disabled'

  # returns threshold percentage if posterizing is enabled, else None
  def _posterize_strength(self) -> Optional[int]:
    if self.posterize_enable_var.get():
      return self.posterize_cutoff_entry.get()
    else:
      return None

class PatterningFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.preview_tile = ttk.Label(self.frame, text='Next Pattern Tile', compound='top') # type:ignore
    self.preview_tile.grid(row=0, column=0)

    self.begin_patterning_button = ttk.Button(self.frame, text='Begin Patterning', command=lambda: event_dispatcher.begin_patterning(), state='enabled')
    self.begin_patterning_button.grid(row=1, column=0)

    self.abort_patterning_button = ttk.Button(self.frame, text='Abort Patterning', command=lambda: event_dispatcher.abort_patterning(), state='disabled')
    self.abort_patterning_button.grid(row=2, column=0)

    ttk.Label(self.frame, text='Exposure Progress', anchor='s').grid(row=3, column=0)
    self.exposure_progress = Progressbar(self.frame, orient='horizontal', mode='determinate', maximum=1000)
    self.exposure_progress.grid(row=4, column = 0, sticky='ew')

    self.set_image(Image.new('RGB', (1, 1)))

    def on_change_patterning_status():
      if event_dispatcher.patterning_busy:
        self.begin_patterning_button['state'] = 'disabled'
        self.abort_patterning_button['state'] = 'normal'
      else:
        self.begin_patterning_button['state'] = 'normal'
        self.abort_patterning_button['state'] = 'disabled'

    event_dispatcher.add_event_listener(Event.PatterningBusyChanged, on_change_patterning_status)
    event_dispatcher.add_event_listener(Event.PatternImageChanged, lambda: self.set_image(event_dispatcher.pattern.processed()))

    def on_progress_changed():
      self.exposure_progress['value'] = event_dispatcher.exposure_progress * 1000.0

    event_dispatcher.add_event_listener(Event.ExposurePatternProgressChanged, on_progress_changed)

  def set_image(self, img: Image.Image):
    # TODO: What is the correct size?
    self.thumb_image = image_to_tk_image(img.resize(THUMBNAIL_SIZE))
    self.preview_tile.configure(image=self.thumb_image) # type:ignore

class RedModeFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent, name='redmodeframe')

    self.red_select_var = StringVar(value='focus')
    self.red_focus_rb = ttk.Radiobutton(self.frame, variable=self.red_select_var, text='Red Focus', value=RedFocusSource.Image.value)
    self.solid_red_rb = ttk.Radiobutton(self.frame, variable=self.red_select_var, text='Solid Red', value=RedFocusSource.Solid.value)
    self.pattern_rb   = ttk.Radiobutton(self.frame, variable=self.red_select_var, text='Same as Pattern', value=RedFocusSource.Pattern.value)

    self.red_focus_rb.grid(row=0, column=0)
    self.solid_red_rb.grid(row=1, column=0)
    self.pattern_rb  .grid(row=2, column=0)

    def on_radiobutton(*_):
      print(f'red select var {self.red_select_var.get()}')
      for s in RedFocusSource:
        if s.value == self.red_select_var.get():
          event_dispatcher.set_red_focus_source(s)
          break
      else:
        raise Exception()

    self.red_select_var.trace_add('write', on_radiobutton)

class UvModeFrame:
  def __init__(self, parent, event_dispatcher):
    self.frame = ttk.Frame(parent, name='uvmodeframe')
    self.exposure_frame = ExposureFrame(self.frame, event_dispatcher)
    self.exposure_frame.frame.grid(row=0, column=0)
    self.patterning_frame = PatterningFrame(self.frame, event_dispatcher)
    self.patterning_frame.frame.grid(row=0, column=1)

class ModeSelectFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.notebook = ttk.Notebook(parent)

    self.red_mode_frame = RedModeFrame(self.notebook, event_dispatcher)
    self.notebook.add(self.red_mode_frame.frame, text='Red Mode')
    self.uv_mode_frame = UvModeFrame(self.notebook, event_dispatcher)
    self.notebook.add(self.uv_mode_frame.frame, text='UV Mode')

    def on_tab_change():
      if self._current_tab() == 'uv':
        event_dispatcher.enter_uv_mode()
      else:
        event_dispatcher.enter_red_mode()
    self.notebook.bind('<<NotebookTabChanged>>', lambda _: on_tab_change())
   
    #def on_tab_event(evt):
    #  self.notebook.select(1 if evt == Event.EnterUvMode else 0)

    #event_dispatcher.add_event_listener(Event.EnterRedMode, lambda: on_tab_event(Event.EnterRedMode))
    #event_dispatcher.add_event_listener(Event.EnterUvMode, lambda: on_tab_event(Event.EnterUvMode))

  def _current_tab(self):
    if 'redmode' in self.notebook.select():
      return 'red'
    else:
      return 'uv'
 
class GlobalSettingsFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.LabelFrame(parent, text='Global Settings')

    def set_value(*_):
      event_dispatcher.autofocus_on_mode_switch = self.autofocus_on_mode_switch_var.get()
    self.autofocus_on_mode_switch_var = BooleanVar(value=True)
    self.autofocus_on_mode_switch_check = ttk.Checkbutton(
      self.frame,
      text='Autofocus on Mode Change',
      variable=self.autofocus_on_mode_switch_var
    )
    self.autofocus_on_mode_switch_check.grid(row=0, column=0, columnspan=2)

    self.autofocus_on_mode_switch_var.trace_add('write', set_value)

    self.autofocus_button = ttk.Button(self.frame, text='Autofocus', command=lambda: event_dispatcher.autofocus())
    self.autofocus_button.grid(row=1, column=0, columnspan=2, sticky='ew')

    # Maybe this should have a scale?
    # Or, even further, maybe this should just be the same as the interface for posterization strength?
    self.border_size_var = IntVar()
    self.border_label = ttk.Label(self.frame, text='Border Size (%)')
    self.border_label.grid(row=2, column=0)
    self.border_entry = IntEntry(self.frame, var=self.border_size_var, default=0, min_value=0, max_value=100)
    self.border_entry.widget.grid(row=2, column=1, sticky='nesw')

    def on_border_size_change(*_):
      event_dispatcher.set_border_size(self.border_size_var.get())
    self.border_size_var.trace_add('write', on_border_size_change)

    self.placeholder_photo = image_to_tk_image(Image.new('RGB', THUMBNAIL_SIZE, 'black'))
    self.photo = None

    self.current_image = ttk.Label(self.frame, image=self.placeholder_photo) # type:ignore
    self.current_image.grid(row=3, column=0, columnspan=2)

    # Disable the autofocus button if autofocus is already running
    def movement_lock_changed():
      if event_dispatcher.movement_lock() == MovementLock.Locked:
        self.autofocus_button.configure(state='disabled')
      else:
        self.autofocus_button.configure(state='normal')
    event_dispatcher.add_event_listener(Event.MovementLockChanged, movement_lock_changed)

    def shown_image_changed():
      img = event_dispatcher.current_image()
      if img is None:
        self.current_image.configure(image=self.placeholder_photo) # type:ignore
      else:
        photo = image_to_tk_image(img.resize(THUMBNAIL_SIZE, Image.Resampling.NEAREST))
        self.current_image.configure(image=photo) # type:ignore
        self.photo = photo

    event_dispatcher.add_event_listener(Event.ShownImageChanged, shown_image_changed)
  

class ExposureHistoryFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.LabelFrame(parent, text='Exposure History')
    self.text = tkinter.Text(self.frame, width=80, height=10, wrap='none', state='disabled')
    self.text.grid(row=0, column=0)
    
    self.event_dispatcher = event_dispatcher

    event_dispatcher.add_event_listener(Event.PatterningBusyChanged, lambda: self._refresh())
  
  def _refresh(self):
    self.text['state'] = 'normal'
    self.text.delete('1.0', 'end')
    for exp_log in self.event_dispatcher.exposure_history[-10:]:
      t = exp_log.time.strftime('%H:%M:%S')
      line = f'{t} {exp_log.name} {int(exp_log.duration)}ms X:{exp_log.coords[0]} Y:{exp_log.coords[1]} Z:{exp_log.coords[2]}\n'

      if self.text.index('end-1c') != 1.0:
        self.text.insert('end', '\n')
      self.text.insert('end', line)
    self.text['state'] = 'disabled'

class LithographerGui:
  root: Tk

  #flatfield_image: ProcessedImage

  event_dispatcher: EventDispatcher

  def __init__(self, stage: StageController, camera: Optional[CameraModule], title='Lithographer'):
    self.root = Tk()

    self.event_dispatcher = EventDispatcher(stage, TkProjector(self.root), self.root, camera is not None, True)

    self.shown_image = ShownImage.Clear

    self.camera = CameraFrame(self.root, self.event_dispatcher, camera)
    self.camera.frame.grid(row=0, column=0)

    self.middle_panel = ttk.Frame(self.root)
    self.middle_panel.grid(row=2, column=0)

    self.bottom_panel = ttk.Frame(self.root)
    self.bottom_panel.grid(row=3, column=0)

    self.exposure_history_frame = ExposureHistoryFrame(self.bottom_panel, self.event_dispatcher)
    self.exposure_history_frame.frame.grid(row=0, column=0)

    self.image_adjust_frame = ImageAdjustFrame(self.bottom_panel, self.event_dispatcher)
    self.image_adjust_frame.frame.grid(row=0, column=1)

    self.global_settings_frame = GlobalSettingsFrame(self.bottom_panel, self.event_dispatcher)
    self.global_settings_frame.frame.grid(row=0, column=2)

    self.stage_position_frame = StagePositionFrame(self.middle_panel, self.event_dispatcher)
    self.stage_position_frame.frame.grid(row=0, column=1)

    self.mode_select_frame = ModeSelectFrame(self.middle_panel, self.event_dispatcher)
    self.mode_select_frame.notebook.grid(row=0, column=2)

    self.exposure_frame = self.mode_select_frame.uv_mode_frame.exposure_frame
    self.patterning_frame = self.mode_select_frame.uv_mode_frame.patterning_frame

    self.pattern_progress = Progressbar(self.root, orient='horizontal', mode='determinate')
    self.pattern_progress.grid(row = 1, column = 0, sticky='ew')

    self.multi_image_select_frame = MultiImageSelectFrame(self.middle_panel, self.event_dispatcher)
    self.multi_image_select_frame.frame.grid(row=0, column=0)

    self.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())
    #self.debug.info("Debug info will appear here")

    messagebox.showinfo(message='BEFORE CONTINUING: Ensure that you move the projector window to the correct display! Click on the fullscreen, completely black window, then press Windows Key + Shift + Left Arrow until it no longer is visible!')
  
  def cleanup(self):
    print("Patterning GUI closed.")
    print('TODO: Cleanup')
    self.camera.cleanup()
    self.root.destroy()
    #if RUN_WITH_STAGE:
      #serial_port.close()


def main():
  with open('config.toml', 'r') as f:
    config = toml.load(f)

  stage_config = config['stage']
  if stage_config['enabled']:
    serial_port = serial.Serial(stage_config['port'], stage_config['baud-rate'])
    print(f'Using serial port {serial_port.name}')
    stage = GrblStage(serial_port, stage_config['scale-factor'], bounds=((-12000,12000),(-12000,12000),(-12000,12000))) 
  else:
    stage = StageController()
  
  camera_config = config['camera']
  flir_enabled = camera_config.get('flir', False)
  webcam_enabled = 'webcam' in camera_config
  if webcam_enabled and flir_enabled:
    print('config.toml cannot enable both the FLIR camera and a webcam!')
    print('Disable the webcam by #commenting it out or set `flir = false`')
    return 1
  elif webcam_enabled:
    num = camera_config['webcam']
    print(f'Using webcam {num}')
    camera = Webcam(num)
  elif flir_enabled:
    import camera.flir.flir_camera as flir 
    print('Using FLIR camera')
    camera = flir.FlirCamera()
  else:
    print('No camera enabled')
    camera = None

  lithographer = LithographerGui(stage, camera)
  lithographer.root.mainloop()

main()
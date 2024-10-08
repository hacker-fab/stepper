import serial
import tomllib
import cv2
import numpy as np
import time
import queue
from datetime import datetime
from functools import partial
from hardware import Lithographer, ImageProcessSettings, StageWrapper
from typing import Callable, List, Optional
from PIL import Image
from camera.camera_module import CameraModule
from camera.webcam import Webcam
from stage_control.stage_controller import StageController
from stage_control.grbl_stage import GrblStage
from projector import ProjectorController, TkProjector
from enum import Enum
from lithographer_lib.gui_lib import IntEntry, Thumbnail
from lithographer_lib.img_lib import image_to_tk_image, fit_image
from tkinter.ttk import Progressbar
from tkinter import ttk, Tk, BooleanVar, IntVar, StringVar

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

  EnterRedMode = 'enterredmode'
  EnterUvMode = 'enteruvmode'

  ShownImageChanged = 'shownimagechanged'
  StagePositionChanged = 'stagepositionchanged'
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

class EventDispatcher:
  hardware: Lithographer

  listeners: dict[Event, List[Callable]]

  exposure_time: int
  posterize_strength: Optional[int]
  patterning_progress: float # ranges from 0.0 to 1.0

  shown_image: ShownImage
  autofocus_busy: bool
  patterning_busy: bool

  pattern_image: Image.Image
  red_focus_image: Image.Image
  uv_focus_image: Image.Image

  def __init__(self, stage, proj, root):
    self.listeners = dict()

    self.hardware = Lithographer(stage, proj) # TODO:

    self.root = root

    self.posterize_strength = None

    self.shown_image = ShownImage.Clear
    self.autofocus_busy = False

  def _update_projector(self):
    match self.shown_image:
      case ShownImage.Clear:
        self.hardware.projector.clear()
      case ShownImage.RedFocus:
        # TODO:
        pass

  def _refresh_pattern(self):
    self.hardware.pattern.update(image=self.pattern_image, settings=ImageProcessSettings(
      posterization=self.posterize_strength,
      flatfield=None,
      color_channels=(False, False, True),
      size=self.hardware.projector.size()
    ))

    self.on_event(Event.PatternImageChanged)

  def _refresh_red_focus(self):
    self.hardware.red_focus.update(image=self.red_focus_image, settings=ImageProcessSettings(
      posterization=self.posterize_strength,
      flatfield=None,
      color_channels=(True, False, False),
      size=self.hardware.projector.size()
    ))

    if self.shown_image == ShownImage.RedFocus:
      self.on_event(Event.ShownImageChanged)
  
  def _refresh_uv_focus(self):
    # TODO:

    if self.shown_image == ShownImage.UvFocus:
      self.on_event(Event.ShownImageChanged)
  
  def set_posterize_strength(self, strength):
    self.posterize_strength = strength
    self._refresh_red_focus()
    self._refresh_uv_focus()
    self._refresh_pattern()
  
  def set_shown_image(self, shown_image: ShownImage):
    self.shown_image = shown_image
    self.on_event(Event.ShownImageChanged)
  
  def set_stage_position(self, x: float, y: float, z: float):
    print('TODO: Actually set stage position')
    self.hardware.stage.move_to({ 'x': x, 'y': y, 'z': z }, commit=True)
    self.on_event(Event.StagePositionChanged)
  
  def set_pattern_image(self, img: Image.Image):
    self.pattern_image = img
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
  
  def abort_patterning(self):
    self.should_abort = True
    print('Aborting patterning')
  
  def stage_position(self):
    pos = self.hardware.stage.stage_position
    return (pos['x'], pos['y'], pos['z'])
    
  def movement_lock(self):
    if self.patterning_busy:
      return MovementLock.Locked
    elif self.autofocus_busy:
      return MovementLock.XYLocked
    elif self.shown_image == ShownImage.UvFocus or self.shown_image == ShownImage.Pattern:
      return MovementLock.XYLocked
    else:
      return MovementLock.Unlocked

  def on_event(self, event: Event, *args, **kwargs):
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

    def update_func(exposure_progress):
      self.set_progress(0.0, exposure_progress)
      self.root.update()
      return self.should_abort
    
    duration = self.exposure_time
    
    print(f"Patterning 1 tiles for {duration}ms\nTotal time: {str(round((duration)/1000))}s")

    self.set_patterning_busy(True)
    self.set_progress(0.0, 0.0)
    self.hardware.do_pattern(0, 0, duration, update_func=update_func)
    self.set_progress(1.0, 1.0)
    self.set_patterning_busy(False)

    if self.should_abort:
      print('Patterning aborted')
      self.should_abort = False






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
  def __init__(self, parent, event_dispatcher: EventDispatcher, c: CameraModule):
    self.frame = ttk.Frame(parent)
    self.label = ttk.Label(self.frame, text='live hackerfab reaction')
    self.label.grid(row=0, column=0)
    
    self.snapshot = SnapshotFrame(self.frame, event_dispatcher)
    self.snapshot.frame.grid(row=1, column=0)

    self.image_focus = 0

    self.snapshots_pending = queue.Queue()
    event_dispatcher.add_event_listener(Event.Snapshot, lambda filename: self.snapshots_pending.put(filename))

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

    if not self.camera.open():
      print('Camera failed to start')
    else:
      self.camera.setSetting('image_format', "rgb888")
      self.camera.setStreamCaptureCallback(cameraCallback)
      if not self.camera.startStreamCapture():
        print('Failed to start stream capture for camera')
  
  def cleanup(self):
    self.camera.close()

  def gui_camera_preview(self, camera_image, dimensions):
    resized_img = cv2.resize(camera_image, (0, 0), fx=0.25, fy=0.25)

    start_time = time.time()
    img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    mean = np.sum(img) / (img.shape[0] * img.shape[1])
    img_lapl = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1) / mean
    self.image_focus = img_lapl.var() / mean
    end_time = time.time()
    #print(f'Took {(end_time - start_time)*1000}ms to process')

    self.gui_img = image_to_tk_image(Image.fromarray(resized_img, mode='RGB'))
    self.label.configure(image=self.gui_img) # type:ignore

class StagePositionFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.position_intputs = []
    self.step_size_intputs = []

    self.xy_widgets = []
    self.z_widgets = []

    # Absolute

    self.absolute_frame = ttk.LabelFrame(self.frame, text='Position')
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
        self.xy_widgets.append(self.position_intputs[i].widget)
        self.xy_widgets.append(self.step_size_intputs[i].widget)

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
            w.configure(state='disabled')
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

class ImageSelectFrame:
  def __init__(self, parent, button_text, import_command, select_command):
    self.frame = ttk.Frame(parent)

    self.thumb = Thumbnail(self.frame, THUMBNAIL_SIZE, on_import=import_command)
    self.thumb.widget.grid(row=0, column=0)

    self.select_button = ttk.Button(self.frame, text=button_text, command=select_command)
    self.select_button.grid(row=1, column=0)

class MultiImageSelectFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    def show_cb(img):
      def cb():
        event_dispatcher.set_shown_image(img)
      return cb

    self.pattern_frame   = ImageSelectFrame(
      self.frame, 'Show pattern',
      lambda t: event_dispatcher.set_pattern_image(self.pattern_image()),
      show_cb(ShownImage.Pattern)
    )
    self.red_focus_frame = ImageSelectFrame(
      self.frame, 'Show Red Focus',
      lambda t: event_dispatcher.set_red_focus_image(self.red_focus_image()),
      show_cb(ShownImage.RedFocus)
    )
    self.uv_focus_frame = ImageSelectFrame(
      self.frame, 'Show UV Focus',
      lambda t: event_dispatcher.set_uv_focus_image(self.uv_focus_image()),
      show_cb(ShownImage.UvFocus)
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

  def set_image(self, img: Image.Image):
    # TODO: What is the correct size?
    self.thumb_image = image_to_tk_image(img.resize(THUMBNAIL_SIZE))
    self.preview_tile.configure(image=self.thumb_image) # type:ignore

class RedModeFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent, name='redmodeframe')

class UvModeFrame:
  def __init__(self, parent, event_dispatcher):
    self.frame = ttk.Frame(parent, name='uvmodeframe')
    self.position_label = ttk.Label(self.frame, text='Stage Position Goes Here')
    self.position_label.grid(row=0, column=0)
    self.exposure_frame = ExposureFrame(self.frame, event_dispatcher)
    self.exposure_frame.frame.grid(row=0, column=1)
    self.patterning_frame = PatterningFrame(self.frame, event_dispatcher)
    self.patterning_frame.frame.grid(row=0, column=2)

class ModeSelectFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.notebook = ttk.Notebook(parent)

    self.red_mode_frame = RedModeFrame(self.notebook, event_dispatcher)
    self.notebook.add(self.red_mode_frame.frame, text='Red Mode', )
    self.uv_mode_frame = UvModeFrame(self.notebook, event_dispatcher)
    self.notebook.add(self.uv_mode_frame.frame, text='UV Mode')

    def on_tab_change():
      event_dispatcher.on_event(self._current_tab())
    self.notebook.bind('<<NotebookTabChanged>>', lambda _: on_tab_change())
   
    def on_tab_event(evt):
      self.notebook.select(1 if evt == Event.EnterUvMode else 0)

    event_dispatcher.add_event_listener(Event.EnterRedMode, lambda: on_tab_event(Event.EnterRedMode))
    event_dispatcher.add_event_listener(Event.EnterUvMode, lambda: on_tab_event(Event.EnterUvMode))

  def _current_tab(self):
    if 'redmode' in self.notebook.select():
      return Event.EnterRedMode
    else:
      return Event.EnterUvMode

class LithographerGui:
  root: Tk

  #flatfield_image: ProcessedImage

  event_dispatcher: EventDispatcher

  def __init__(self, stage: StageController, camera, title='Lithographer'):
    self.root = Tk()

    self.event_dispatcher = EventDispatcher(stage, TkProjector(self.root), self.root)

    self.shown_image = ShownImage.Clear

    self.camera = CameraFrame(self.root, self.event_dispatcher, camera)
    self.camera.frame.grid(row=0, column=0)

    self.bottom_panel = ttk.Frame(self.root)
    self.bottom_panel.grid(row=2, column=0)

    self.stage_position_frame = StagePositionFrame(self.bottom_panel, self.event_dispatcher)
    self.stage_position_frame.frame.grid(row=0, column=1)

    self.mode_select_frame = ModeSelectFrame(self.bottom_panel, self.event_dispatcher)
    self.mode_select_frame.notebook.grid(row=0, column=2)

    self.exposure_frame = self.mode_select_frame.uv_mode_frame.exposure_frame
    self.patterning_frame = self.mode_select_frame.uv_mode_frame.patterning_frame

    self.pattern_progress = Progressbar(self.root, orient='horizontal', mode='determinate')
    self.pattern_progress.grid(row = 1, column = 0, sticky='ew')

    self.multi_image_select_frame = MultiImageSelectFrame(self.bottom_panel, self.event_dispatcher)
    self.multi_image_select_frame.frame.grid(row=0, column=0)

    self.patterning_status = PatterningStatus.Idle

    self.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())
    #self.debug.info("Debug info will appear here")
  
  def cleanup(self):
    print("Patterning GUI closed.")
    print('TODO: Cleanup')
    self.camera.cleanup()
    self.root.destroy()
    #if RUN_WITH_STAGE:
      #serial_port.close()

  def autofocus(self):
    self.hardware.stage.move_by({ 'z': -100.0 })


import camera.amscope.amscope_camera as amscope_camera
#import camera.flir.flir_camera as flir 


def main():
  with open('default.toml', 'rb') as f:
    config = tomllib.load(f)

  stage_config = config['stage']
  if stage_config['enabled']:
    serial_port = serial.Serial(stage_config['port'], stage_config['baud-rate'])
    print(f'Using serial port {serial_port.name}')
    stage = GrblStage(serial_port, stage_config['scale-factor'], bounds=((-12000,12000),(-12000,12000),(-12000,12000))) 
  else:
    stage = StageController()

  camera_config = config['camera']
  if camera_config['enabled']:
    # TODO: Why doesn't this import properly?
    #lithographer = LithographerGui(stage, flir.FlirCamera())
    pass

  lithographer = LithographerGui(stage, Webcam())
  lithographer.root.mainloop()

main()
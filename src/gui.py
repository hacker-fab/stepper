import serial
import tomllib
from hardware import Lithographer, ImageProcessSettings
from typing import Callable, List, Optional
from PIL import Image
from stage_control.stage_controller import StageController
from stage_control.grbl_stage import GrblStage
from projector import ProjectorController, TkProjector
from enum import Enum
from lithographer_lib.gui_lib import IntEntry, Thumbnail
from lithographer_lib.img_lib import image_to_tk_image
from tkinter.ttk import Progressbar
from tkinter import ttk, Tk, BooleanVar, IntVar

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
  StageOffsetRequest = 'stageoffsetrequest'
  StageSetRequest = 'stagesetrequest'
  ImportPattern = 'importpattern'
  ImportRedFocus = 'importredfocus'
  PosterizeChange = 'posterizechange'
  BeginPatterning = 'beginpatterning'
  AbortPatterning = 'abortpatterning'
  ChangePatterningStatus = 'changepatterningstatus'

class EventDispatcher:
  shown_image_change_listeners: List[OnShownImageChange]

  listeners: dict[Event, List[Callable]]

  def __init__(self):
    self.shown_image_change_listeners = []
    self.change_patterning_status_listeners = []
    self.listeners = dict()

  def on_event(self, event: Event, *args, **kwargs):
    for l in self.listeners[event]:
      l(*args, **kwargs)
  
  def on_event_cb(self, event: Event, *args, **kwargs):
    return lambda: self.on_event(event, *args, **kwargs) 
  
  def add_event_listener(self, event: Event, listener: Callable):
    if event not in self.listeners:
      self.listeners[event] = []
    self.listeners[event].append(listener)

  def on_shown_image_change_cb(self, shown_image: ShownImage):
    def callback():
      for l in self.shown_image_change_listeners:
        l(shown_image)
    return callback

  def add_shown_image_change_listener(self, listener: OnShownImageChange):
    self.shown_image_change_listeners.append(listener)

class StagePositionFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.position_intputs = []
    self.step_size_intputs = []

    for i, coord in ((0, 'x'), (1, 'y'), (2, 'z')):
      self.position_intputs.append(IntEntry(parent=self.frame, default=0))
      self.position_intputs[-1].widget.grid(row=0,column=i)

      self.step_size_intputs.append(IntEntry(parent=self.frame, default=10, min_value=-1000, max_value=1000))
      self.step_size_intputs[-1].widget.grid(row=5,column=i)

      def callback_pos():
        event_dispatcher.on_event(Event.StageOffsetRequest, { coord:  self.step_sizes()[i] })
      def callback_neg():
        event_dispatcher.on_event(Event.StageOffsetRequest, { coord: -self.step_sizes()[i] })

      coord_inc_button = ttk.Button(self.frame, text=f'+{coord.upper()}', command=callback_pos)
      coord_dec_button = ttk.Button(self.frame, text=f'-{coord.upper()}', command=callback_neg)

      coord_inc_button.grid(row=1, column=i)
      coord_dec_button.grid(row=2, column=i)

    def callback_set():
      x, y, z = self.position()
      event_dispatcher.on_event(Event.StageSetRequest, { 'x': x, 'y': y, 'z': z })

    set_position_button = ttk.Button(self.frame, text='Set Stage Position', command=callback_set)
    set_position_button.grid(row=3, column=0, columnspan=3, sticky='ew')

    ttk.Label(self.frame, text='Stage Step Size (microns)', anchor='center').grid(row=4, column=0, columnspan=3, sticky='ew')
  
  def position(self) -> tuple[int, int, int]:
    return tuple(intput.get() for intput in self.position_intputs)
  
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

    # TODO: Update slicer on pattern upload
    self.pattern_frame = ImageSelectFrame(self.frame, 'Show pattern', lambda t: event_dispatcher.on_event(Event.ImportPattern), event_dispatcher.on_shown_image_change_cb(ShownImage.Pattern))
    self.red_focus_frame = ImageSelectFrame(self.frame, 'Show Red Focus', lambda t: event_dispatcher.on_event(Event.ImportRedFocus), event_dispatcher.on_shown_image_change_cb(ShownImage.RedFocus))
    self.uv_focus_frame = ImageSelectFrame(self.frame, 'Show UV Focus', None, event_dispatcher.on_shown_image_change_cb(ShownImage.UvFocus))

    self.pattern_frame.frame.grid(row=0, column=0)
    self.red_focus_frame.frame.grid(row=1, column=0)
    self.uv_focus_frame.frame.grid(row=1, column=1)

    event_dispatcher.add_shown_image_change_listener(lambda img: self.highlight_button(img))
  
  def pattern_image(self):
    return self.pattern_frame.thumb.image
  
  def red_focus_image(self):
    return self.red_focus_frame.thumb.image
  
  def highlight_button(self, which: ShownImage):
    # TODO:
    pass

class ExposureFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.frame.columnconfigure(0, weight=1)

    ttk.Label(self.frame, text='Exposure Time (ms)', anchor='w').grid(row=0, column=0)
    self.exposure_time_entry = IntEntry(self.frame, default=1000, min_value=0)
    self.exposure_time_entry.widget.grid(row=0, column=1, columnspan=2, sticky='nesw')

    # Posterization

    def on_posterize_change(*args):
      event_dispatcher.on_event(Event.PosterizeChange)

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
  def posterize_strength(self) -> Optional[int]:
    if self.posterize_enable_var.get():
      return self.posterize_cutoff_entry.get()
    else:
      return None

class PatterningFrame:
  def __init__(self, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    self.preview_tile = ttk.Label(self.frame, text='Next Pattern Tile', compound='top') # type:ignore
    self.preview_tile.grid(row=0, column=0)

    self.begin_patterning_button = ttk.Button(self.frame, text='Begin Patterning', command=event_dispatcher.on_event_cb(Event.BeginPatterning), state='enabled')
    self.begin_patterning_button.grid(row=1, column=0)

    self.abort_patterning_button = ttk.Button(self.frame, text='Abort Patterning', command=event_dispatcher.on_event_cb(Event.AbortPatterning), state='disabled')
    self.abort_patterning_button.grid(row=2, column=0)

    ttk.Label(self.frame, text='Exposure Progress', anchor='s').grid(row=3, column=0)
    self.exposure_progress = Progressbar(self.frame, orient='horizontal', mode='determinate', maximum=1000)
    self.exposure_progress.grid(row=4, column = 0, sticky='ew')

    self.set_image(Image.new('RGB', (1, 1)))

    def on_change_patterning_status(status: PatterningStatus):
      if status == PatterningStatus.Idle:
        self.begin_patterning_button['state'] = 'normal'
        self.abort_patterning_button['state'] = 'disabled'
      elif status == PatterningStatus.Patterning:
        self.begin_patterning_button['state'] = 'disabled'
        self.abort_patterning_button['state'] = 'normal'

    event_dispatcher.add_event_listener(Event.ChangePatterningStatus, on_change_patterning_status)

  def set_image(self, img: Image.Image):
    # TODO: What is the correct size?
    self.thumb_image = image_to_tk_image(img.resize(THUMBNAIL_SIZE))
    self.preview_tile.configure(image=self.thumb_image) # type:ignore
  
class LithographerGui:
  root: Tk

  hardware: Lithographer

  #flatfield_image: ProcessedImage

  event_dispatcher: EventDispatcher

  def __init__(self, stage: StageController, title='Lithographer'):
    self.root = Tk()

    self.event_dispatcher = EventDispatcher()
    self.event_dispatcher.add_shown_image_change_listener(lambda img: self.on_shown_image_change(img))
    self.event_dispatcher.add_event_listener(Event.BeginPatterning, lambda: self.begin_patterning())
    self.event_dispatcher.add_event_listener(Event.AbortPatterning, lambda: self.change_patterning_status(PatterningStatus.Aborting))
    self.event_dispatcher.add_event_listener(Event.ChangePatterningStatus, lambda status: self.on_change_patterning_status(status))

    self.event_dispatcher.add_event_listener(Event.ImportPattern, lambda: self.refresh_pattern())
    self.event_dispatcher.add_event_listener(Event.ImportRedFocus, lambda: self.refresh_red_focus())
    self.event_dispatcher.add_event_listener(Event.PosterizeChange, lambda: self.refresh_pattern())
    self.event_dispatcher.add_event_listener(Event.PosterizeChange, lambda: self.refresh_red_focus())

    self.shown_image = ShownImage.Clear

    self.hardware = Lithographer(stage, TkProjector(self.root))

    self.event_dispatcher.add_event_listener(
      Event.StageOffsetRequest,
      lambda offsets: self.hardware.stage.move_by(offsets, commit=True)
    )

    self.event_dispatcher.add_event_listener(
      Event.StageSetRequest,
      lambda coords: self.hardware.stage.move_to(coords, commit=True)
    )

    self.settings_notebook = ttk.Notebook(self.root)
    self.stage_position_frame = StagePositionFrame(self.settings_notebook, self.event_dispatcher)
    self.settings_notebook.add(self.stage_position_frame.frame, text='Stage')
    self.exposure_frame = ExposureFrame(self.settings_notebook, self.event_dispatcher)
    self.settings_notebook.add(self.exposure_frame.frame, text='Exposure')

    self.settings_notebook.grid(row=3, column=1)

    self.pattern_progress = Progressbar(self.root, orient='horizontal', mode='determinate')
    self.pattern_progress.grid(row = 1, column = 0, sticky='ew')

    self.multi_image_select_frame = MultiImageSelectFrame(self.root, self.event_dispatcher)
    self.multi_image_select_frame.frame.grid(row=3, column=0)

    self.patterning_frame = PatterningFrame(self.root, self.event_dispatcher)
    self.patterning_frame.frame.grid(row=3, column = 2, sticky='nesw')

    self.patterning_status = PatterningStatus.Idle

    self.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())
    #self.debug.info("Debug info will appear here")
  
  def refresh_pattern(self):
    self.hardware.pattern.update(image=self.multi_image_select_frame.pattern_image(), settings=ImageProcessSettings(
      posterization=self.exposure_frame.posterize_strength(),
      flatfield=None,
      color_channels=(False, False, True),
      size=self.hardware.projector.size()
    ))

    self.patterning_frame.set_image(self.hardware.pattern.processed())

  def refresh_red_focus(self):
    self.hardware.red_focus.update(image=self.multi_image_select_frame.red_focus_image(), settings=ImageProcessSettings(
      posterization=self.exposure_frame.posterize_strength(),
      flatfield=None,
      color_channels=(True, False, False),
      size=self.hardware.projector.size()
    ))

    if self.shown_image == ShownImage.RedFocus:
      self.hardware.projector.show(self.hardware.red_focus.processed())
  
  def cleanup(self):
    print("Patterning GUI closed.")
    print('TODO: Cleanup')
    self.root.destroy()
    #if RUN_WITH_STAGE:
      #serial_port.close()

  def on_shown_image_change(self, which: ShownImage):
    self.shown_image = which
    match which:
      case ShownImage.Clear:
        self.hardware.projector.clear()
      case ShownImage.Pattern:
        self.hardware.projector.show(self.hardware.pattern.processed())
      #case ShownImage.Flatfield:
      #  self.hardware.projector.show(self.hardware.flatfield.processed())
      case ShownImage.RedFocus:
        self.hardware.projector.show(self.hardware.red_focus.processed())
      case ShownImage.UvFocus:
        self.hardware.projector.show(self.hardware.uv_focus.processed())
  
  def change_patterning_status(self, status: PatterningStatus):
    self.event_dispatcher.on_event(Event.ChangePatterningStatus, status)
    self.patterning_status = status

  def on_change_patterning_status(self, status: PatterningStatus, notify=False):
    self.patterning_status = status

  def begin_patterning(self):
    # TODO: Update patterning preview

    def update_func(exposure_progress):
      global GUI
      self.patterning_frame.exposure_progress['value'] = round(exposure_progress * 1000)
      self.root.update()
      return self.patterning_status == PatterningStatus.Aborting
    
    #duration = duration_intput.get() # TODO:
    # TODO: Duration
    duration = 8000
    
    self.pattern_progress['value'] = 0
    self.pattern_progress['maximum'] = 1
    print(f"Patterning 1 tiles for {duration}ms\nTotal time: {str(round((duration)/1000))}s")

    self.change_patterning_status(PatterningStatus.Patterning)
    while True:
      if self.patterning_status == PatterningStatus.Aborting:
        break
      self.patterning_frame.exposure_progress['value'] = 0
      self.hardware.do_pattern(0, 0, duration, update_func=update_func)
      self.patterning_frame.exposure_progress['value'] = 1000
      if self.patterning_status == PatterningStatus.Aborting:
        break

      self.pattern_progress['value'] += 1
      break

    # give user feedback
    self.pattern_progress['value'] = 0
    if self.patterning_status == PatterningStatus.Aborting:
      print("Patterning aborted")
    else:
      print("Done")
    self.change_patterning_status(PatterningStatus.Idle)

# attach cleanup function to GUI close event

def main():
  with open('default.toml', 'rb') as f:
    config = tomllib.load(f)

  stage_config = config['stage']
  if stage_config['enabled']:
    serial_port = serial.Serial(stage_config['port'], stage_config['baud-rate'])
    print(f'Using serial port {serial_port.name}')
    stage = GrblStage(serial_port, bounds=((-12000,12000),(-12000,12000),(-12000,12000))) 
  else:
    stage = StageController()

  camera_config = config['camera']
  if camera_config['enabled']:
    print('TODO: Use camera')
  #  setup_camera_from_py()

  lithographer = LithographerGui(stage)
  lithographer.root.mainloop()

main()
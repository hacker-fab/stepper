# Hacker Fab
# Luca Garlati, 2024
# Kent Wirant
# Main lithography script

from typing import Literal, Optional, List
from enum import Enum
from tkinter import Button, Label
from tkinter.ttk import Progressbar
from tkinter import ttk
from collections import namedtuple
from PIL import  Image
from time import sleep
from stage_control.grbl_stage import GrblStage, StageController
from hardware import Lithographer

from lithographer_lib.gui_lib import *
from lithographer_lib.img_lib import *
from lithographer_lib.backend_lib import *

# import configuration variables
from config import RUN_WITH_CAMERA, RUN_WITH_STAGE

if RUN_WITH_CAMERA:
  from config import camera as camera_hw
  import cv2

if RUN_WITH_STAGE:
  from config import stage_file, baud_rate, scale_factor
  import serial

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
OnBeginPatterning = Callable[[], None]
OnChangePatterningStatus = Callable[[PatterningStatus], None]

class Event(Enum):
  StageOffsetRequest = 'stageoffsetrequest'

class EventDispatcher:
  shown_image_change_listeners: List[OnShownImageChange]
  begin_patterning_listeners: List[OnBeginPatterning]
  change_patterning_status_listeners: List[OnChangePatterningStatus]

  listeners: dict[Event, List[Callable]]

  def __init__(self):
    self.shown_image_change_listeners = []
    self.begin_patterning_listeners = []
    self.change_patterning_status_listeners = []
    self.listeners = dict()

  def on_event(self, event: Event, *args, **kwargs):
    for l in self.listeners[event]:
      l(*args, **kwargs)
  
  def on_event_cb(self, event: Event, *args, **kwargs):
    return lambda: self.on_event(*args, **kwargs) 
  
  def add_event_listener(self, event: Event, listener: Callable):
    if event not in self.listeners:
      self.listeners[event] = []
    self.listeners[event].append(listener)

  def on_shown_image_change_cb(self, shown_image: ShownImage):
    def callback():
      for l in self.shown_image_change_listeners:
        l(shown_image)
    return callback

  def on_begin_patterning_cb(self):
    def callback():
      for l in self.begin_patterning_listeners:
        l()
    return callback
  
  def on_change_patterning_status_cb(self, status: PatterningStatus):
    def callback():
      for l in self.change_patterning_status_listeners:
        l(status)
    return callback

  def add_shown_image_change_listener(self, listener: OnShownImageChange):
    self.shown_image_change_listeners.append(listener)

  def add_begin_patterning_listener(self, listener: OnBeginPatterning):
    self.begin_patterning_listeners.append(listener)
  
  def add_change_patterning_status_listener(self, listener: OnChangePatterningStatus):
    self.change_patterning_status_listeners.append(listener)

# TODO
# - Camera Integration
#     make camera (and thumbnails ideally) auto resize images / have images fill the widget
# - CV Integration
#     Make CV control the stage and patterning
# - Stage integration
#     Make stage move with stage controls and tiling
# 
# Low Priority
# - CLI
# - add a button to show pure white image for flatfield correction
# - fix bug where flatfield pattern is reapplied on second pattern show
#     to reproduce, import flatfield and pattern, enable posterize and flatfield, press show twice
# - Make an interactive version of the help message popup, it's getting long
# - add secondary list in this file to store the modified tiled images similar to the "temp" and
#     "original" images in the thumbnail widgets. This would speed up repeat patternings
# - Add user controllable tile adjustment and continue
# - use a paste command to put the preview on a black background to represent the actual exposure. 

VERSION: str = "1.6.3"
''' Patch Notes

**Minor**
- modified repo structure; changed /litho/scripts to /src
- moved stage step factor, baud rate, and serial port to config.py

**TODO**
- Improve camera preview sizing on GUI (i.e. take up less space) 
- Make camera settings configurable
- Automatic serial port detection 
- Code refactoring
- Automatic step and expose
- UI improvements
'''

#region: setup 
THUMBNAIL_SIZE: tuple[int,int] = (160,90)
CHIN_SIZE: int = 200
GUI: GUI_Controller = GUI_Controller(grid_size = (14,11),
                                     title = "Lithographer "+VERSION,
                                     add_window_size=(0,CHIN_SIZE))
SPACER_SIZE: int = 0#GUI.window_size[0]//(GUI.grid_size[1])
# Debugger
debug: Debug = Debug(root=GUI.root)
GUI.add_widget("debug", debug)

slicer: Slicer = Slicer(tiling_pattern='snake',
                        debug=debug)

# overall pattern progress bar
pattern_progress: Progressbar = Progressbar(
  GUI.root,
  orient='horizontal',
  mode='determinate',
  )
pattern_progress.grid(
  row = 1,
  column = 0,
  columnspan = GUI.grid_size[1],
  sticky='nesw')
GUI.add_widget("pattern_progress", pattern_progress)

# Current exposure Progress
exposure_progress: Progressbar = Progressbar(
  GUI.root,
  orient='horizontal',
  mode='determinate',
  )
exposure_progress.grid(
  row = 2,
  column = 0,
  columnspan = GUI.grid_size[1],
  sticky='nesw')
GUI.proj.progressbar = exposure_progress
GUI.add_widget("exposure_progress", exposure_progress)

class StagePositionFrame:
  def __init__(self, gui, parent, event_dispatcher: EventDispatcher):
    global stage
    
    self.frame = ttk.Frame(parent)

    self.position_intputs = []
    self.step_size_intputs = []

    for i, coord, default in ((0, 'x', stage.x()), (1, 'y', stage.y()), (2, 'z', stage.z())):
      self.position_intputs.append(Intput(
        gui=GUI,
        parent=self.frame,
        name=f'{coord}_intput',
        default=int(default)
      ))
      self.position_intputs[-1].grid(row=0,col=i)

      self.step_size_intputs.append(Intput(
          gui=gui,
          parent=self.frame,
          name=f'{coord}_step_intput',
          default=10,
          min=-1000,
          max=1000
      ))
      self.step_size_intputs[-1].grid(row=5,col=i)

      def callback_pos():
        event_dispatcher.on_event(Event.StageOffsetRequest, { coord:  self.step_sizes()[i] })
      def callback_neg():
        event_dispatcher.on_event(Event.StageOffsetRequest, { coord: -self.step_sizes()[i] })

      coord_inc_button = ttk.Button(self.frame, text=f'+{coord.upper()}', command=callback_pos)
      coord_dec_button = ttk.Button(self.frame, text=f'-{coord.upper()}', command=callback_neg)

      coord_inc_button.grid(row=1, column=i)
      coord_dec_button.grid(row=2, column=i)

    on_set_position = lambda: stage.set(self.position_intputs[0].get(), self.position_intputs[1].get(), self.position_intputs[2].get())
    set_position_button = ttk.Button(self.frame, text='Set Stage Position', command = on_set_position)
    set_position_button.grid(row=3, column=0, columnspan=3, sticky='ew')

    ttk.Label(self.frame, text='Stage Step Size (microns)', anchor='center').grid(row=4, column=0, columnspan=3, sticky='ew')
  
  def step_sizes(self) -> tuple[int, int, int]:
    return tuple(intput.get() for intput in self.step_size_intputs)

class FineAdjustmentFrame:
  def __init__(self, gui, parent):
    global stage
    
    self.frame = ttk.Frame(parent)

    self.position_intputs = []
    self.step_size_intputs = []

    for i, coord, default in ((0, 'x', stage.x()), (1, 'y', stage.y()), (2, 'theta', stage.z())):
      self.position_intputs.append(Intput(
        gui=GUI,
        parent=self.frame,
        name=f'fine_{coord}_intput',
        default=int(default)
      ))
      self.position_intputs[-1].grid(row=0,col=i)
      # TODO: FIXME:
      #fine_adjust.update_funcs["x"]["fine x intput"] = lambda: fine_x_intput.set(int(fine_adjust.x()))

      self.step_size_intputs.append(Intput(
          gui=gui,
          parent=self.frame,
          name=f'{coord}_step_intput',
          default=10,
          min=-1000,
          max=1000
      ))
      self.step_size_intputs[-1].grid(row=5,col=i)

      coord_inc_button = ttk.Button(self.frame, text=f'+{coord.upper()}', command = lambda : fine_step_update(f'+{coord}'))
      coord_dec_button = ttk.Button(self.frame, text=f'-{coord.upper()}', command = lambda : fine_step_update(f'-{coord}'))

      coord_inc_button.grid(row=1, column=i)
      coord_dec_button.grid(row=2, column=i)

    on_set_position = lambda: stage.set(self.position_intputs[0].get(), self.position_intputs[1].get(), self.position_intputs[2].get())
    set_position_button = ttk.Button(self.frame, text='Set Fine Adjustment', command = on_set_position)
    set_position_button.grid(row=3, column=0, columnspan=3, sticky='ew')

    ttk.Label(self.frame, text='Fine Adjustment Step Size', anchor='center').grid(row=4, column=0, columnspan=3, sticky='ew')
  
  def fine_step_size(self) -> tuple[int, int, int]:
    return tuple(intput.get() for intput in self.step_size_intputs)

#region: Debug and Help 
# the debug widget needs to be added immedaitely, so this is all that needs to be here
debug.grid(GUI.grid_size[0]-1,0,colspan=GUI.grid_size[1]-1)

help_text: str = """
How do I move the projector window?
- On Windows, click the projector window, then win + shift + arrow keys to move it to the second screen 
- On Mac, no clue :P


How do I import an image?
- Just click on the black thumbnail and a dialog will open
  - The UI will try to fix images in the incorrect mode
  - The UI will reject incorrect file formats
- The "show" buttons below the previews will show the image on the projector


How do I use the stage controls?
- You can type in coordinates, then press "set stage position" to move the stage
- Or, you can use the step buttons on the GUI or the arrow keys on your keyboard (ctrl/shift+up/down for z axis)
- You can also modify the step sizes for each axis. Those are applied immediately. 
- Default units are in microns.


How do I use flatfield correction?
1. Take a flatfield image
  - Set the projector to UV mode
  - Display a fully white image on the projector
  - Put a clean blank chip under the projector
  - Take a snapshot with the amscope camera (1080p is plenty)
  - Crop out any black borders if present
2. Import the flatfield image
  - Just click on the flatfield image preview thumbnial
  - The UI will automatically guess the correct correction intensity to use 
3. Make sure flatfield correction is enabled
  - press the "use flatfield" button to toggle it
4. Done, though some things to note
  - Flatfield correction will only be applied to the pattern, not red or uv focus
  - The intensity of the correction is normalized from 0 to 100 for your convenience:
    - 0   means no correction, ie completely transparent
    - 50  means the correction is applied at full strength, from pure black pixels to pure white
    - 100 means max correction, ie completely opaque


Posterizer? I barely know her!
- TL;DR, make pattern monochrome for sharper edges
- What is that number next to it?
  - That is the cutoff value for what is considered white / black
  - Unless you're losing features or lines are growing / shrinking, leave it at 50
  - 100 is max cutoff, so only pure white will stay white
  -  50 is default, light greys will be white, and dark greys will be black
  -   0 is min cutoff, so only pure black will stay black
  

What are those tiling fields? and why are they zero?
- They are zero by default because zero is the keyword for "auto calculate". It's recommended to always leave at least one as zero
- The left is how many columns, and the right is how many rows.
- The preview window shows the next tile that will be displayed


Think something is missing? Have a suggestion?
see our website for contact methods:
http://hackerfab.ece.cmu.edu


This tool was made by Luca Garlati and Kent Wirant for Hacker Fab
"""
help_popup: TextPopup = TextPopup(
  root=GUI.root,
  title="Help Popup",
  button_text="Help",
  popup_text=help_text,
  debug=debug)
help_popup.grid(GUI.grid_size[0]-1,GUI.grid_size[1]-1)
GUI.add_widget("help_popup", help_popup)
#endregion

def guess_alpha(flatfield_thumb: Thumbnail):
  flatfield_thumb.image.add("name", "flatfield", True)
  brightness: tuple[int,int] = get_brightness_range(flatfield_thumb.image.image, downsample_target=480)
  options_frame.set_flatfield_strength(round(((brightness[1]-brightness[0])*100)/510))
  
class ImageSelectFrame:
  def __init__(self, gui, parent, name, button_text, import_command, select_command):
    self.frame = ttk.Frame(parent)

    thumb: Thumbnail = Thumbnail(
      gui=gui,
      parent=self.frame,
      name=name,
      thumb_size=THUMBNAIL_SIZE,
      func_on_success = import_command
    )
    thumb.grid(row=0, col=0)

    self.select_button = ttk.Button(self.frame, text=button_text, command=select_command)
    self.select_button.grid(row=1, column=0)


OnShowCallback = Callable[[Literal['pattern', 'flatfield', 'red_focus', 'uv_focus']], None]

class MultiImageSelectFrame:
  def __init__(self, gui, parent, event_dispatcher: EventDispatcher):
    self.frame = ttk.Frame(parent)

    # TODO: Update slicer on pattern upload
    pattern_frame = ImageSelectFrame(gui, self.frame, 'pattern_thumb', 'Show pattern', lambda t: (), event_dispatcher.on_shown_image_change_cb(ShownImage.Pattern))
    flatfield_frame = ImageSelectFrame(gui, self.frame, 'flatfield_thumb', 'Show flatfield', guess_alpha, event_dispatcher.on_shown_image_change_cb(ShownImage.Flatfield))
    red_focus_frame = ImageSelectFrame(gui, self.frame, 'red_focus_thumb', 'Show Red Focus', lambda t: t.image.add("name", 'red focus', True), event_dispatcher.on_shown_image_change_cb(ShownImage.RedFocus))
    uv_focus_frame = ImageSelectFrame(gui, self.frame, 'uv_focus_thumb', 'Show UV Focus', lambda t: t.image.add("name", 'uv focus', True), event_dispatcher.on_shown_image_change_cb(ShownImage.UvFocus))

    pattern_frame.frame.grid(row=0, column=0)
    flatfield_frame.frame.grid(row=0, column=1)
    red_focus_frame.frame.grid(row=1, column=0)
    uv_focus_frame.frame.grid(row=1, column=1)

    event_dispatcher.add_shown_image_change_listener(lambda img: self.highlight_button(img))
  
  def highlight_button(self, which: ShownImage):
    # TODO:
    pass

#region: keyboard input

def bind_stage_controls() -> None:
  GUI.root.bind('<Up>',           lambda event: step_update('+y'))
  GUI.root.bind('<Down>',         lambda event: step_update('-y'))
  GUI.root.bind('<Left>',         lambda event: step_update('-x'))
  GUI.root.bind('<Right>',        lambda event: step_update('+x'))
  GUI.root.bind('<Control-Up>',   lambda event: step_update('+z'))
  GUI.root.bind('<Control-Down>', lambda event: step_update('-z'))
  GUI.root.bind('<Shift-Up>',     lambda event: step_update('+z'))
  GUI.root.bind('<Shift-Down>',   lambda event: step_update('-z'))
def unbind_stage_controls() -> None:
  GUI.root.unbind('<Up>')
  GUI.root.unbind('<Down>')
  GUI.root.unbind('<Left>')
  GUI.root.unbind('<Right>')
  GUI.root.unbind('<Control-Up>')
  GUI.root.unbind('<Control-Down>')
  GUI.root.unbind('<Shift-Up>')
  GUI.root.unbind('<Shift-Down>')

#endregion

#region: Fine Adjustment Area

# IMPORTANT:
# to reduce complexity, the fine adjustment area reuses the stage controller class but
# instead of the z field and methods reprersenting the z axis, they represent the theta axis
# this is confusing, but it's better than adding unnecessary complixty to the stage controller class

fine_adjust: Stage_Controller = Stage_Controller(
  debug=debug,
  verbosity=1)

#TODO: Check to make sure there is no chance of the semaphore getting stuck on
fine_step_busy: bool = False
def fine_step_update(axis: Literal['-x','+x','-y','+y','-z','+z']):
  global fine_step_busy
  if(fine_step_busy):
    debug.warn("Fine Adjustment is busy, slow down!")
    return
  fine_step_busy = True
  debug.info(f"Fine Step Update: {axis}")
  # first check if the step size has changed
  fine_adjust.step_size = fine_adjustment_frame.fine_step_size()
  # next update the values
  fine_adjust.step(axis)
  # update image
  update_displayed_image()
  # remove semaphore lock
  fine_step_busy = False

#endregion

def set_and_update_fine_adjustment() -> None:
  fine_adjust.set(fine_x_intput.get(), fine_y_intput.get(), fine_theta_floatput.get())
  update_displayed_image()

#endregion

#region: keyboard input
def bind_fine_controls() -> None:
  GUI.root.bind('<Up>',           lambda event: fine_step_update('+y'))
  GUI.root.bind('<Down>',         lambda event: fine_step_update('-y'))
  GUI.root.bind('<Left>',         lambda event: fine_step_update('-x'))
  GUI.root.bind('<Right>',        lambda event: fine_step_update('+x'))
  GUI.root.bind('<Control-Up>',   lambda event: fine_step_update('+z'))
  GUI.root.bind('<Control-Down>', lambda event: fine_step_update('-z'))
  GUI.root.bind('<Shift-Up>',     lambda event: fine_step_update('+z'))
  GUI.root.bind('<Shift-Down>',   lambda event: fine_step_update('-z'))
def unbind_fine_controls() -> None:
  GUI.root.unbind('<Up>')
  GUI.root.unbind('<Down>')
  GUI.root.unbind('<Left>')
  GUI.root.unbind('<Right>')
  GUI.root.unbind('<Control-Up>')
  GUI.root.unbind('<Control-Down>')
  GUI.root.unbind('<Shift-Up>')
  GUI.root.unbind('<Shift-Down>')
  
#endregion

#endregion

class OptionsFrame:
  def __init__(self, gui, parent):
    self.frame = ttk.Frame(parent)

    ttk.Label(self.frame, text='Exposure Time (ms)', anchor='w').grid(row=0, column=0)
    self.exposure_time_entry = Intput(gui, parent=self.frame, default=1000, min=0)
    self.exposure_time_entry.grid(row=0, col=1, colspan=3)

    ttk.Label(self.frame, text='Tiles (horiz, vert)', anchor='w').grid(row=1, column=0)
    self.tiles_horiz_entry = Intput(gui, parent=self.frame, default=0, min=0)
    self.tiles_horiz_entry.grid(row=1, col=1)
    self.tiles_vert_entry = Intput(gui, parent=self.frame, default=0, min=0)
    self.tiles_vert_entry.grid(row=1, col=2)
    self.tiles_snake_button = ttk.Button(self.frame, text='Snake')
    self.tiles_snake_button.grid(row=1, column=3, sticky='ew')

    ttk.Label(self.frame, text='Flatfield Strength (%)', anchor='w').grid(row=2, column=0)
    self.flatfield_strength_entry = Intput(gui, parent=self.frame, default=0, min=0, max=100)
    self.flatfield_strength_entry.grid(row=2, col=2)
    self.flatfield_button = ttk.Button(self.frame, text='NOT Using Flatfield')
    self.flatfield_button.grid(row=2, column=3, sticky='ew')

    ttk.Label(self.frame, text='Posterize Cutoff (%)').grid(row=3, column=0)
    self.posterize_cutoff_entry = Intput(gui, parent=self.frame, default=50, min=0, max=100)
    self.posterize_cutoff_entry.grid(row=3, col=2)
    self.posterize_button = ttk.Button(self.frame, text='NOT Posterizing')
    self.posterize_button.grid(row=3, column=3, sticky='ew')

    ttk.Label(self.frame, text='Fine Adj. Border (%)').grid(row=4, column=0)
    self.fine_adj_border_entry = Intput(gui, parent=self.frame, default=20, min=0, max=100)
    self.fine_adj_border_entry.grid(row=4, col=1)
    self.fine_adj_button_1 = ttk.Button(self.frame, text='Button 1')
    self.fine_adj_button_1.grid(row=4, column=2, sticky='ew')
    self.fine_adj_button_2 = ttk.Button(self.frame, text='Button 2')
    self.fine_adj_button_2.grid(row=4, column=3, sticky='ew')

    for i, ty in enumerate(('Pattern', 'Red Focus', 'UV Focus')):
        ttk.Label(self.frame, text=f'{ty} Channels').grid(row=5+i, column=0)
        #red = ToggleButton(self.frame, ButtonStyle('RED ENABLED', 'red'), ButtonStyle('Red Disabled', 'black'))
        #red.widget.grid(row=5+i, column=1, sticky='ew')
        #green = ToggleButton(self.frame, ButtonStyle('GREEN ENABLED', 'green'), ButtonStyle('Green Disabled', 'black'))
        #green.widget.grid(row=5+i, column=2, sticky='ew')
        #blue = ToggleButton(self.frame, ButtonStyle('BLUE ENABLED', 'blue'), ButtonStyle('Blue Disabled', 'black'))
        #blue.widget.grid(row=5+i, column=3, sticky='ew')
    
    ttk.Label(self.frame, text='Calibration Controls').grid(row=8, column=0)
    self.calibrate_button = ttk.Button(self.frame, text='Calibrate', command = lambda: stage.calibrate(
        step_size = self.calibrate_entry.get(),
        calibrate_backlash = 'symmetric',
        return_to_start = True
      )
    )
    self.calibrate_button.grid(row=8, column=1, sticky='ew')
    self.calibrate_entry = Intput(gui, parent=self.frame, default=100, min=1)
    self.calibrate_entry.grid(row=8, col=2)
    self.goto_button = ttk.Button(self.frame, text='goto DISABLED')
    self.goto_button.grid(row=8, column=3, sticky='ew')
 
  def horiz_tiles(self) -> int:
    return self.tiles_horiz_entry.get()
  
  def vert_tiles(self) -> int:
    return self.tiles_vert_entry.get()
  
  # returns threshold percentage if posterizing is enabled, else None
  def posterize_strength(self) -> Optional[int]:
    # TODO: enable/disable
    return self.posterize_cutoff_entry.get()

  def flatfield_strength(self) -> Optional[int]:
    # TODO: enable/disable
    return self.flatfield_strength_entry.get()

  def set_flatfield_strength(self, strength: Optional[int]):
    # TODO:
    pass
  
  def pattern_channels(self) -> tuple[bool, bool, bool]:
    # TODO:
    return (False, False, False)
 
  def red_focus_channels(self) -> tuple[bool, bool, bool]:
    # TODO:
    return (False, False, False)
  
  def uv_focus_channels(self) -> tuple[bool, bool, bool]:
    # TODO:
    return (False, False, False)
  
  def fine_adj_enabled(self) -> bool:
    # TODO:
    return False


class PatterningFrame:
  def __init__(self, gui, parent, event_dispatcher):
    self.frame = ttk.Frame(parent)
    self.event_dispatcher = event_dispatcher

    #region: Current Tile
    ttk.Label(self.frame, text='Next Pattern Image').grid(row=0, column=0)

    tile_placeholder = rasterize(Image.new('RGB', THUMBNAIL_SIZE, (0,0,0)))
    self.next_tile_image = Label(self.frame, image=tile_placeholder, justify='center', anchor='center')
    self.next_tile_image.grid(row=1, column=0)

    lower_frame = ttk.Frame(self.frame)
    lower_frame.grid(row=2, column=0)

    self.clear_button = Button(lower_frame)
    self.clear_button.grid(row=0, column=0)

    pattern_button_timed = Button(lower_frame, text='Begin\nPatterning', command=event_dispatcher.on_begin_patterning_cb(), bg='red', fg='white')
    pattern_button_timed.grid(row=0, column=1)

    event_dispatcher.add_change_patterning_status_listener(lambda s: self.change_patterning_status(s))
    self.change_patterning_status(PatterningStatus.Idle)
  
  def set_next_tile_image(self, image):
    self.next_tile_image.config(image=image)

  def change_patterning_status(self, status: PatterningStatus) -> None:
    match status:
      case PatterningStatus.Idle:
        self.clear_button.config(text='Clear', bg='black', fg='white', command=self.event_dispatcher.on_shown_image_change_cb(ShownImage.Clear))
      case PatterningStatus.Patterning:
        # TODO: Disable/hide pattern button
        self.clear_button.config(text='Abort', bg='red', fg='white', command=self.event_dispatcher.on_change_patterning_status_cb(PatterningStatus.Aborting))
      case PatterningStatus.Aborting:
        pass


'''
slicer_pattern_cycle: Cycle = Cycle(gui=GUI, name="slicer_pattern_cycle")
slicer_pattern_cycle.add_state(text = "Snake", colors=("black","pale green"))
slicer_pattern_cycle.add_state(text = "Row Major", colors=("black","light blue"))
slicer_pattern_cycle.add_state(text = "Col Major", colors=("black","light pink"))
slicer_pattern_cycle.grid(pattern_row+options_row+2,pattern_col+options_col+3)

flatfield_cycle: Cycle = Cycle(gui=GUI, name="flatfield_cycle")
flatfield_cycle.add_state(text = "NOT Using Flatfield")
flatfield_cycle.add_state(text = "Using Flatfield", colors=("white", "gray"))
flatfield_cycle.grid(pattern_row+options_row+3,pattern_col+options_col+3)

posterize_cycle: Cycle = Cycle(gui=GUI, name="posterize_cycle")
posterize_cycle.add_state(text = "NOT Posterizing")
posterize_cycle.add_state(text = "Now Posterizing", colors=("white", "gray"))
posterize_cycle.grid(pattern_row+options_row+4,pattern_col+options_col+3)

reset_adj_cycle: Cycle = Cycle(gui=GUI, name="reset_adj_cycle")
reset_adj_cycle.add_state(text = "Reset Nothing")
reset_adj_cycle.add_state(text = "Reset XY only", colors=("black","light pink"))
reset_adj_cycle.add_state(text = "Reset theta only", colors=("black","light blue"))
reset_adj_cycle.add_state(text = "Reset All", colors=("white", "gray"))
reset_adj_cycle.grid(pattern_row+options_row+5,pattern_col+options_col+2)

fine_adjustment_cycle: Cycle = Cycle(gui=GUI,
                                     name="fine_adjustment_cycle",
                                     func_always=set_and_update_fine_adjustment)
fine_adjustment_cycle.add_state(text = "NOT Fine Adjust")
fine_adjustment_cycle.add_state(text = "Now Fine Adjust", colors=("white", "gray"))
fine_adjustment_cycle.grid(pattern_row+options_row+5,pattern_col+options_col+3)
#endregion

#region: pattern RGB
pattern_rgb_text: Label = Label(
  GUI.root,
  text = "Pattern Channels",
  justify = 'left',
  anchor = 'w'
)
pattern_rgb_text.grid(
  row = pattern_row+options_row+6,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("pattern_rgb_text", pattern_rgb_text)

pattern_red_cycle: Cycle = Cycle(gui=GUI, name="pattern_red_cycle")
pattern_red_cycle.add_state(text = "Red")
pattern_red_cycle.add_state(text = "Red", colors=("black","light pink"))
pattern_red_cycle.goto(1)
pattern_red_cycle.grid(pattern_row+options_row+6,pattern_col+options_col+1)

pattern_green_cycle: Cycle = Cycle(gui=GUI, name="pattern_green_cycle")
pattern_green_cycle.add_state(text = "Green")
pattern_green_cycle.add_state(text = "Green", colors=("black","pale green"))
pattern_green_cycle.goto(1)
pattern_green_cycle.grid(pattern_row+options_row+6,pattern_col+options_col+2)

pattern_blue_cycle: Cycle = Cycle(gui=GUI, name="pattern_blue_cycle")
pattern_blue_cycle.add_state(text = "Blue")
pattern_blue_cycle.add_state(text = "Blue", colors=("black","light blue"))
pattern_blue_cycle.goto(1)
pattern_blue_cycle.grid(pattern_row+options_row+6,pattern_col+options_col+3)

#endregion

#region: red focus RGB
red_focus_rgb_text: Label = Label(
  GUI.root,
  text = "Red Focus Channels",
  justify = 'left',
  anchor = 'w'
)
red_focus_rgb_text.grid(
  row = pattern_row+options_row+7,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("red_focus_rgb_text", red_focus_rgb_text)

red_focus_red_cycle: Cycle = Cycle(gui=GUI, name="red_focus_red_cycle")
red_focus_red_cycle.add_state(text = "Red")
red_focus_red_cycle.add_state(text = "Red", colors=("black","light pink"))
red_focus_red_cycle.goto(1)
red_focus_red_cycle.grid(pattern_row+options_row+7,pattern_col+options_col+1)

red_focus_green_cycle: Cycle = Cycle(gui=GUI, name="red_focus_green_cycle")
red_focus_green_cycle.add_state(text = "Green")
red_focus_green_cycle.add_state(text = "Green", colors=("black","pale green"))
red_focus_green_cycle.goto(0)
red_focus_green_cycle.grid(pattern_row+options_row+7,pattern_col+options_col+2)

red_focus_blue_cycle: Cycle = Cycle(gui=GUI, name="red_focus_blue_cycle")
red_focus_blue_cycle.add_state(text = "Blue")
red_focus_blue_cycle.add_state(text = "Blue", colors=("black","light blue"))
red_focus_blue_cycle.goto(0)
red_focus_blue_cycle.grid(pattern_row+options_row+7,pattern_col+options_col+3)


#endregion

#region: UV Focus RGB
uv_focus_rgb_text: Label = Label(
  GUI.root,
  text = "UV Focus Channels",
  justify = 'left',
  anchor = 'w'
)
uv_focus_rgb_text.grid(
  row = pattern_row+options_row+8,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("uv_focus_rgb_text", uv_focus_rgb_text)

uv_focus_red_cycle: Cycle = Cycle(gui=GUI, name="uv_focus_red_cycle")
uv_focus_red_cycle.add_state(text = "Red")
uv_focus_red_cycle.add_state(text = "Red", colors=("black","light pink"))
uv_focus_red_cycle.goto(0)
uv_focus_red_cycle.grid(pattern_row+options_row+8,pattern_col+options_col+1)

uv_focus_green_cycle: Cycle = Cycle(gui=GUI, name="uv_focus_green_cycle")
uv_focus_green_cycle.add_state(text = "Green")
uv_focus_green_cycle.add_state(text = "Green", colors=("black","pale green"))
uv_focus_green_cycle.goto(0)
uv_focus_green_cycle.grid(pattern_row+options_row+8,pattern_col+options_col+2)

uv_focus_blue_cycle: Cycle = Cycle(gui=GUI, name="uv_focus_blue_cycle")
uv_focus_blue_cycle.add_state(text = "Blue")
uv_focus_blue_cycle.add_state(text = "Blue", colors=("black","light blue"))
uv_focus_blue_cycle.goto(1)
uv_focus_blue_cycle.grid(pattern_row+options_row+8,pattern_col+options_col+3)
#endregion

# goto cycle
def goto_func() -> None:
  # this is a little hacky, but I can't be bothered to rewrite the backend
  # again to support this in a more elegant way.
  coords = GUI.get_coords("camera", CAMERA_IMAGE_SIZE)
  if(BTW_tuple(0,coords,1) and goto_cycle.state == 1):
    stage.goto(coords)
  if(goto_cycle.state == 1):
    goto_func()

goto_cycle: Cycle = Cycle(gui=GUI, name="goto_cycle")
goto_cycle.add_state(text = "goto DISABLED")
goto_cycle.add_state(text = "goto ENABLED",
                     colors=("black", "light pink"),
                     enter = goto_func)
goto_cycle.goto(0)
goto_cycle.grid(pattern_row+options_row+9,pattern_col+options_col+3)
#endregion: calibration

'''

#region: Camera Setup
cv_stage_job = None
gui_camera_preview_job = None
cv_stage_job_time = 0
gui_camera_preview_job_time = 0

# sends image to stage controller
def cv_stage(camera_image):
  grayscale = cv2.normalize(camera_image, None, 0, 255, cv2.NORM_MINMAX)
  #stage_ll.updateImage(grayscale)

# updates camera preview on GUI
import numpy as np

# from skimage.measure import block_reduce
def gui_camera_preview(camera_image, dimensions):
  pil_img = Image.fromarray(camera_image, mode='RGB')
  gui_img = rasterize(pil_img.resize(fit_image(pil_img, (GUI.window_size[0],int((GUI.window_size[0]*dimensions[0])/dimensions[1]//1))), Image.Resampling.NEAREST))
  camera.config(image=gui_img)
  camera.image = gui_img
  # print(dimensions)
  # target_ratio = div(dimensions, fill_image(dimensions, GUI.window_size))
  # small_np = block_reduce(camera_image, block_size=(target_ratio[1], target_ratio[0], 1), func=np.mean)
  # raster_image = ImageTk.PhotoImage(image=Image.fromarray(small_np, mode='RGB'))
  # camera.config(image=raster_image)
  # camera.image = raster_image
  # print(small_np.shape)


# called by camera hardware as separate thread
CAMERA_IMAGE_SIZE: tuple[int,int] = (0,0)
def cameraCallback(image, dimensions, format):
  global cv_stage_job
  global gui_camera_preview_job
  global cv_stage_job_time
  global gui_camera_preview_job_time
  
  #region for goto functionality
  global CAMERA_IMAGE_SIZE
  if(CAMERA_IMAGE_SIZE == (0,0)):
    CAMERA_IMAGE_SIZE = dimensions
    debug.info(f"Camera image size: {CAMERA_IMAGE_SIZE}")
  #endregion
  
  '''
  # might be susceptible to TOC-TOU race condition
  if cv_stage_job is None or not cv_stage_job.is_alive():
    cv_stage_job = threading.Thread(target=cv_stage, args=(image,))
    cv_stage_job.start()
    new_time = time.time()
    print(f"CV-Stage Time: {new_time - cv_stage_job_time}s", flush=True)
    cv_stage_job_time = new_time
  '''
  gui_camera_preview(image, dimensions)
  # if gui_camera_preview_job is None or not gui_camera_preview_job.is_alive():
  #   gui_camera_preview_job = threading.Thread(target=gui_camera_preview, args=(image, dimensions,))
  #   gui_camera_preview_job.start()
  #   new_time = time.time()
  #   #print(f"GUI-Camera Time: {new_time - gui_camera_preview_job_time}s", flush=True)
  #   gui_camera_preview_job_time = new_time

  # print(f'image captured; num_threads={len(threading.enumerate())}', flush=True)


def setup_camera_from_py():
  if not camera_hw.open():
    debug.error("Camera failed to start.")
  else:
    camera_hw.setSetting('image_format', "rgb888")
    camera_hw.setStreamCaptureCallback(cameraCallback)

    if not camera_hw.startStreamCapture():
      debug.error('Failed to start stream capture for camera')
#endregion: Camera Setup

class LithographerGui:
  hardware: Lithographer

  pattern_image: ProcessedImage
  flatfield_image: ProcessedImage
  red_focus_image: ProcessedImage
  uv_focus_image: ProcessedImage

  event_dispatcher: EventDispatcher

  def __init__(self, gui, use_camera, stage: StageController):
    self.event_dispatcher = EventDispatcher()
    self.event_dispatcher.add_shown_image_change_listener(lambda img: self.on_shown_image_change(img))
    self.event_dispatcher.add_begin_patterning_listener(lambda: self.begin_patterning())
    self.event_dispatcher.add_change_patterning_status_listener(lambda status: self.on_change_patterning_status(status))

    self.hardware = Lithographer(stage, projector)

    self.event_dispatcher.add_event_listener(
      Event.StageOffsetRequest,
      lambda offsets: self.hardware.stage.move_by(offsets, commit=True)
    )

    if use_camera:
      camera_placeholder = rasterize(Image.new('RGB', (GUI.window_size[0],(GUI.window_size[0]*9)//16), (0,0,0)))
      # TODO properly implement the camera image size
      self.camera_label = Label(gui.root, image=camera_placeholder)
      self.camera_label.grid(row = 0, column = 0, sticky='nesw')
    else:
      self.camera_label = None

    self.stage_notebook = ttk.Notebook(GUI.root)
    self.stage_notebook.grid(row=3, column=2)
    self.stage_position_frame = StagePositionFrame(GUI, self.stage_notebook, self.event_dispatcher)
    self.stage_notebook.add(self.stage_position_frame.frame, text='Stage Position')
    self.fine_adjustment_frame = FineAdjustmentFrame(GUI, self.stage_notebook)
    self.stage_notebook.add(self.fine_adjustment_frame.frame, text='Fine Adjustment')

    self.config_notebook = ttk.Notebook(GUI.root)
    self.config_notebook.grid(row=3, column=3)
    self.options_frame = OptionsFrame(GUI, self.config_notebook)
    self.config_notebook.add(self.options_frame.frame, text='Options')
    self.patterning_frame = PatterningFrame(GUI, self.config_notebook, self.event_dispatcher)
    self.config_notebook.add(self.patterning_frame.frame, text='Patterning')

    def pattern_image_settings() -> ImageProcessSettings:
      return ImageProcessSettings(self.options_frame.posterize_strength(), self.options_frame.flatfield_strength(), self.options_frame.pattern_channels())

    # TODO: Red focus has no flatfield correction, is this intentional?
    def red_focus_image_settings() -> ImageProcessSettings:
      return ImageProcessSettings(self.options_frame.posterize_strength(), None, self.options_frame.red_focus_channels())
    
    # TODO: UV focus has no flatfield or posterization, is this intentional?
    def uv_focus_image_settings() -> ImageProcessSettings:
      return ImageProcessSettings(None, None, self.options_frame.uv_focus_channels())

    # TODO: Automatically recompute preview on settings change
    self.pattern_image = ProcessedImage(Image.new('RGB', THUMBNAIL_SIZE), pattern_image_settings)
    self.red_focus_image = ProcessedImage(Image.new('RGB', THUMBNAIL_SIZE), red_focus_image_settings)
    self.uv_focus_image = ProcessedImage(Image.new('RGB', THUMBNAIL_SIZE), uv_focus_image_settings)

    self.multi_image_select_frame = MultiImageSelectFrame(gui, gui.root, self.event_dispatcher)
    self.multi_image_select_frame.frame.grid(row=3, column=0)

    self.patterning_status = PatterningStatus.Idle

    gui.root.protocol("WM_DELETE_WINDOW", lambda: self.cleanup())
    gui.debug.info("Debug info will appear here")
  
  def cleanup(self):
    print("Patterning GUI closed.")
    print('TODO: Cleanup')
    #GUI.root.destroy()
    #if RUN_WITH_STAGE:
      #serial_port.close()

  
  def transform_image(self, image: Image.Image, theta_factor: float = 0.1) -> Image.Image:
    if self.options_frame.fine_adj_enabled():
      return better_transform(image, (*round_tuple(fine_adjust.xy()), (2*pi*fine_adjust.z())/360), GUI.proj.size(), border_size_intput.get())
    return image
  
  def on_shown_image_change(self, which: ShownImage):
    match which:
      case ShownImage.Clear:
        self.hardware.projector.clear()
      case ShownImage.Pattern:
        debug.info("Showing Pattern")
        self.hardware.projector.show(self.transform_image(self.pattern_image.processed()))
      case ShownImage.Flatfield:
        debug.info("Showing flatfield image")
        self.hardware.projector.show(self.transform_image(self.flatfield_image.processed()))
      case ShownImage.RedFocus:
        debug.info("Showing red focus")
        self.hardware.projector.show.show(self.transform_image(self.red_focus_image.processed()))
      case ShownImage.UvFocus:
        self.hardware.projector.show.info("Showing UV focus")
        GUI.proj.show(self.transform_image(self.uv_focus_image.processed()))
  
  def change_patterning_status(self, status: PatterningStatus):
    (self.event_dispatcher.on_change_patterning_status_cb(status))()
    self.patterning_status = status

  def on_change_patterning_status(self, status: PatterningStatus, notify=False):
    self.patterning_status = status

  def begin_patterning(self):
    def update_next_tile_preview(mode: Literal['current','peek']='peek'):
      #get either current image, or peek ahead to next image
      preview: Image.Image | None
      preview = slicer.peek() if mode == 'peek' else slicer.image()
      #if at end of slicer, use blank image
      if preview is None:
        preview = Image.new('RGB', THUMBNAIL_SIZE)
      raster = rasterize(preview.resize(fit_image(preview, THUMBNAIL_SIZE), Image.Resampling.NEAREST))
      self.patterning_frame.set_next_tile_image(raster)
    
    global pattern_status
    debug.info("Slicing pattern...")
    slicer.update(image=self.pattern_image.processed(),
                  horizontal_tiles=self.options_frame.horiz_tiles(),
                  vertical_tiles=self.options_frame.vert_tiles(),
                  tiling_pattern=slicer.pattern_list[slicer_pattern_cycle.state])
    pattern_progress['value'] = 0
    pattern_progress['maximum'] = slicer.tile_count()
    debug.info(f"Patterning {slicer.tile_count()} tiles for {duration_intput.get()}ms\nTotal time: {str(round((slicer.tile_count()*duration_intput.get())/1000))}s")

    # TODO implement fine adjustment with CV
    # delta_vector: tuple[int,int,float] = (0,0,0)
    self.change_patterning_status(PatterningStatus.Patterning)
    while True:
      # update next tile preview
      update_next_tile_preview()
      # get patterning image
      image = process_img2(slicer.image(), ImageProcessSettings(self.options_frame.posterize_strength(), self.options_frame.flatfield_strength(), self.options_frame.pattern_channels()))
      #TODO apply fine adjustment vector to image
      #TODO remove once camera is implemented
      #camera_image_preview = rasterize(image.resize(fit_image(image, (GUI.window_size[0],(GUI.window_size[0]*9)//16)), Image.Resampling.LANCZOS))
      #camera.config(image=camera_image_preview)
      #camera.image = camera_image_preview
      #pattern
      if self.patterning_status == PatterningStatus.Aborting:
        break
      stage.lock()
      debug.info("Patterning tile...")
      GUI.proj.show(image, duration=duration_intput.get())
      stage.unlock()
      if self.patterning_status == PatterningStatus.Aborting:
        break

      # if(result):
      #   # TODO remove once camera is implemented
      #   camera.config(image=camera_placeholder)
      #   camera.image = camera_placeholder
      # repeat
      if slicer.next():
        pattern_progress['value'] += 1
        debug.info("Finished")
        #TODO: implement CV
        #delta_vector = tuple(map(float, input("Next vector [dX dY theta]:").split(None,3)))
      else:
        break
      #TODO: delete this pause. This is to "emulate" the CV taking time to move the stage
      sleep(0.5)
    # restart slicer
    slicer.restart()
    # update next tile preview
    update_next_tile_preview(mode='current')
    # TODO remove once camera is implemented
    # camera.config(image=camera_placeholder)
    # camera.image = camera_placeholder
    # reset fine adjustment parameters based on reset_adj_cycle
    match reset_adj_cycle.state_name():
      case "Reset Nothing":
        pass
      case "Reset XY only":
        fine_adjust.set(0,0,fine_adjust.z())
      case "Reset theta only":
        fine_adjust.set(*fine_adjust.xy(),0)
      case "Reset All":
        fine_adjust.set(0,0,0)
      case _:
        debug.warn("Invalid state for reset_adj_cycle")

    # give user feedback
    pattern_progress['value'] = 0
    if self.patterning_status == PatterningStatus.Aborting:
      debug.warn("Patterning aborted")
    else:
      debug.info("Done")
    self.change_patterning_status(PatterningStatus.Idle)




# attach cleanup function to GUI close event

if(RUN_WITH_CAMERA):
  setup_camera_from_py()

if RUN_WITH_STAGE:
  serial_port = serial.Serial(stage_file, baud_rate)
  print(f"Using serial port {serial_port.name}")
  stage = GrblStage(serial_port, bounds=((-12000,12000),(-12000,12000),(-12000,12000))) 
else:
  stage = StageController()

lithographer = LithographerGui(GUI, False, stage)

GUI.mainloop()


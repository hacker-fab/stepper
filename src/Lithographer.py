# Hacker Fab
# Luca Garlati, 2024
# Kent Wirant
# Main lithography script

from typing import Literal
from tkinter import Button, Label
from tkinter.ttk import Progressbar
from collections import namedtuple
from PIL import  Image
from time import sleep
from stage_control.grbl_stage import GrblStage

from lithographer_lib.gui_lib import *
from lithographer_lib.img_lib import *
from lithographer_lib.backend_lib import *

# import configuration variables
from config import RUN_WITH_CAMERA
if(RUN_WITH_CAMERA):
  from config import camera as camera_hw
  import cv2
from config import RUN_WITH_STAGE
if(RUN_WITH_STAGE):
  from config import stage_file, baud_rate, scale_factor
  import serial

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

# main processing function
# will apply and reset: posterizing, flatfield, resizing, and color channels
# is smart, so it will only apply changes if necessary
# can specify a smart image xor a thumbnail
# NOTE this WILL modify the input smart image
#TODO don't actually need to reset for flatfield, just modify the alpha channel. Not worth hassle right now
#TODO allow toggling of what processing is applied to the image
#TODO add an option to not modify input
def process_img(image_in: Image.Image | Smart_Image | Thumbnail) -> Image.Image:
  
  # would use a switch, but python's match sucks and is impossible to use
  img: Smart_Image
  if(type(image_in) == Thumbnail):
    img = image_in.image
  elif(type(image_in) == Smart_Image):
    img = image_in
  elif(type(image_in) == Image.Image):
    img = Smart_Image(image_in)
  
#region: convenience vars
  color_channels: tuple#[bool,bool,bool] would type properly, but python throws a hissy fit if I do
  match img.get("name"):
    case "pattern":
      color_channels = (pattern_red_cycle.state, pattern_green_cycle.state, pattern_blue_cycle.state)
    case "red focus":
      color_channels = (red_focus_red_cycle.state, red_focus_green_cycle.state, red_focus_blue_cycle.state)
    case "uv focus":
      color_channels = (uv_focus_red_cycle.state, uv_focus_green_cycle.state, uv_focus_blue_cycle.state)
    case _:
      if(type(image_in)!=Image.Image):
        debug.warn("Unrecognized / unnamed smart image")
      color_channels = (True, True, True)
  saved_alpha: Image.Image | None = None
  
  # states of each type of processing in the following format:
  # (processing enabled, option changed, applied to image)
  State = namedtuple("State", ["enabled", "changed", "applied"])
  posterize_state = State(bool(posterize_cycle.state), post_strength_intput.changed(), img.get("posterize", False))
  flatfield_state = State(bool(flatfield_cycle.state), FF_strength_intput.changed(), img.get("flatfield", False))
  #endregion
  
  #region: resetting
  
  # reset matrix  | enabled changed applied |
  # --------------|-------------------------|
  # reset + apply |    o       o       o    |
  #         apply |    o       -       x    |
  #   pass / save |    o       x       o    |
  #         reset |    x       -       o    |
  #          pass |    x       -       x    |
  
  # if flatfield settings haven't changed, no need to recalculate. Extract the alpha channel and reapply later
  if(flatfield_state.enabled and not flatfield_state.changed and flatfield_state.applied and
     (img.mode() == "RGBA" or img.mode() == "LA")):
    saved_alpha = img.image.getchannel('A')
    
  # this is ugly but lets debug messages state what caused the reset
  reset: bool = False
  if(posterize_state.enabled and posterize_state.changed and posterize_state.applied):
    debug.info("Posterizing settings changed, resetting...")
    reset = True
  elif(not posterize_state.enabled and posterize_state.applied):
    debug.info("Posterizing disabled, resetting...")
    reset = True
  elif(flatfield_state.enabled and flatfield_state.changed and flatfield_state.applied):
    debug.info("Flatfield settings changed, resetting...")
    reset = True
  elif(not flatfield_state.enabled and flatfield_state.applied):
    debug.info("Flatfield disabled, resetting...")
    reset = True
  elif(img.get("color", (True, True, True)) != color_channels):
    debug.info("Color settings changed, resetting...")
    reset = True
  
  # reset
  if(reset):
    img.reset()
      
  #endregion
  
  #region: processing
  
  #TODO test what order is fastest
  # need to apply if:
  # enabled
  # and
  #   reset
  #   or
  #   changed or not applied
  
  # posterize
  if(posterize_state.enabled and 
     (reset or posterize_state.changed or not posterize_state.applied)):
    debug.info("Posterizing...")
    img.image = posterize(img.image, round((post_strength_intput.get()*255)/100))
    img.add("posterize", True)
  
  # color channel toggling (must be after posterizing)
  if(reset or img.get("color", (True, True, True)) != color_channels):
    debug.info("Toggling color channels...")
    img.image = toggle_channels(img.image, *color_channels)
    img.add("color", color_channels)
  
  # early flatfield
  if(flatfield_state.enabled and 
     (reset or flatfield_state.changed or not flatfield_state.applied)):
    # we need to apply flatfield, check if saved alpha works
    if(saved_alpha != None and saved_alpha.size == img.size()):
      debug.info("Applying flatfield correction...")
      img.image.putalpha(saved_alpha)
      img.add("flatfield", True)
  
  # resizeing
  if(img.size() != fit_image(img.image, GUI.proj.size())):
    debug.info("Resizing...")
    img.image = img.image.resize(fit_image(img.image, GUI.proj.size()), Image.Resampling.LANCZOS)
  
  # flatfield and check to make sure it wasn't applied early
  if(flatfield_state.enabled and not img.get("flatfield", False) and
     (reset or flatfield_state.changed or not flatfield_state.applied)):
    debug.info("Applying flatfield correction...")
    if(saved_alpha != None and saved_alpha.size == img.size()):
      img.image.putalpha(saved_alpha)
    else:
      alpha_channel = convert_to_alpha_channel(img.image,
                                               new_scale=dec_to_alpha(FF_strength_intput.get()),
                                               target_size=img.size(),
                                               downsample_target=540)
      img.image.putalpha(alpha_channel)
    img.add("flatfield", True)
  
  #endregion
  
  if(type(image_in) == Thumbnail):
    image_in.update()
  return img.image

#region: Camera and progress bars
if(RUN_WITH_CAMERA):
  camera_placeholder = rasterize(Image.new('RGB', (GUI.window_size[0],(GUI.window_size[0]*9)//16), (0,0,0)))
  # TODO properly implement the camera image size
  camera: Label = Label(
    GUI.root,
    image=camera_placeholder
    )
  camera.grid(
    row = 0,
    column = 0,
    columnspan = GUI.grid_size[1],
    sticky='nesw')
  GUI.add_widget("camera", camera)
else:
  GUI.root.rowconfigure(0,weight=0)

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

#endregion

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

#region: imports / thumbnails
import_row: int = 3
import_col: int = 0

#TODO: optimize so properties aren't reset unnecessarily
showing_state: Literal['pattern','red_focus','uv_focus','flatfield', 'clear'] = 'clear'
def highlight_button(button: Button | None) -> None:
  global showing_state
  if(button == pattern_button_fixed):
    pattern_button_fixed.config(bg="gray", fg="white")
    showing_state = 'pattern'
  else:
    pattern_button_fixed.config(bg="white", fg="black")
  if(button == red_focus_button):
    red_focus_button.config(bg="gray", fg="white")
    showing_state = 'red_focus'
  else:
    red_focus_button.config(bg="white", fg="black")
  if(button == uv_focus_button):
    uv_focus_button.config(bg="gray", fg="white")
    showing_state = 'uv_focus'
  else:
    uv_focus_button.config(bg="white", fg="black")
  if(button == flatfield_button):
    flatfield_button.config(bg="gray", fg="white")
    showing_state = 'flatfield'
  else:
    flatfield_button.config(bg="white", fg="black")

#region: Pattern
def pattern_import_func() -> None:
  pattern_thumb.image.add("name", "pattern", True)
  process_img(pattern_thumb)
  slicer.update(image=pattern_thumb.image.image,
                horizontal_tiles=slicer_horiz_intput.get(),
                vertical_tiles=slicer_vert_intput.get())
  # pattern_thumb.temp_image = slicer.image()
  raster = rasterize(slicer.image().resize(fit_image(slicer.image(), THUMBNAIL_SIZE), Image.Resampling.NEAREST))
  next_tile_image.config(image=raster)
  next_tile_image.image = raster
  
pattern_thumb: Thumbnail = Thumbnail(
  gui=GUI,
  name="pattern_thumb",
  thumb_size=THUMBNAIL_SIZE,
  func_on_success=pattern_import_func)
pattern_thumb.grid(import_row,import_col, rowspan=4)


def show_pattern_fixed(mode: Literal['update', 'slient']='update') -> None:
  highlight_button(pattern_button_fixed)
  process_img(pattern_thumb)
  if(mode == 'update'):
    pattern_thumb.update()
    debug.info("Showing Pattern")
  # apply affine transformation
  GUI.proj.show(transform_image(pattern_thumb.image.image))
pattern_button_fixed: Button = Button(
  GUI.root,
  text = 'Show Pattern',
  command = show_pattern_fixed)
pattern_button_fixed.grid(
  row = import_row+4,
  column = import_col,
  sticky='nesw')
GUI.add_widget("pattern_button_fixed", pattern_button_fixed)

#endregion

#region: Flatfield
# return a guess for correction intensity, 0 to 50 %
def guess_alpha():
  flatfield_thumb.image.add("name", "flatfield", True)
  brightness: tuple[int,int] = get_brightness_range(flatfield_thumb.image.image, downsample_target=480)
  FF_strength_intput.set(round(((brightness[1]-brightness[0])*100)/510))
flatfield_thumb: Thumbnail = Thumbnail( 
  gui=GUI,
  name="flatfield_thumb",
  thumb_size=THUMBNAIL_SIZE,
  accept_alpha=True,
  func_on_success=guess_alpha)
flatfield_thumb.grid(import_row,import_col+1, rowspan=4)

def show_flatfield(mode: Literal['update', 'slient']='update') -> None:
  highlight_button(flatfield_button)
  # resizeing
  image: Image.Image = flatfield_thumb.image.image
  if(image.size != fit_image(image, GUI.proj.size())):
    debug.info("Resizing image for projection...")
    flatfield_thumb.image.image = image.resize(fit_image(image, GUI.proj.size()), Image.Resampling.LANCZOS)
  if(mode == 'update'):
    debug.info("Showing flatfield image")
  GUI.proj.show(transform_image(flatfield_thumb.image.image))

flatfield_button: Button = Button(
  GUI.root,
  text = 'Show flatfield',
  command = show_flatfield)
flatfield_button.grid(
  row = import_row+4,
  column = import_col+1,
  sticky='nesw')
GUI.add_widget("flatfield_button", flatfield_button)

#endregion

#region: Red Focus
red_focus_thumb: Thumbnail = Thumbnail(
  gui=GUI,
  name="red_focus_thumb",
  thumb_size=THUMBNAIL_SIZE,
  func_on_success=lambda: red_focus_thumb.image.add("name", "red focus", True))
red_focus_thumb.grid(import_row+5,import_col, rowspan=4)

#TODO replace with process_img
def show_red_focus(mode: Literal['update', 'slient']='update') -> None:
  highlight_button(red_focus_button)
  # posterizeing
  image: Image.Image = red_focus_thumb.image.image
  if(posterize_cycle.state and (image.mode != 'L' or post_strength_intput.changed())):
    debug.info("Posterizing image...")
    red_focus_thumb.image.image = posterize(red_focus_thumb.image.image, round((post_strength_intput.get()*255)/100))
    if(mode == 'update'):
      red_focus_thumb.update()
  elif(not posterize_cycle.state and image.mode == 'L'):
    debug.info("Resetting image...")
    red_focus_thumb.image.reset()
    if(mode == 'update'):
      red_focus_thumb.update()
  # resizeing
  image: Image.Image = red_focus_thumb.image.image
  if(image.size != fit_image(image, GUI.proj.size())):
    debug.info("Resizing image for projection...")
    red_focus_thumb.image.image = image.resize(fit_image(image, GUI.proj.size()), Image.Resampling.LANCZOS)
  # color channel toggling
  if(not (uv_focus_red_cycle.state and uv_focus_green_cycle.state and uv_focus_blue_cycle.state)):
    debug.info("Toggling color channels...")
    image: Image.Image = toggle_channels (red_focus_thumb.image.image,
                                          red_focus_red_cycle.state,
                                          red_focus_green_cycle.state,
                                          red_focus_blue_cycle.state)
    if(mode == 'update'):
      red_focus_thumb.update(image)
  if(mode == 'update'):
    debug.info("Showing red focus image")
  GUI.proj.show(transform_image(image))
red_focus_button: Button = Button(
  GUI.root,
  text = 'Show Red Focus',
  command = show_red_focus)
red_focus_button.grid(
  row = import_row+9,
  column = import_col,
  sticky='nesw')
GUI.add_widget("red_focus_button", red_focus_button)
#endregion

#region: UV Focus
uv_focus_thumb: Thumbnail = Thumbnail(
  gui=GUI,
  name="uv_focus_thumb",
  thumb_size=THUMBNAIL_SIZE,
  func_on_success = lambda: uv_focus_thumb.image.add("name", "uv focus", True))
uv_focus_thumb.grid(import_row+5,import_col+1, rowspan=4)

def show_uv_focus(mode: Literal['update', 'slient']='update') -> None:
  highlight_button(uv_focus_button)
  # resizeing
  image: Image.Image = uv_focus_thumb.image.image
  if(image.size != fit_image(image, GUI.proj.size())):
    debug.info("Resizing image for projection...")
    uv_focus_thumb.image.image = image.resize(fit_image(image, GUI.proj.size()), Image.Resampling.LANCZOS)
  # color channel toggling
  if(not (uv_focus_red_cycle.state and uv_focus_green_cycle.state and uv_focus_blue_cycle.state)):
    debug.info("Toggling color channels...")
    image: Image.Image = toggle_channels (uv_focus_thumb.image.image,
                                          uv_focus_red_cycle.state,
                                          uv_focus_green_cycle.state,
                                          uv_focus_blue_cycle.state)
    if(mode == 'update'):
      uv_focus_thumb.update(image)
  if(mode == 'update'):
    debug.info("Showing UV focus image")
  GUI.proj.show(transform_image(image))

uv_focus_button: Button = Button(
  GUI.root,
  text = 'Show UV Focus',
  command = show_uv_focus)
uv_focus_button.grid(
  row = import_row+9,
  column = import_col+1,
  sticky='nesw')
GUI.add_widget("uv_focus_button", uv_focus_button)
#endregion

#endregion

GUI.root.grid_columnconfigure(2, minsize=SPACER_SIZE)

#region: Stage and Fine Adjustment Smart Area

#region: Smart Area
stage_row: int = 3
stage_col: int = 3
#create smart area for stage and fine adjustment
center_area: Smart_Area = Smart_Area(
  gui=GUI,
  debug=debug,
  name="center area")
#create button to toggle between stage and fine adjustment
center_area_cycle: Cycle = Cycle(gui = GUI, name = "center_area_cycle")
center_area_cycle.add_state(text = "- Stage Position (microns) -",
                            enter = lambda: center_area.jump(0))
center_area_cycle.add_state(text = "- Fine Adjustment -",
                            colors = ("black","light blue"),
                            enter = lambda: center_area.jump(1))
center_area_cycle.grid(row = stage_row,
                        col = stage_col,
                        colspan = 3)

#endregion

#region: Stage Control

stage: Stage_Controller = Stage_Controller(
  debug=debug,
  location_query = lambda: GUI.get_coords("camera", CAMERA_IMAGE_SIZE),
  verbosity=1)
def step_update(axis: Literal['-x','+x','-y','+y','-z','+z']):
  # first check if the step size has changed
  if(x_step_intput.changed() or y_step_intput.changed() or z_step_intput.changed()):
    stage.step_size = (x_step_intput.get(), y_step_intput.get(), z_step_intput.get())
  stage.step(axis)

#region: Stage Position

x_intput = Intput(
  gui=GUI,
  name="x_intput",
  default=int(stage.x()))
x_intput.grid(stage_row+1,stage_col,rowspan=2)
stage.update_funcs["x"]["x intput"] = lambda: x_intput.set(int(stage.x()))

y_intput = Intput(
  gui=GUI,
  name="y_intput",
  default=int(stage.y()))
y_intput.grid(stage_row+1,stage_col+1,rowspan=2)
stage.update_funcs["y"]["y intput"] = lambda: y_intput.set(int(stage.y()))

z_intput = Intput(
  gui=GUI,
  name="z_intput",
  default=int(stage.z()))
z_intput.grid(stage_row+1,stage_col+2,rowspan=2)
stage.update_funcs["z"]["z intput"] = lambda: z_intput.set(int(stage.z()))

#endregion

#region: Stage Step size
step_size_row: int = 7

step_size_text: Label = Label(
  GUI.root,
  text = "Stage Step Size (microns)",
  justify = 'center',
  anchor = 'center'
)
step_size_text.grid(
  row = stage_row+step_size_row,
  column = stage_col,
  columnspan = 3,
  sticky='nesw'
)
GUI.add_widget("step_size_text", step_size_text)

x_step_intput = Intput(
  gui=GUI,
  name="x_step_intput",
  default=10,
  min=-1000,
  max=1000)
x_step_intput.grid(stage_row+step_size_row+1,stage_col)

y_step_intput = Intput(
  gui=GUI,
  name="y_step_intput",
  default=10,
  min=-1000,
  max=1000)
y_step_intput.grid(stage_row+step_size_row+1,stage_col+1)

z_step_intput = Intput(
  gui=GUI,
  name="z_step_intput",
  default=10,
  min=-1000,
  max=1000)
z_step_intput.grid(stage_row+step_size_row+1,stage_col+2)

#endregion

#region: stepping buttons
step_button_row = 3
### X axis ###
up_x_button: Button = Button(
  GUI.root,
  text = '+x',
  command = lambda : step_update('+x')
  )
up_x_button.grid(
  row = stage_row+step_button_row,
  column = stage_col,
  sticky='nesw')
GUI.add_widget("up_x_button", up_x_button)

down_x_button: Button = Button(
  GUI.root,
  text = '-x',
  command = lambda : step_update('-x')
  )
down_x_button.grid(
  row = stage_row+step_button_row+1,
  column = stage_col,
  sticky='nesw')
GUI.add_widget("down_x_button", down_x_button)

### Y axis ###
up_y_button: Button = Button(
  GUI.root,
  text = '+y',
  command = lambda : step_update('+y')
  )
up_y_button.grid(
  row = stage_row+step_button_row,
  column = stage_col+1,
  sticky='nesw')
GUI.add_widget("up_y_button", up_y_button)

down_y_button: Button = Button(
  GUI.root,
  text = '-y',
  command = lambda : step_update('-y')
  )
down_y_button.grid(
  row = stage_row+step_button_row+1,
  column = stage_col+1,
  sticky='nesw')
GUI.add_widget("down_y_button", down_y_button)

### Z axis ###
up_z_button: Button = Button(
  GUI.root,
  text = '+z',
  command = lambda : step_update('+z')
  )
up_z_button.grid(
  row = stage_row+step_button_row,
  column = stage_col+2,
  sticky='nesw')
GUI.add_widget("up_z_button", up_z_button)

down_z_button: Button = Button(
  GUI.root,
  text = '-z',
  command = lambda : step_update('-z')
  )
down_z_button.grid(
  row = stage_row+step_button_row+1,
  column = stage_col+2,
  sticky='nesw')
GUI.add_widget("down_z_button", down_z_button)

set_coords_button: Button = Button(
  GUI.root,
  text = 'Set Stage Position',
  command = lambda : stage.set(x_intput.get(), y_intput.get(), z_intput.get())
  )
set_coords_button.grid(
  row = stage_row+step_button_row+2,
  column = stage_col,
  columnspan = 3,
  sticky='nesw')
GUI.add_widget("set_coords_button", set_coords_button)

#endregion

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

center_area.add(0,["set_coords_button",
                   "x_intput",
                   "y_intput",
                   "z_intput",
                   "step_size_text",
                   "x_step_intput",
                   "y_step_intput",
                   "z_step_intput",
                   "up_x_button",
                   "down_x_button",
                   "up_y_button",
                   "down_y_button",
                   "up_z_button",
                   "down_z_button"]) 
center_area.add_func(0,bind_stage_controls, unbind_stage_controls)
#endregion

#region: Fine Adjustment Area

# IMPORTANT:
# to reduce complexity, the fine adjustment area reuses the stage controller class but
# instead of the z field and methods reprersenting the z axis, they represent the theta axis
# this is confusing, but it's better than adding unnecessary complixty to the stage controller class

fine_adjust: Stage_Controller = Stage_Controller(
  debug=debug,
  verbosity=1)
def transform_image(image: Image.Image, theta_factor: float = 0.1) -> Image.Image:
  if(fine_adjustment_cycle.state == 1):
    return better_transform(image, (*round_tuple(fine_adjust.xy()), (2*pi*fine_adjust.z())/360), GUI.proj.size(), border_size_intput.get())
  return image
def update_displayed_image() -> None:
  match showing_state:
    case 'clear':
      return
    case 'pattern':
      show_pattern_fixed(mode='slient')
    case 'flatfield':
      show_flatfield(mode='slient')
    case 'red_focus':
      show_red_focus(mode='slient')
    case 'uv_focus':
      show_uv_focus(mode='slient')
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
  if(fine_x_step_intput.changed() or fine_y_step_intput.changed() or fine_theta_step_floatput.changed()):
    fine_adjust.step_size = (fine_x_step_intput.get(), fine_y_step_intput.get(), fine_theta_step_floatput.get())
  # next update the values
  fine_adjust.step(axis)
  # update image
  update_displayed_image()
  # remove semaphore lock
  fine_step_busy = False

#TODO: make whole background blue
# create a light blue background label with no border first so it appears underneath the other widgets
background_label: Label = Label(
  GUI.root,
  bg="light blue"
  )
background_label.grid(
  row = stage_row+1,
  column = stage_col,
  rowspan = 9,
  columnspan = 3,
  sticky='nesw')
GUI.add_widget("background_label", background_label)

#region: Fine Adjustment Position
fine_x_intput = Intput(
  gui=GUI,
  name="fine_x_intput",
  default=int(fine_adjust.x()))
fine_x_intput.grid(stage_row+1,stage_col,rowspan=2)
fine_adjust.update_funcs["x"]["fine x intput"] = lambda: fine_x_intput.set(int(fine_adjust.x()))

fine_y_intput = Intput(
  gui=GUI,
  name="fine_y_intput",
  default=int(fine_adjust.y()))
fine_y_intput.grid(stage_row+1,stage_col+1,rowspan=2)
fine_adjust.update_funcs["y"]["fine y intput"] = lambda: fine_y_intput.set(int(fine_adjust.y()))

fine_theta_floatput = Floatput(
  gui=GUI,
  name="fine_theta_floatput",
  default=fine_adjust.z())
fine_theta_floatput.grid(stage_row+1,stage_col+2,rowspan=2)
fine_adjust.update_funcs["z"]["fine theta floatput"] = lambda: fine_theta_floatput.set(fine_adjust.z())

#endregion

#region: fine stepping buttons
fine_step_button_row = 3
### X axis ###
fine_up_x_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '+x',
  command = lambda : fine_step_update('+x')
  )
fine_up_x_button.grid(
  row = stage_row+fine_step_button_row,
  column = stage_col,
  sticky='nesw')
GUI.add_widget("fine_up_x_button", fine_up_x_button)

fine_down_x_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '-x',
  command = lambda : fine_step_update('-x')
  )
fine_down_x_button.grid(
  row = stage_row+fine_step_button_row+1,
  column = stage_col,
  sticky='nesw')
GUI.add_widget("fine_down_x_button", fine_down_x_button)

### Y axis ###
fine_up_y_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '+y',
  command = lambda : fine_step_update('+y')
  )
fine_up_y_button.grid(
  row = stage_row+fine_step_button_row,
  column = stage_col+1,
  sticky='nesw')
GUI.add_widget("fine_up_y_button", fine_up_y_button)

fine_down_y_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '-y',
  command = lambda : fine_step_update('-y')
  )
fine_down_y_button.grid(
  row = stage_row+fine_step_button_row+1,
  column = stage_col+1,
  sticky='nesw')
GUI.add_widget("fine_down_y_button", fine_down_y_button)

### Theta ###
fine_up_theta_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '+theta',
  command = lambda : fine_step_update('+z')
  )
fine_up_theta_button.grid(
  row = stage_row+fine_step_button_row,
  column = stage_col+2,
  sticky='nesw')
GUI.add_widget("fine_up_theta_button", fine_up_theta_button)

fine_down_theta_button: Button = Button(
  GUI.root,
  bg = "light blue",
  text = '-theta',
  command = lambda : fine_step_update('-z')
  )
fine_down_theta_button.grid(
  row = stage_row+fine_step_button_row+1,
  column = stage_col+2,
  sticky='nesw')
GUI.add_widget("fine_down_theta_button", fine_down_theta_button)

def set_and_update_fine_adjustment() -> None:
  fine_adjust.set(fine_x_intput.get(), fine_y_intput.get(), fine_theta_floatput.get())
  update_displayed_image()
set_adjustment_button: Button = Button(
  GUI.root,
  text = 'Set Fine Adjustment',
  bg="light blue",
  command = set_and_update_fine_adjustment
  )
set_adjustment_button.grid(
  row = stage_row+fine_step_button_row+2,
  column = stage_col,
  columnspan = 3,
  sticky='nesw')
GUI.add_widget("set_adjustment_button", set_adjustment_button)

#endregion

#region: Fine Adjustment Step size
fine_step_size_row: int = 7

fine_step_size_text: Label = Label(
  GUI.root,
  text = "Fine Adjustment Step Size",
  bg = "light blue",
  justify = 'center',
  anchor = 'center'
)
fine_step_size_text.grid(
  row = stage_row+fine_step_size_row,
  column = stage_col,
  columnspan = 3,
  sticky='nesw'
)
GUI.add_widget("fine_step_size_text", fine_step_size_text)

fine_x_step_intput = Intput(
  gui=GUI,
  name="fine_x_step_intput",
  default=1)
fine_x_step_intput.grid(stage_row+fine_step_size_row+1,stage_col)

fine_y_step_intput = Intput(
  gui=GUI,
  name="fine_y_step_intput",
  default=1)
fine_y_step_intput.grid(stage_row+fine_step_size_row+1,stage_col+1)

fine_theta_step_floatput = Floatput(
  gui=GUI,
  name="fine_theta_step_floatput",
  default=1)
fine_theta_step_floatput.grid(stage_row+fine_step_size_row+1,stage_col+2)

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

center_area.add(1,[ "background_label",
                    "set_adjustment_button",
                    "fine_x_intput",
                    "fine_y_intput",
                    "fine_theta_floatput",
                    "fine_step_size_text",
                    "fine_x_step_intput",
                    "fine_y_step_intput",
                    "fine_theta_step_floatput",
                    "fine_up_x_button",
                    "fine_down_x_button",
                    "fine_up_y_button",
                    "fine_down_y_button",
                    "fine_up_theta_button",
                    "fine_down_theta_button"])
center_area.add_func(1,bind_fine_controls, unbind_fine_controls)
#endregion

center_area.jump(0)

#endregion

GUI.root.grid_columnconfigure(6, minsize=SPACER_SIZE)

#region: patterning and options Smart Area

#region: smart area
pattern_row: int = 3
pattern_col: int = 7
#create smart area for patterning and options
right_area: Smart_Area = Smart_Area(
  gui=GUI,
  debug=debug,
  name="right area")
#create button to toggle between patterning and options
patterning_area_cycle: Cycle = Cycle(gui = GUI, name = "patterning_area_cycle")
patterning_area_cycle.add_state(text = "- Options -",
                                enter = lambda: right_area.jump(0))
patterning_area_cycle.add_state(text = "- Patterning -",
                                colors = ("white", "red"),
                                enter = lambda: right_area.jump(1))
patterning_area_cycle.grid(row = pattern_row,
                        col = pattern_col,
                        colspan = 4)

#endregion

#region: Options
options_row: int = 0
options_col: int = 0

#region: duration
duration_text: Label = Label(
  GUI.root,
  text = "Exposure Time (ms)",
  justify = 'left',
  anchor = 'w'
)
duration_text.grid(
  row = pattern_row+options_row+1,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("duration_text", duration_text)

duration_intput: Intput = Intput(
  gui=GUI,
  name="duration_intput",
  default=1000,
  min = 0)
duration_intput.grid(pattern_row+options_row+1,pattern_col+options_col+1, colspan=3)

#endregion

#region: slicer settings

slicer_horiz_text: Label = Label(
  GUI.root,
  text = "Tiles (horiz, vert)",
  justify = 'left',
  anchor = 'w'
)
slicer_horiz_text.grid(
  row = pattern_row+options_row+2,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("slicer_horiz_text", slicer_horiz_text)

slicer_horiz_intput: Intput = Intput(
  gui=GUI,
  name="slicer_horiz_intput",
  default=0,
  min=0,
)
slicer_horiz_intput.grid(pattern_row+options_row+2,pattern_col+options_col+1)

slicer_vert_intput: Intput = Intput(
  gui=GUI,
  name="slicer_vert_intput",
  default=0,
  min=0,
)
slicer_vert_intput.grid(pattern_row+options_row+2,pattern_col+options_col+2)

slicer_pattern_cycle: Cycle = Cycle(gui=GUI, name="slicer_pattern_cycle")
slicer_pattern_cycle.add_state(text = "Snake", colors=("black","pale green"))
slicer_pattern_cycle.add_state(text = "Row Major", colors=("black","light blue"))
slicer_pattern_cycle.add_state(text = "Col Major", colors=("black","light pink"))
slicer_pattern_cycle.grid(pattern_row+options_row+2,pattern_col+options_col+3)

#endregion

#region: flatfield
FF_strength_text: Label = Label(
  GUI.root,
  text = "Flatfield Strength (%)",
  justify = 'left',
  anchor = 'w'
)
FF_strength_text.grid(
  row = pattern_row+options_row+3,
  column = pattern_col+options_col,
  columnspan=2,
  sticky='nesw'
)
GUI.add_widget("FF_strength_text", FF_strength_text)

FF_strength_intput: Intput = Intput(
  gui=GUI,
  name="FF_strength_intput",
  default=0,
  min = 0,
  max = 100)
FF_strength_intput.grid(pattern_row+options_row+3,pattern_col+options_col+2)

flatfield_cycle: Cycle = Cycle(gui=GUI, name="flatfield_cycle")
flatfield_cycle.add_state(text = "NOT Using Flatfield")
flatfield_cycle.add_state(text = "Using Flatfield", colors=("white", "gray"))
flatfield_cycle.grid(pattern_row+options_row+3,pattern_col+options_col+3)
#endregion

#region: posterize
post_strength_text: Label = Label(
  GUI.root,
  text = "Posterize Cutoff (%)",
  justify = 'left',
  anchor = 'w'
)
post_strength_text.grid(
  row = pattern_row+options_row+4,
  column = pattern_col+options_col,
  columnspan=2,
  sticky='nesw'
)
GUI.add_widget("post_strength_text", post_strength_text)

post_strength_intput: Intput = Intput(
  gui=GUI,
  name="post_strength_intput",
  default=50,
  min=0,
  max=100
)
post_strength_intput.grid(pattern_row+options_row+4,pattern_col+options_col+2)

posterize_cycle: Cycle = Cycle(gui=GUI, name="posterize_cycle")
posterize_cycle.add_state(text = "NOT Posterizing")
posterize_cycle.add_state(text = "Now Posterizing", colors=("white", "gray"))
posterize_cycle.grid(pattern_row+options_row+4,pattern_col+options_col+3)

#endregion

#region: fine adjustment
fine_adjustment_text: Label = Label(
  GUI.root,
  text = "Fine Adj. Border (%)",
  justify = 'left',
  anchor = 'w'
)
fine_adjustment_text.grid(
  row = pattern_row+options_row+5,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("fine_adjustment_text", fine_adjustment_text)

border_size_intput: Intput = Intput(
  gui=GUI,
  name="border_size_intput",
  default=20,
  min=0,
  max=100
)
border_size_intput.grid(pattern_row+options_row+5,pattern_col+options_col+1)

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

#region: calibration

calibration_label: Label = Label(
  GUI.root,
  text = "Calibration Controls",
  justify = 'left',
  anchor = 'w'
)
calibration_label.grid(
  row = pattern_row+options_row+9,
  column = pattern_col+options_col,
  sticky='nesw'
)
GUI.add_widget("calibration_label", calibration_label)

# button to begin calibration
calibrate_button: Button = Button(
  GUI.root,
  text = 'Calibrate',
  command = lambda: stage.calibrate(
    step_size = calibrate_step_intput.get(),
    calibrate_backlash = 'symmetric',
    return_to_start = True)
  )
calibrate_button.grid(
  row = pattern_row+options_row+9,
  column = pattern_col+options_col+1,
  sticky='nesw')
GUI.add_widget("calibrate_button", calibrate_button)

# intput for calibration step size
calibrate_step_intput: Intput = Intput(
  gui=GUI,
  name="calibrate_step_intput",
  default=100,
  min=1)
calibrate_step_intput.grid(pattern_row+options_row+9,pattern_col+options_col+2)

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

right_area.add(0,["duration_text",
                  "duration_intput",
                  "slicer_horiz_text",
                  "slicer_horiz_intput",
                  "slicer_vert_intput",
                  "slicer_pattern_cycle",
                  "FF_strength_text",
                  "FF_strength_intput",
                  "flatfield_cycle",
                  "post_strength_text",
                  "post_strength_intput",
                  "posterize_cycle",
                  "fine_adjustment_text",
                  "border_size_intput",
                  "reset_adj_cycle",
                  "fine_adjustment_cycle",
                  "pattern_rgb_text",
                  "pattern_red_cycle",
                  "pattern_green_cycle",
                  "pattern_blue_cycle",
                  "red_focus_rgb_text",
                  "red_focus_red_cycle",
                  "red_focus_green_cycle",
                  "red_focus_blue_cycle",
                  "uv_focus_rgb_text",
                  "uv_focus_red_cycle",
                  "uv_focus_green_cycle",
                  "uv_focus_blue_cycle",
                  "calibration_label",
                  "calibrate_button",
                  "calibrate_step_intput",
                  "goto_cycle"])

#endregion

#region: Patterning Area

#TODO: make the patterning area the same width as options, it's annoying that it changes

#region: Current Tile
current_tile_row = 1
current_tile_col = 0

Current_tile_text: Label = Label(
  GUI.root,
  text = "Next Pattern Image",
)
Current_tile_text.grid(
  row = pattern_row+current_tile_row,
  column = pattern_col+current_tile_col,
  columnspan = 4,
  sticky='nesw'
)
GUI.add_widget("Current_tile_text", Current_tile_text)

tile_placeholder = rasterize(Image.new('RGB', THUMBNAIL_SIZE, (0,0,0)))
next_tile_image: Label = Label(
  GUI.root,
  image = tile_placeholder,
  justify = 'center',
  anchor = 'center'
)
next_tile_image.grid(
  row = pattern_row+current_tile_row+1,
  column = pattern_col+current_tile_col,
  rowspan=5,
  columnspan=4,
  sticky='nesw'
)
GUI.add_widget("next_tile_image", next_tile_image)

#endregion

# region: Danger Buttons
buttons_row = 7
buttons_col = 0
pattern_rowspan = 3
pattern_colspan = 3
clear_rowspan = 3
clear_colspan = 1

pattern_status: Literal['idle','patterning', 'aborting'] = 'idle'
def change_patterning_status(new_status: Literal['idle','patterning', 'aborting']) -> None:
  global pattern_status
  match pattern_status:
    case 'idle':
      match new_status:
        case 'patterning':
          # reset all "show" buttons
          highlight_button(None)
          # change clear button to abort button
          clear_button.config(
            text='Abort',
            bg='red',
            fg='white',
            command=lambda: change_patterning_status('aborting'))
          clear_button.grid(rowspan=pattern_rowspan if pattern_rowspan == clear_rowspan else pattern_rowspan+clear_rowspan,
                            columnspan=pattern_colspan if pattern_colspan == clear_colspan else pattern_colspan+clear_colspan)
          # disable pattern button
          pattern_button_timed.config(
            command=lambda: None)
          pattern_status = 'patterning'
        case 'aborting':
          debug.warn("invalid state transition: idle -> aborting")
        case 'idle':
          debug.warn("invalid state transition: idle -> idle")
    case 'patterning':
      match new_status:
        case 'idle':
          # normal transition, reset changes
          clear_button.config(
            text='Clear',
            bg='black',
            fg='white',
            command=clear_button_func)
          clear_button.grid(rowspan=clear_rowspan, columnspan=clear_colspan)
          # re-enable pattern button
          pattern_button_timed.config(
            command=begin_patterning)
          pattern_status = 'idle'
        case 'aborting':
          # abort button was pressed while patterning, change global status and warn
          pattern_status = 'aborting'
          GUI.proj.clear()
          debug.warn("aborting patterning...")
        case 'patterning':
          debug.warn("invalid state transition: patterning -> patterning")
    case 'aborting':
      match new_status:
        case 'idle':
          # abort resolved, reset changes
          clear_button.config(
            text='Clear',
            bg='black',
            fg='white',
            command=clear_button_func)
          clear_button.grid(rowspan=clear_rowspan, columnspan=clear_colspan)
          # re-enable pattern button
          pattern_button_timed.config(
            command=begin_patterning)
          pattern_status = 'idle'
        case 'patterning':
          debug.warn("invalid state transition: aborting -> patterning")
        case 'aborting':
          debug.warn("invalid state transition: aborting -> aborting")

# big red danger button
tile_number: int = 0
def begin_patterning():
  def update_next_tile_preview(mode: Literal['current','peek']='peek'):
    #get either current image, or peek ahead to next image
    preview: Image.Image | None
    if(mode=='peek'):
      preview = slicer.peek()
    else:
      preview = slicer.image()
    #if at end of slicer, use blank image
    if(preview == None):
      preview = Image.new('RGB', THUMBNAIL_SIZE)
    raster = rasterize(preview.resize(fit_image(preview, THUMBNAIL_SIZE), Image.Resampling.NEAREST))
    next_tile_image.config(image=raster)
    next_tile_image.image = raster
  
  global pattern_status
  debug.info("Slicing pattern...")
  slicer.update(image=pattern_thumb.image.image,
                horizontal_tiles=slicer_horiz_intput.get(),
                vertical_tiles=slicer_vert_intput.get(),
                tiling_pattern=slicer.pattern_list[slicer_pattern_cycle.state])
  pattern_progress['value'] = 0
  pattern_progress['maximum'] = slicer.tile_count()
  debug.info("Patterning "+str(slicer.tile_count())+" tiles for "+str(duration_intput.get())+"ms \n  Total time: "+str(round((slicer.tile_count()*duration_intput.get())/1000))+"s")
  change_patterning_status('patterning')
  # TODO implement fine adjustment with CV
  # delta_vector: tuple[int,int,float] = (0,0,0)
  while True:
    # update next tile preview
    update_next_tile_preview()
    # get patterning image
    image: Image.Image
    if(slicer.tile_count() == 1):
      image = process_img(pattern_thumb)
    else:
      img = Smart_Image(slicer.image())
      img.add("name", "pattern")
      image = process_img(img)
    image = transform_image(image)
    #TODO apply fine adjustment vector to image
    #TODO remove once camera is implemented
    #camera_image_preview = rasterize(image.resize(fit_image(image, (GUI.window_size[0],(GUI.window_size[0]*9)//16)), Image.Resampling.LANCZOS))
    #camera.config(image=camera_image_preview)
    #camera.image = camera_image_preview
    #pattern
    if(pattern_status == 'aborting'):
      break
    stage.lock()
    debug.info("Patterning tile...")
    result = GUI.proj.show(image, duration=duration_intput.get())
    stage.unlock()
    if(pattern_status == 'aborting'):
      break
    # if(result):
    #   # TODO remove once camera is implemented
    #   camera.config(image=camera_placeholder)
    #   camera.image = camera_placeholder
    # repeat
    if(slicer.next()):
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
  if(pattern_status == 'aborting'):
    debug.warn("Patterning aborted")
  else:
    debug.info("Done")
  # return to idle state
  change_patterning_status('idle')
  
pattern_button_timed: Button = Button(
  GUI.root,
  text = 'Begin\nPatterning',
  command = begin_patterning,
  bg = 'red',
  fg = 'white')
pattern_button_timed.grid(
  row = pattern_row+buttons_row,
  column = pattern_col+buttons_col+1,
  columnspan=pattern_colspan,
  rowspan=pattern_rowspan,
  sticky='nesw')
GUI.add_widget("pattern_button_timed", pattern_button_timed)

# clear button has to come after to show ontop, annoying but inevitable
def clear_button_func():
  # reset all "show" buttons
  pattern_button_fixed.config(bg="white", fg="black")
  red_focus_button.config(bg="white", fg="black")
  uv_focus_button.config(bg="white", fg="black")
  flatfield_button.config(bg="white", fg="black")
  # set global state to clear
  global showing_state
  showing_state = 'clear'
  # clear the projection
  GUI.proj.clear()

clear_button: Button = Button(
  GUI.root,
  text = 'Clear',
  bg='black',
  fg='white',
  command = clear_button_func)
clear_button.grid(
  row = pattern_row+buttons_row,
  column = pattern_col+buttons_col,
  columnspan=clear_colspan,
  rowspan=clear_rowspan,
  sticky='nesw')
GUI.add_widget("clear_button", clear_button)

#endregion

right_area.add(1,["Current_tile_text",
                  "next_tile_image",
                  "pattern_button_timed",
                  "clear_button"])

right_area.jump(0)

#endregion

#region: Stage Control Setup
if(RUN_WITH_STAGE):
  serial_port = serial.Serial(stage_file, baud_rate)
  print(f"Using serial port {serial_port.name}")
  stage_low_level = GrblStage(serial_port, bounds=((-12000,12000),(-12000,12000),(-12000,12000))) 
  stage.step_size = (x_step_intput.get(), y_step_intput.get(), z_step_intput.get()) # init defaults

previous_xyz: tuple[int,int,int] = stage.xyz()
def move_stage():
  global previous_xyz
  current = stage.xyz()
  dx = scale_factor*(current[0]-previous_xyz[0])
  dy = scale_factor*(current[1]-previous_xyz[1])
  dz = scale_factor*(current[2]-previous_xyz[2])
  previous_xyz = stage.xyz()
  # print(f"test {dx} {dy} {dz}", flush=True)
  stage_low_level.move_by({'x':dx,'y':dy,'z':dz})

stage.update_funcs['any'] = {'any': move_stage}
#endregion: Stage Control Setup

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

def benchmark():
  from sys import exit
  from numpy import random
  from time import time
  from os import path, getcwd
  from io import TextIOWrapper
  from datetime import datetime 
  
  debug.warn("!!! Benchmarking !!!")
  center_area.jump(1)
  #options
  iterations = 5
  target_res_size = (3840, 2160) #4K
  processing_repeats: int = 3
  
  #region: create and open log file
  log_file: TextIOWrapper
  if(path.exists(path.join(getcwd(), "benchmark.log"))):
    log_file = open("benchmark.log", "a")
    log_file.write("\n\n")
  else:
    log_file = open("benchmark.log", "w")
    log_file.write("Lithographer Benchmark Log\n\n")
  log_file.write("Benchmark\n")
  log_file.write("| "+str(datetime.now())+"\n")
  log_file.write("| Lithographer Version: "+str(VERSION)+"\n")
  log_file.write("| Iterations: "+str(iterations)+"\n")
  log_file.write("| Resolution: "+str(target_res_size)+"\n")
  log_file.write("| Processing Repeats: "+str(processing_repeats)+"\n")
  log_file.flush()
  #endregion
  
  #region: Test image generation
  log_file.write("Image Generation\n| ")
  images = []
  times = []
  for i in range(iterations):
    temp = time()
    images.append(Image.fromarray((random.rand(target_res_size[1],target_res_size[0],3)*255).astype('uint8')).convert('RGB'))
    times.append(time()-temp)
    log_file.write("#")
    log_file.flush()
  log_file.write("\n")
  log_file.write("| max: "+str(round((max(times)*1000)))+" ms\n")
  log_file.write("| avg: "+str(round((sum(times)*1000)/(iterations)))+" ms\n")
  log_file.write("| min: "+str(round((min(times)*1000)))+" ms\n")
  log_file.flush()
  #endregion
  
  #region: Image processing
  log_file.write("Image processing\n| ")
  posterize_cycle.goto(1)
  flatfield_cycle.goto(1)
  fine_adjustment_cycle.goto(1)
  processed_times = []
  proj_size = GUI.proj.size()
  proj_size = (proj_size[0]//2, proj_size[1]//2)
  for i in range(processing_repeats):
    processed_times.append([])
  # process all the images with fully random settings
  for image in images:
    #randomize transform amounts
    border_size_intput.set(random.randint(0,99))
    fine_x_intput.set(random.randint(-proj_size[0], proj_size[0]))
    fine_y_intput.set(random.randint(-proj_size[1], proj_size[1]))
    fine_theta_floatput.set(random.randint(-360,360))
    set_and_update_fine_adjustment()
    #randomize posterize cutoff
    post_strength_intput.set(random.randint(0,99))
    # randomize flatfield strength
    FF_strength_intput.set(random.randint(0,99))  
    #randomize color channels
    pattern_red_cycle.goto(random.randint(0,2))
    pattern_green_cycle.goto(random.randint(0,2))
    pattern_blue_cycle.goto(random.randint(0,2))
    #set thumb image to this image
    pattern_thumb.image = Smart_Image(image)
    pattern_thumb.image.add("name", "pattern", True)
    pattern_thumb.update()
    #apply processing
    for i in range(processing_repeats):
      temp = time()
      GUI.proj.show(transform_image(process_img(pattern_thumb)))
      processed_times[i].append(time()-temp)
    log_file.write("#")
    log_file.flush()
  log_file.write("\n")
  GUI.proj.clear()
  log_file.write("| First processing:\n")
  log_file.write("| | max: "+str(round((max(processed_times[0])*1000)))+" ms\n")
  log_file.write("| | avg: "+str(round((sum(processed_times[0])*1000)/(iterations)))+" ms\n")
  log_file.write("| | min: "+str(round((min(processed_times[0])*1000)))+" ms\n")
  if(processing_repeats > 1):
    # merge the times for the 3rd+ processing
    repeat_times = []
    for image in range(iterations):
      image_sum = 0
      for rep in range(1, processing_repeats):
        image_sum += processed_times[rep][image]
      repeat_times.append(image_sum/(processing_repeats-1))
    log_file.write("| repeat processing:\n")
    log_file.write("| | max: "+str(round((max(repeat_times)*1000)))+" ms\n")
    log_file.write("| | avg: "+str(round((sum(repeat_times)*1000)/(iterations)))+" ms\n")
    log_file.write("| | min: "+str(round((min(repeat_times)*1000)))+" ms\n")
  log_file.flush()
  #endregion
  
  log_file.close()
  GUI.proj.__TL__.destroy()
  GUI.root.quit()
  exit()
# benchmark()
# cleanup function for graceful program exit
def cleanup():
  print("Patterning GUI closed.")
  GUI.root.destroy()
  if(RUN_WITH_STAGE):  
    serial_port.close()

# attach cleanup function to GUI close event
GUI.root.protocol("WM_DELETE_WINDOW", cleanup)
GUI.debug.info("Debug info will appear here")

if(RUN_WITH_CAMERA):
  setup_camera_from_py()

GUI.mainloop()


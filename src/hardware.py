from PIL import Image
from dataclasses import dataclass
from typing import Optional, Callable, List
from lithographer_lib.img_lib import posterize, toggle_channels, fit_image, convert_to_alpha_channel, dec_to_alpha
from stage_control.stage_controller import StageController
from projector import ProjectorController
import copy

@dataclass
class ImageProcessSettings:
  posterization: Optional[int]
  flatfield: Optional[int]
  color_channels: tuple[bool, bool, bool]
  image_adjust: tuple[float, float, float] # x, y, theta
  size: tuple[int, int] # width, height

@dataclass
class ProcessedImage:
  original: Image.Image
  cached_settings: Optional[ImageProcessSettings]
  cached_processed: Image.Image

  def __init__(self):
    self.original = Image.new('RGB', (1, 1))
    self.cached_processed = Image.new('RGB', (1, 1))
    self.cached_settings = None

  def update(self, image: Optional[Image.Image] = None, settings: Optional[ImageProcessSettings] = None):
    if image is not None:
      self.original = image
      self.cached_settings = None
      assert(settings is not None)

    if settings is not None and settings != self.cached_settings:
      self.cached_settings = settings
      self.cached_processed = process_img(self.original, self.cached_settings)
  
  def processed(self) -> Image.Image:
    return self.cached_processed

def process_img(image: Image.Image, settings: ImageProcessSettings) -> Image.Image:
  #TODO test what order is fastest

  new_image = image
  
  # posterize
  if settings.posterization is not None:
    #debug.info("Posterizing...")
    new_image = posterize(new_image, round(settings.posterization*255/100))
  
  # color channel toggling (must be after posterizing)
  #debug.info("Toggling color channels...")
  new_image = toggle_channels(new_image, *settings.color_channels)

  original_size = new_image.size

  bg = Image.new('RGB', settings.size, 'black')

  if abs(settings.image_adjust[2]) > 0.01: # 1/100th of a degree is effectively nothing
    new_image = new_image.rotate(settings.image_adjust[2], Image.Resampling.BILINEAR, expand=True)
  
  new_cropped_size = fit_image(original_size, settings.size)
  # Could specialize fit_image to avoid loss of precision here
  new_uncropped_size = (round(new_image.size[0] * new_cropped_size[0] / original_size[0]), round(new_image.size[1] * new_cropped_size[1] / original_size[1]))

  new_image = new_image.resize(new_uncropped_size, Image.Resampling.LANCZOS)

  paste_corner = (round(settings.image_adjust[0] + (bg.size[0] - new_image.size[0]) / 2), round(settings.image_adjust[1] + (bg.size[1] - new_image.size[1]) / 2))

  bg.paste(new_image, paste_corner)
  new_image = bg
  
  # flatfield and check to make sure it wasn't applied early
  if settings.flatfield is not None:
    #debug.info("Applying flatfield correction...")
    # TODO: Figure out what this actually does
    alpha_channel = convert_to_alpha_channel(new_image,
                                             new_scale=dec_to_alpha(settings.flatfield),
                                             target_size=new_image.size,
                                             downsample_target=540)
    new_image.putalpha(alpha_channel)
  
  return new_image 

class StageWrapper:
    stage_position: dict[str, float] # position committed to the stage
    target_position: dict[str, float] # position set by the GUI, possibly not yet committed

    stage: StageController

    def __init__(self, stage: StageController, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.stage = stage
        coords = { 'x': origin[0], 'y': origin[1], 'z': origin[2] }
        self.stage_position = copy.copy(coords)
        self.target_position = copy.copy(coords)

    def move_by(self, offsets: dict[str, float], commit=True):
        for axis, offset in offsets.items():
            assert(axis in self.target_position)
            self.target_position[axis] += offset
        
        if commit:
           self.commit()
    
    def move_to(self, position: dict[str, float], commit=True):
        for axis, pos in position.items():
           assert(axis in self.target_position)
           self.target_position[axis] = pos

        if commit:
           self.commit()

    def commit(self):
        self.stage.move_to(self.target_position)
        self.stage_position = copy.copy(self.target_position)

class Lithographer:
    red_focus: ProcessedImage
    uv_focus: ProcessedImage
    pattern: ProcessedImage

    stage: StageWrapper

    projector: ProjectorController

    def __init__(self, stage: StageController, projector: ProjectorController):
        self.red_focus = ProcessedImage()
        self.uv_focus = ProcessedImage()
        self.pattern = ProcessedImage()

        self.stage = StageWrapper(stage)
        self.projector = projector
    
    def sliced_pattern_tile(self, tile_row: int, tile_col: int) -> Image.Image:
        assert(tile_row == 0 and tile_col == 0) # TODO:
        return self.pattern.processed()
    
    def sliced_image(self, tile_row: int, tile_col: int):
        img = self.sliced_pattern_tile(tile_row, tile_col)
        # Processing steps happened before slicing, don't need to reapply them
        # TODO: Flatfield would need to be reapplied here!
        # TODO: Image adjust should be applied here!
        img = process_img(img, ImageProcessSettings(None, None, (True, True, True), (0.0, 0.0, 0.0), self.projector.size()))
        return img
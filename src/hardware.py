from dataclasses import dataclass
from typing import Optional

from PIL import Image

from lib.img import (
    convert_to_alpha_channel,
    dec_to_alpha,
    fit_image,
    posterize,
    select_channels,
)
from projector import ProjectorController
from stage_control.stage_controller import StageController


@dataclass
class ImageProcessSettings:
    posterization: Optional[int]
    flatfield: Optional[int]
    color_channels: tuple[bool, bool, bool]
    image_adjust: tuple[float, float, float]  # x, y, theta
    border_size: float
    size: tuple[int, int]  # width, height


@dataclass
class ProcessedImage:
    original: Image.Image
    cached_settings: Optional[ImageProcessSettings]
    cached_processed: Image.Image

    def __init__(self):
        self.original = Image.new("RGB", (1, 1))
        self.cached_processed = Image.new("RGB", (1, 1))
        self.cached_settings = None

    def update(
        self,
        image: Optional[Image.Image] = None,
        settings: Optional[ImageProcessSettings] = None,
    ):
        if image is not None:
            self.original = image
            self.cached_settings = None
            assert settings is not None

        if settings is not None and settings != self.cached_settings:
            self.cached_settings = settings
            self.cached_processed = process_img(self.original, self.cached_settings)

    def processed(self) -> Image.Image:
        return self.cached_processed


def process_img(image: Image.Image, settings: ImageProcessSettings) -> Image.Image:
    # TODO test what order is fastest

    new_image = image

    # posterize
    if settings.posterization is not None:
        # debug.info("Posterizing...")
        new_image = posterize(new_image, round(settings.posterization * 255 / 100))

    # color channel toggling (must be after posterizing)
    # debug.info("Toggling color channels...")
    new_image = select_channels(new_image, *settings.color_channels)

    bordered_size = (
        round((1.0 - settings.border_size / 100.0) * settings.size[0]),
        round((1.0 - settings.border_size / 100.0) * settings.size[1]),
    )

    original_size = new_image.size

    bg = Image.new("RGB", settings.size, "black")

    if (
        abs(settings.image_adjust[2]) > 0.01
    ):  # 1/100th of a degree is effectively nothing
        new_image = new_image.rotate(
            settings.image_adjust[2], Image.Resampling.BILINEAR, expand=True
        )

    new_cropped_size = fit_image(original_size, bordered_size)
    # Could specialize fit_image to avoid loss of precision here
    new_uncropped_size = (
        round(new_image.size[0] * new_cropped_size[0] / original_size[0]),
        round(new_image.size[1] * new_cropped_size[1] / original_size[1]),
    )

    new_image = new_image.resize(new_uncropped_size, Image.Resampling.LANCZOS)

    paste_corner = (
        round(settings.image_adjust[0] + (bg.size[0] - new_image.size[0]) / 2),
        round(settings.image_adjust[1] + (bg.size[1] - new_image.size[1]) / 2),
    )

    bg.paste(new_image, paste_corner)
    new_image = bg

    # flatfield and check to make sure it wasn't applied early
    if settings.flatfield is not None:
        # debug.info("Applying flatfield correction...")
        # TODO: Figure out what this actually does
        alpha_channel = convert_to_alpha_channel(
            new_image,
            new_scale=dec_to_alpha(settings.flatfield),
            target_size=new_image.size,
            downsample_target=540,
        )
        new_image.putalpha(alpha_channel)

    return new_image


class Lithographer:
    stage: StageController

    projector: ProjectorController

    def __init__(self, stage: StageController, projector: ProjectorController):
        self.stage = stage
        self.projector = projector

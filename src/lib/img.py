# Hacker Fab
# Luca Garlati, 2024
# backend image processing and convenience functions

from math import ceil
from PIL import Image, ImageTk
from PIL.ImageOps import invert


def select_channels(
    image: Image.Image, red: bool = True, green: bool = True, blue: bool = True
) -> Image.Image:
    img_cpy: Image.Image = image.copy()
    # check image is RGB or RGBA
    if img_cpy.mode != "RGB" and img_cpy.mode != "RGBA":
        img_cpy = img_cpy.convert("RGB")

    if red and green and blue:
        return img_cpy

    # at least one channel is off, create blank image for those channels
    blank: Image.Image = Image.new("L", img_cpy.size)

    # split image into channels
    channels: tuple[Image.Image, ...] = img_cpy.split()
    assert len(channels) in (3, 4)

    print("red" if red else "green" if green else "blue")

    return Image.merge(
        "RGB",
        (
            channels[0] if red else blank,
            channels[1] if green else blank,
            channels[2] if blue else blank,
        ),
    )


# return max image size that will fit in [win_size] without cropping
def fit_image(img_size: tuple[int, int], win_size: tuple[int, int]) -> tuple[int, int]:
    # determine orientation to fit to
    if (win_size[0] / win_size[1]) > (img_size[0] / img_size[1]):
        # window wider than image: fit to height
        return (round(img_size[0] * (win_size[1] / img_size[1])), win_size[1])
    elif (win_size[0] / win_size[1]) < (img_size[0] / img_size[1]):
        # image wider than window: fit to width
        return (win_size[0], round(img_size[1] * (win_size[0] / img_size[0])))
    else:
        # same ratio
        return win_size


# return min image size that will fill in window
def fill_image(
    image: Image.Image | tuple[int, int], win_size: tuple[int, int]
) -> tuple[int, int]:
    # for easier access
    img_size: tuple[int, int]
    if type(image) == tuple:
        img_size = image
    elif type(image) == Image.Image:
        img_size = (image.width, image.height)
    # determine orientation to fit to
    if (win_size[0] / win_size[1]) > (img_size[0] / img_size[1]):
        # window wider than image: fit to width
        return (win_size[0], round(img_size[1] * (win_size[0] / img_size[0])))
    elif (win_size[0] / win_size[1]) < (img_size[0] / img_size[1]):
        # image wider than window: fit to height
        return (round(img_size[0] * (win_size[1] / img_size[1])), win_size[1])
    else:
        # same ratio
        return win_size


# return a center cropped version of image at desired resolution
# example: if size == window then this will fill and crop image to window
def center_crop(image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    # copy image
    cropped: Image.Image = image.copy()

    assert crop_size[0] > 0 and crop_size[1] > 0

    # resample image to fill desired size
    cropped = cropped.resize(
        fill_image(image, crop_size), resample=Image.Resampling.LANCZOS
    )

    # determine which orientation needs cropping
    assert cropped.width == crop_size[0] or cropped.height == crop_size[1]
    if cropped.width == crop_size[0]:
        # width matches, crop height
        diff: int = cropped.height - crop_size[1]
        top: int = diff // 2
        bottom: int = cropped.height - top
        # if odd, we're adding an extra row, so subtract one from bottom to correct
        if diff % 2 == 1:
            bottom -= 1
        cropped = cropped.crop((0, top, cropped.width, bottom))
    elif cropped.height == crop_size[1]:
        # height matches, crop width
        diff: int = cropped.width - crop_size[0]
        left: int = diff // 2
        right: int = cropped.width - left
        # if odd, we're adding an extra column, so subtract one from right to correct
        if diff % 2 == 1:
            right -= 1
        cropped = cropped.crop((left, 0, right, cropped.height))

    # done
    return cropped


# convert a value on one scale to the same location on another scale
def rescale_value(
    old_scale: tuple[int, int], new_scale: tuple[int, int], value: int
) -> int:
    if old_scale[0] == old_scale[1]:
        return new_scale[1]
    assert old_scale[0] <= old_scale[1]
    if new_scale[0] == new_scale[1]:
        return new_scale[0]
    assert new_scale[0] < new_scale[1]
    # get % into the scale
    d = (value - old_scale[0]) / (old_scale[1] - old_scale[0])
    # convert to second scale
    return round((d * (new_scale[1] - new_scale[0])) + new_scale[0])


# return the max and min brightness values of an image
# optionally specify downsampling target
def get_brightness_range(
    image: Image.Image, downsample_target: int = 0
) -> tuple[int, int]:
    img_copy: Image.Image = image.copy()
    # first make sure image is single channel
    if img_copy.mode != "L":
        img_copy = img_copy.convert("L")
    # downsample if specified
    if downsample_target > 0:
        while img_copy.width > downsample_target or img_copy.height > downsample_target:
            img_copy = img_copy.resize(
                (img_copy.width // 2, img_copy.height // 2),
                resample=Image.Resampling.NEAREST,
            )
    # get brightness range
    brightness: list[int] = [255, 0]
    for col in range(img_copy.width):
        for row in range(img_copy.height):
            # get single-value brightness since it's grayscale
            pixel: int = img_copy.getpixel((col, row))  # type:ignore
            if pixel < brightness[0]:
                brightness[0] = pixel
            if pixel > brightness[1]:
                brightness[1] = pixel
    return (brightness[0], brightness[1])


# returns a rescaled copy of an alpha mask
# can be slow on larger images, only accepts L format images
def rescale(image: Image.Image, new_scale: tuple[int, int]) -> Image.Image:
    assert image.mode == "L"
    mask: Image.Image = image.copy()
    # first step is getting brightest and darkest pixel values
    brightness: tuple[int, int] = get_brightness_range(mask)
    # now rescale each pixel
    lut: dict = {}
    for col in range(mask.width):
        for row in range(mask.height):
            # get pixel and lookup
            pixel: int = mask.getpixel((col, row))  # type:ignore
            lookup: int = lut.get(pixel, -1)
            if lookup == -1:
                lookup = rescale_value((brightness[0], brightness[1]), new_scale, pixel)
            mask.putpixel((col, row), lookup)
    return mask


# returns an alpha channel mask equivalent from source image
# optionally specify new scale
# optionally specify new cropped size
def convert_to_alpha_channel(
    input_image: Image.Image,
    new_scale: tuple[int, int] | None = None,
    target_size: tuple[int, int] = (0, 0),
    downsample_target: int = 1080,
) -> Image.Image:
    # copy the image
    mask: Image.Image = input_image.copy()
    # convert it to grayscale to normalize all values
    if mask.mode != "L":
        mask = mask.convert("L")

    # Invert all colors since we want the mask, not the image itself
    mask = invert(mask)
    if new_scale is not None:
        # if no target, save current size
        if target_size == (0, 0):
            target_size = mask.size
        # downsample
        while mask.width > downsample_target or mask.height > downsample_target:
            mask = mask.resize(
                (mask.width // 2, mask.height // 2), resample=Image.Resampling.NEAREST
            )
        # rescale
        mask = rescale(mask, new_scale)
        # resample to desired dimentions
        mask = center_crop(mask, target_size)
    elif target_size != (0, 0):
        mask = center_crop(mask, target_size)
    # done
    return mask


# actually posterize an image since pil.posterize doesn't work
# optionally specify threashold
def posterize(Input_image: Image.Image, threashold: int = 127) -> Image.Image:
    output_image: Image.Image = Input_image.copy()
    output_image = output_image.convert("L")
    output_image = output_image.point(lambda p: 255 if p > threashold else 0)
    return output_image


# This function is just a wrapper for ImageTk.PhotoImage() because of a bug
# for whatever reason, photoimage removes the alpha channel from LA images
# so this converts inputted LA images to RGBA before passing to PhotoImage
def image_to_tk_image(image: Image.Image) -> ImageTk.PhotoImage:
    if image.mode == "LA":
        return ImageTk.PhotoImage(image.convert("RGBA"))
    else:
        return ImageTk.PhotoImage(image)


# convert from 0 to 100 intensity scale to tuple values
# 0   = (255, 255)
# 50  = (0,   255)
# 100 = (0,   0)
def dec_to_alpha(dec: int) -> tuple[int, int]:
    if dec < 0:
        return (255, 255)
    if dec <= 50:
        return (255 - ceil((255 * dec) / 50), 255)
    if dec <= 100:
        return (0, 255 - ceil((255 * (dec - 50)) / 50))
    return (0, 0)

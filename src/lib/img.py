# Hacker Fab
# Luca Garlati, 2024
# backend image processing and convenience functions

from math import ceil, cos, pi, sin

from PIL import Image, ImageTk
from PIL.ImageOps import invert

from .tuple import *


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


# yeah would work but the


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


# return an alpha mask image applied to another image
def apply_mask(
    input_image: Image.Image,
    mask_image: Image.Image,
    new_scale: tuple[int, int] = (0, 255),
) -> Image.Image:
    # setup
    input_img_copy: Image.Image = input_image.copy()
    target_size: tuple[int, int] = input_image.size
    # create mask and apply
    alpha_mask: Image.Image = convert_to_alpha_channel(
        mask_image, new_scale=new_scale, target_size=target_size
    )
    input_img_copy.putalpha(alpha_mask)
    # flatten image
    background: Image.Image = Image.new("RGB", target_size)
    background.paste(input_img_copy, (0, 0), input_img_copy)
    # return image
    return background


# actually posterize an image since pil.posterize doesn't work
# optionally specify threashold
def posterize(Input_image: Image.Image, threashold: int = 127) -> Image.Image:
    output_image: Image.Image = Input_image.copy()
    output_image = output_image.convert("L")
    output_image = output_image.point(lambda p: 255 if p > threashold else 0)
    return output_image


# returns a copy of the input image without alpha channel
def RGBA_to_RGB(image: Image.Image) -> Image.Image:
    assert image.mode == "RGBA"
    channels: tuple[Image.Image, ...] = image.split()
    assert len(channels) == 4
    return Image.merge("RGB", channels[0:3])


# returns a copy of the input image without alpha channel
def LA_to_L(image: Image.Image) -> Image.Image:
    assert image.mode == "LA"
    channels: tuple[Image.Image, ...] = image.split()
    assert len(channels) == 2
    return Image.merge("L", [channels[0]])


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


def alpha_to_dec(alpha: tuple[int, int]) -> int:
    return int(((510 - alpha[0] - alpha[1]) * 100) / 510)


# given an x, y, and theta transform, return tuple (a,b,c,d,e,f)
# representing the following affine matrix
# | a b c |
# | d e f |
# | 0 0 1 |
# theta is a tuple of format (x distance to origin, y distance to origin, rotation in radians)
def build_affine(
    x: float = 0, y: float = 0, theta: None | tuple[int, int, float] = None
) -> tuple[float, ...]:
    if theta == None or theta[2] == 0:
        # nice, simple translation matrix :)
        return (1, 0, x, 0, 1, y)
    else:
        # horribly ugly translation and rotation around center matrix :(
        # in order, translate to origin, rotate, translate back, translate by xy
        # {{Cos[θ], -Sin[θ], a - a Cos[θ] + x Cos[θ] + b Sin[θ] - y Sin[θ]}, {Sin[θ], Cos[θ], b - b Cos[θ] + y Cos[θ] - a Sin[θ] + x Sin[θ]}, {0, 0, 1}}
        theta = (theta[0], theta[1], -theta[2])
        dx: int = theta[0]
        dy: int = theta[1]
        r: float = theta[2]
        return (
            cos(r),
            -sin(r),
            dx - dx * cos(r) + x * cos(r) + dy * sin(r) - y * sin(r),
            sin(r),
            cos(r),
            dy - dy * cos(r) + y * cos(r) - dx * sin(r) + x * sin(r),
        )


# Summary:
#   transforms input image by x, y, and theta within new output sized image with
#   specified border size.
# Inputs:
#   [image] is the input image to be transformed
#   [vector] is of format (x, y, theta) with x and y being in pixels and theta in *RADS*
#   [output_size] is the size of the output image in pixels
#   [border] is a percentage of output image size.
#     Can specify distinct (x, y) percentages, or just one to apply to both. for a 100x100 image:
#     0% border would be 100x100
#     50% border would be 50x50 (25% off each side)
#     100% border would display zero pixels of image
def better_transform(
    image: Image.Image,
    vector: tuple[int, int, int | float],
    output_size: tuple[int, int],
    border: float,
) -> Image.Image:
    final_image: Image.Image = Image.new("RGB", output_size)
    img_cpy: Image.Image = image.copy()

    # prepare image
    # if border is 100%, return a black image
    if border < 0 or border >= 100:
        return final_image
    # next we need to scale image to the requested size
    fit_size: tuple[int, int] = fit_image(
        image.size,
        (
            round(output_size[0] * ((100 - border) / 100)),
            round(output_size[1] * ((100 - border) / 100)),
        ),
    )
    img_cpy = img_cpy.resize(fit_size, resample=Image.Resampling.LANCZOS)

    # compute the affine matrix to move the image to the center of the output, then the vector, then back
    affine_matrix: tuple[float, ...] = build_affine(
        -vector[0] - (output_size[0] - fit_size[0]) // 2,
        vector[1] - (output_size[1] - fit_size[1]) // 2,
        (fit_size[0] // 2, fit_size[1] // 2, -vector[2]),
    )
    img_cpy = Image.Image.transform(
        img_cpy,
        output_size,
        Image.Transform.AFFINE,
        affine_matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor="black",
    )
    return img_cpy


# slices image into parts
def slice_image(
    image: Image.Image,
    horizontal_tiles: int = 0,
    vertical_tiles: int = 0,
    output_resolution: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], tuple[Image.Image, ...]]:
    # if no parameters specified, return original image
    if horizontal_tiles <= 0 and vertical_tiles <= 0 and output_resolution == (0, 0):
        return ((1, 1), (image.copy(),))

    input_ratio: float = image.size[0] / image.size[1]
    output_ratio: float
    if output_resolution == (0, 0):
        output_ratio = image.size[0] / image.size[1]
    else:
        output_ratio = output_resolution[0] / output_resolution[1]

    grid: tuple[int, int]
    slice_size: tuple[int, int]
    if horizontal_tiles > 0 and vertical_tiles > 0:
        # both specified, make this the new ratio
        output_ratio = horizontal_tiles / vertical_tiles
        grid = (horizontal_tiles, vertical_tiles)
        slice_size = (
            ceil(image.size[0] / horizontal_tiles),
            ceil(image.size[1] / vertical_tiles),
        )
    elif horizontal_tiles <= 0 and vertical_tiles <= 0:
        # neither specified, use output resolution
        grid = (
            ceil(image.size[0] / output_resolution[0]),
            ceil(image.size[1] / output_resolution[1]),
        )
        slice_size = output_resolution
    elif horizontal_tiles > 0:
        temp: float = input_ratio * 1 / output_ratio * horizontal_tiles
        grid = (ceil(temp), horizontal_tiles)
        slice_size = (
            ceil(image.size[0] / temp),
            ceil(image.size[1] / horizontal_tiles),
        )
    elif vertical_tiles > 0:
        temp: float = output_ratio * 1 / input_ratio * vertical_tiles
        grid = (vertical_tiles, ceil(temp))
        slice_size = (ceil(image.size[0] / vertical_tiles), ceil(image.size[1] / temp))
    else:
        # this is unreachable, but it makes the linter happy
        grid = (1, 1)
        slice_size = image.size

    output: list[Image.Image] = []
    for row in range(grid[1]):
        for col in range(grid[0]):
            cropped: Image.Image = image.crop(
                (
                    col * slice_size[0],
                    row * slice_size[1],
                    (col + 1) * slice_size[0],
                    (row + 1) * slice_size[1],
                )
            )
            if output_resolution != (0, 0):
                cropped = cropped.resize(
                    output_resolution, resample=Image.Resampling.LANCZOS
                )
            output.append(cropped)
    return (grid, tuple(output))


# automated test suite
def __run_tests():
    # will print a and b on fail
    def print_assert(a, b, name: str = ""):
        if a == b:
            assert True
        if a != b:
            print(name, a, "!=", b)
            assert False

    dim0 = (50, 200)
    dim1 = (150, 100)
    dim2 = (177, 377)
    dim3 = (277, 77)
    img0 = Image.new("RGBA", dim0)
    img1 = Image.new("RGBA", dim1)
    img2 = Image.new("RGBA", dim2)
    img3 = Image.new("RGBA", dim3)

    # fit to height
    print_assert(fit_image(img0.size, dim1), (25, 100))
    # fit to width
    print_assert(fit_image(img1.size, dim0), (50, 33))
    # fill to width
    print_assert(fill_image(img0, dim1), (150, 600))
    # fill to height
    print_assert(fill_image(img1, dim0), (300, 200))

    old_scale = (10, 110)
    new_scale = (5, 15)
    # min value check
    print_assert(rescale_value(old_scale, new_scale, 10), 5)
    # max value check
    print_assert(rescale_value(old_scale, new_scale, 110), 15)
    # 7%
    print_assert(rescale_value(old_scale, new_scale, 7 + 10), 1 + 5)
    # 70%
    print_assert(rescale_value(old_scale, new_scale, 70 + 10), 7 + 5)

    # check both width and height crops for...
    # ...even...
    print_assert(center_crop(img0, dim1).size, dim1)
    print_assert(center_crop(img1, dim0).size, dim0)
    # ...odd...
    print_assert(center_crop(img2, dim3).size, dim3)
    print_assert(center_crop(img3, dim2).size, dim2)
    # ...mixed...
    print_assert(center_crop(img0, dim3).size, dim3)
    print_assert(center_crop(img1, dim2).size, dim2)
    print_assert(center_crop(img2, dim1).size, dim1)
    print_assert(center_crop(img3, dim0).size, dim0)
    # ...same
    print_assert(center_crop(img0, dim0).size, dim0)
    print_assert(center_crop(img2, dim2).size, dim2)

    # inputs for visual tests
    # image: Image.Image = Image.open(filedialog.askopenfilename(title ='Test Image')).copy()
    # mask: Image.Image  = Image.open(filedialog.askopenfilename(title ='Test mask')).copy()

    # apply_mask(image, mask).show()

    # (0,127) is incorrect, but should still work
    alphas = [(0, 0), (0, 127), (0, 255), (127, 255), (255, 255)]
    # correct conversions
    decs = [100, 75, 50, 25, 0]
    for i in range(len(alphas)):
        print_assert(alpha_to_dec(alphas[i]), decs[i], str(i) + ":0")
        print_assert(dec_to_alpha(decs[i]), alphas[i], str(i) + ":1")
        print_assert(dec_to_alpha(alpha_to_dec(alphas[i])), alphas[i], str(i) + ":2")
        print_assert(alpha_to_dec(dec_to_alpha(decs[i])), decs[i], str(i) + ":3")
    # test every number, why not
    for i in range(100):
        print_assert(alpha_to_dec(dec_to_alpha(i)), i, str(i) + ":4")

    print_assert(add((1, 2, 3), (3, 2, 1)), (4, 4, 4))
    print_assert(add((1, 2, 3), 1), (2, 3, 4))
    print_assert(add(1, (3, 2, 1)), (4, 3, 2))
    print_assert(add(1, 1), 2)

    print_assert(mult((1, 2, 3), (3, 2, 1)), (3, 4, 3))
    print_assert(mult((1, 2, 3), 2), (2, 4, 6))
    print_assert(mult(2, (3, 2, 1)), (6, 4, 2))
    print_assert(mult(2, 2), 4)

    # test select_channels
    black: Image.Image = Image.new("RGB", (1, 1), (0, 0, 0))
    red: Image.Image = Image.new("RGB", (1, 1), (255, 0, 0))
    green: Image.Image = Image.new("RGB", (1, 1), (0, 255, 0))
    blue: Image.Image = Image.new("RGB", (1, 1), (0, 0, 255))
    purple: Image.Image = Image.new("RGB", (1, 1), (255, 0, 255))
    yellow: Image.Image = Image.new("RGB", (1, 1), (255, 255, 0))
    cyan: Image.Image = Image.new("RGB", (1, 1), (0, 255, 255))
    white: Image.Image = Image.new("RGB", (1, 1), (255, 255, 255))
    test_colors: list[Image.Image] = [
        black,
        red,
        green,
        blue,
        purple,
        yellow,
        cyan,
        white,
    ]

    # test disabling all channels
    for color in test_colors:
        print_assert(select_channels(color, False, False, False), black)
    # test disabling channels
    print_assert(select_channels(white, False, False, False), black)
    print_assert(select_channels(white, False, False, True), blue)
    print_assert(select_channels(white, False, True, False), green)
    print_assert(select_channels(white, False, True, True), cyan)
    print_assert(select_channels(white, True, False, False), red)
    print_assert(select_channels(white, True, False, True), purple)
    print_assert(select_channels(white, True, True, False), yellow)
    print_assert(select_channels(white, True, True, True), white)
    # test enabling empty channels
    for color in test_colors:
        print_assert(select_channels(color, True, True, True), color)
    # test bad image modes
    L_image: Image.Image = Image.new("L", (1, 1), 255)
    LA_image: Image.Image = Image.new("LA", (1, 1), (255, 255))
    CMYK_image: Image.Image = Image.new("CMYK", (1, 1), (0, 0, 0, 0))
    test_modes: list[Image.Image] = [L_image, LA_image, CMYK_image]
    for mode in test_modes:
        print_assert(select_channels(mode, True, True, True), white, mode.mode)
        print_assert(select_channels(mode, False, False, False), black, mode.mode)

    # between tuple tests
    print_assert(BTW_tuple(1, 2, 3), True)
    print_assert(BTW_tuple(1, 1, 3), True)
    print_assert(BTW_tuple(1, 2, 2), True)
    print_assert(BTW_tuple(1, 2, 1), False)
    print_assert(BTW_tuple(1, 1, 1), True)
    # mixed tuple tests
    print_assert(BTW_tuple((1, 1, 1), 2, (3, 3, 3)), True)
    print_assert(BTW_tuple(1, (1, 1, 1), 3), True)
    print_assert(BTW_tuple(1, 2, (2, 2, 2)), True)
    print_assert(BTW_tuple((1, 1, 1), 2, (1, 1, 1)), False)
    print_assert(BTW_tuple((1, 1, 1), 1, 1), True)
    print_assert(BTW_tuple(1, (1, 1, 1), 1), True)
    print_assert(BTW_tuple(1, 1, (1, 1, 1)), True)
    print_assert(BTW_tuple((1, 1, 1), 2, (1, 1, 1)), False)

    print("All tests passed")


# __run_tests()

# performance benchmarks for transform
if False:
    from random import randint, random
    from time import time

    import numpy

    def __timing_tests(resolution: tuple[int, int], runs: int = 10):
        print("Running", runs, "tests at", resolution, "...")
        overall = time()
        total = 0
        for i in range(runs):
            # img = Image.new("RGB", resolution, (randint(0,255),randint(0,255),randint(0,255)))
            img = Image.fromarray(
                (numpy.random.rand(resolution[0], resolution[1], 3) * 255).astype(
                    "uint8"
                )
            ).convert("RGB")
            start = time()
            better_transform(
                img,
                (
                    randint(-resolution[0], resolution[0]),
                    randint(-resolution[1], resolution[1]),
                    random() * 2 * pi,
                ),
                resolution,
                random() * 0.9,
            )
            total += time() - start
        print("| finished in " + str(int(time() - overall)) + "s")
        print("| average of  " + str(int(total / runs * 1000)) + "ms")

    print("Tranform timing tests:")
    __timing_tests((1920, 1080), 50)
    __timing_tests((2560, 1440), 50)
    __timing_tests((3840, 2160), 50)

# propt user for image
# image: Image.Image = Image.open(filedialog.askopenfilename(title ='Test Image')).copy()
# for i in slice(image, vertical_tiles=2)[1]: i.show()
# better_transform(image, (0,0,0), (1500,1500), 0.2).show()
# better_transform(image, (100,-50,1), (1500,1500), 0.2).save("test.png")

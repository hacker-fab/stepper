# Hacker Fab
# Tiling utilities for pattern slicing and stage positioning

from typing import List, Tuple
from PIL import Image


def generate_snake_sequence(x_amount: int, y_amount: int) -> List[Tuple[int, int]]:
    """Generates (x_idx, y_idx) tile coordinate list in a boustrophedon (snake) pattern.

    Left to right on even rows, right to left on odd rows.
    """
    coords: List[Tuple[int, int]] = []
    for y_idx in range(y_amount):
        if y_idx % 2 == 0:
            for x_idx in range(x_amount):
                coords.append((x_idx, y_idx))
        else:
            for x_idx in range(x_amount - 1, -1, -1):
                coords.append((x_idx, y_idx))
    return coords


def calculate_tile_position(
    x_start: float,
    y_start: float,
    x_dir: int,
    y_dir: int,
    x_idx: int,
    y_idx: int,
    x_offset: float,
    y_offset: float,
) -> Tuple[float, float]:
    """Calculates target absolute stage coordinates for a given tile index."""
    target_x = x_start + x_dir * x_idx * x_offset
    target_y = y_start + y_dir * y_idx * y_offset
    return target_x, target_y


def split_image_into_tiles(
    img: Image.Image,
    tile_width: int = 3840,
    tile_height: int = 2160,
    overlap_x: int = 200,
    overlap_y: int = 200,
) -> Tuple[List[Image.Image], int, int]:
    """Splits an in-memory PIL Image into overlapping tiles arranged in snake order.

    Returns:
        (tiles_in_snake_order, x_tiles_count, y_tiles_count)
    """
    img_w, img_h = img.size
    stride_x = max(1, tile_width - overlap_x)
    stride_y = max(1, tile_height - overlap_y)

    # Compute all top-left coordinates
    x_positions: List[int] = []
    y_positions: List[int] = []

    # Horizontal positions
    x = 0
    while True:
        if x + tile_width >= img_w:
            x = max(0, img_w - tile_width)
            x_positions.append(x)
            break
        x_positions.append(x)
        x += stride_x

    # Vertical positions
    y = 0
    while True:
        if y + tile_height >= img_h:
            y = max(0, img_h - tile_height)
            y_positions.append(y)
            break
        y_positions.append(y)
        y += stride_y

    x_count = len(x_positions)
    y_count = len(y_positions)

    # Extract 2D grid of tiles
    tile_grid: dict[Tuple[int, int], Image.Image] = {}
    for y_idx, top in enumerate(y_positions):
        for x_idx, left in enumerate(x_positions):
            right = min(img_w, left + tile_width)
            bottom = min(img_h, top + tile_height)
            box = (left, top, right, bottom)
            tile = img.crop(box)
            # If cropped tile is smaller than expected tile size, paste onto background
            if tile.size != (tile_width, tile_height):
                bg = Image.new(img.mode, (tile_width, tile_height), 0)
                bg.paste(tile, (0, 0))
                tile = bg
            tile_grid[(x_idx, y_idx)] = tile

    # Order tiles according to snake sequence
    snake_coords = generate_snake_sequence(x_count, y_count)
    ordered_tiles: List[Image.Image] = [tile_grid[c] for c in snake_coords]

    return ordered_tiles, x_count, y_count


def split_image_with_overlap(
    image_path: str,
    tile_width: int = 3840,
    tile_height: int = 2160,
    overlap_x: int = 200,
    overlap_y: int = 200,
    output_dir: str = "tiles",
) -> Tuple[int, int, int, Tuple[int, int]]:
    """Takes an arbitrary sized image and splits it into overlapping tiles on disk.

    Returns:
        (x_tiles_count, y_tiles_count, total_tiles_count, (img_w, img_h))
    """
    import os
    img = Image.open(image_path)
    img_w, img_h = img.size
    os.makedirs(output_dir, exist_ok=True)

    stride_x = max(1, tile_width - overlap_x)
    stride_y = max(1, tile_height - overlap_y)

    x_positions: List[int] = []
    y_positions: List[int] = []

    x = 0
    while True:
        if x + tile_width >= img_w:
            x = max(0, img_w - tile_width)
            x_positions.append(x)
            break
        x_positions.append(x)
        x += stride_x

    y = 0
    while True:
        if y + tile_height >= img_h:
            y = max(0, img_h - tile_height)
            y_positions.append(y)
            break
        y_positions.append(y)
        y += stride_y

    tile_count = 0
    for tile_id_y, top in enumerate(y_positions):
        for tile_id_x, left in enumerate(x_positions):
            right = min(img_w, left + tile_width)
            bottom = min(img_h, top + tile_height)
            box = (left, top, right, bottom)
            tile = img.crop(box)
            tile.save(os.path.join(output_dir, f"tile_{tile_id_y},{tile_id_x}.png"))
            tile_count += 1

    return len(x_positions), len(y_positions), tile_count, (img_w, img_h)

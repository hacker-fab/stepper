# Hacker Fab
# Tiling utilities for pattern slicing and stage positioning using NumPy and OpenCV

from typing import List, Tuple
import cv2
import numpy as np


def generate_snake_sequence(x_tiles: int, y_tiles: int) -> List[Tuple[int, int]]:
    """Generates (x_idx, y_idx) tile coordinate list in a boustrophedon (snake) pattern.

    Left to right on even rows, right to left on odd rows.
    """
    coords: List[Tuple[int, int]] = []
    for y_idx in range(y_tiles):
        if y_idx % 2 == 0:
            for x_idx in range(x_tiles):
                coords.append((x_idx, y_idx))
        else:
            for x_idx in range(x_tiles - 1, -1, -1):
                coords.append((x_idx, y_idx))
    return coords

def split_image_into_tiles(
    img: np.ndarray,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Splits an in-memory NumPy image into overlapping tiles arranged in snake order.

    Returns:
        (tiles_in_snake_order, snake_coords)
    """
    img_h, img_w = img.shape[:2]
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
    tile_grid: dict[Tuple[int, int], np.ndarray] = {}
    for y_idx, top in enumerate(y_positions):
        for x_idx, left in enumerate(x_positions):
            tile = img[top : top + tile_height, left : left + tile_width]
            th, tw = tile.shape[:2]
            # If cropped tile is smaller than expected tile size, pad onto background
            if tw != tile_width or th != tile_height:
                if img.ndim == 3:
                    bg = np.zeros((tile_height, tile_width, img.shape[2]), dtype=img.dtype)
                else:
                    bg = np.zeros((tile_height, tile_width), dtype=img.dtype)
                bg[:th, :tw] = tile
                tile = bg
            tile_grid[(x_idx, y_idx)] = tile

    # Order tiles according to snake sequence
    snake_coords = generate_snake_sequence(x_count, y_count)
    ordered_tiles: List[np.ndarray] = [tile_grid[c] for c in snake_coords]

    return ordered_tiles, snake_coords


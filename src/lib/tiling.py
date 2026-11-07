# Hacker Fab
# Tiling utilities for pattern slicing and stage positioning using NumPy and OpenCV

from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from core.coordinate_system import (
    apply_homography_2d,
    build_projector_to_chip_matrix,
)


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


def fit_pattern_to_projector_canvas(
    img: np.ndarray,
    projector_size: Tuple[int, int] = (1920, 1080),
    nearest_interp: bool = True,
) -> np.ndarray:
    """Fits and centers a pattern image onto the projector canvas (Fullscreen Projector mode)."""
    proj_w, proj_h = int(projector_size[0]), int(projector_size[1])
    img_h, img_w = img.shape[:2]
    if proj_w <= 0 or proj_h <= 0 or img_w <= 0 or img_h <= 0:
        return img.copy()

    if img_w == proj_w and img_h == proj_h:
        return img.copy()

    scale = min(proj_w / float(img_w), proj_h / float(img_h))
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))
    interp = cv2.INTER_NEAREST if nearest_interp else (cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if resized.ndim == 2:
        canvas = np.zeros((proj_h, proj_w), dtype=resized.dtype)
    else:
        canvas = np.zeros((proj_h, proj_w, resized.shape[2]), dtype=resized.dtype)

    x_off = (proj_w - new_w) // 2
    y_off = (proj_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def resolve_layer_physical_size(
    img_shape: Tuple[int, ...],
    width_um: float,
    height_um: float,
    projector_size: Tuple[int, int] = (1920, 1080),
    projector_pixel_size: float = 1.0,
) -> Tuple[float, float]:
    """Resolves (w_um, h_um) from layer width_um and height_um, inferring from aspect ratio if one is <= 0.
    If both are <= 0 (fullscreen mode), resolves to the calculated projector size in physical units (µm).
    """
    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    pw, ph = int(projector_size[0]), int(projector_size[1])
    proj_px = float(projector_pixel_size) if float(projector_pixel_size) > 0 else 1.0
    if width_um > 0 and height_um > 0:
        return float(width_um), float(height_um)
    elif width_um > 0 and height_um <= 0:
        w = float(width_um)
        h = w * (float(img_h) / max(1.0, float(img_w)))
        return w, h
    elif height_um > 0 and width_um <= 0:
        h = float(height_um)
        w = h * (float(img_w) / max(1.0, float(img_h)))
        return w, h
    return float(pw * proj_px), float(ph * proj_px)


def generate_projector_aligned_tiles(
    img: np.ndarray,
    width_um: float,
    height_um: float,
    overlap_x_um: float = 5.0,
    overlap_y_um: float = 5.0,
    projector_size: Tuple[int, int] = (1920, 1080),
    camera_size: Tuple[int, int] = (1920, 1080),
    projector_pixel_size: float = 1.0,
    camera_pixel_size: float = 1.0,
    proj_to_cam_matrix: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    corr_to_chip_matrix: Optional[np.ndarray] = None,
    pattern_transform_chip: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    nearest_interp: bool = True,
    single_fullscreen_tile: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Generates projector-space tiles covering the layer's chip-space area.

    If single_fullscreen_tile is True and both width_um <= 0 and height_um <= 0,
    operates in Fullscreen Projector single-tile mode (single tile fitted to the projector
    canvas at offset (0.0, 0.0), e.g. for Quick Expose Here).
    Otherwise, when width_um <= 0 and height_um <= 0, resolves the physical footprint
    using the calculated projector FOV and generates standard projector-aligned tiles
    (tiling behavior covering the chip-space footprint under rotation).

    Otherwise:
      1. Resolves physical dimensions (w_um, h_um) in chip-space.
      2. Determines the projector footprint and axis unit vectors in chip-space.
      3. Steps full projector frames along projector axes with (overlap_x_um, overlap_y_um)
         in snake order to cover the rotated chip-space bounding box.
      4. Warps the layer image into each projector tile using cv2.warpPerspective.

    Returns:
        (tiles, tile_rel_chip_coords) where each coord is (dx_chip_um, dy_chip_um)
        relative to the pattern center in chip-space.
    """
    pw, ph = int(projector_size[0]), int(projector_size[1])
    img_h, img_w = img.shape[:2]

    if single_fullscreen_tile and width_um <= 0 and height_um <= 0:
        # Fullscreen projector single-tile mode (e.g. quick expose here)
        tile = fit_pattern_to_projector_canvas(img, projector_size=(pw, ph), nearest_interp=nearest_interp)
        return [tile], [(0.0, 0.0)]

    w_um, h_um = resolve_layer_physical_size(
        img.shape,
        width_um,
        height_um,
        projector_size=(pw, ph),
        projector_pixel_size=projector_pixel_size,
    )
    if w_um <= 0 or h_um <= 0:
        # Fallback if size could not be determined
        tile = fit_pattern_to_projector_canvas(img, projector_size=(pw, ph), nearest_interp=nearest_interp)
        return [tile], [(0.0, 0.0)]

    # 1. Layer image pixel (u_img, v_img) -> local pattern-space (centered at (0, 0)),
    # then transformed into local chip-space via pattern_transform_chip (identity by default).
    M_img_to_pat = np.array(
        [
            [w_um / float(img_w), 0.0, -0.5 * w_um],
            [0.0, h_um / float(img_h), -0.5 * h_um],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    M_pat_to_chip = np.eye(3, dtype=np.float64)
    if pattern_transform_chip is not None:
        arr = np.asarray(pattern_transform_chip, dtype=np.float64)
        if arr.shape == (3, 3):
            M_pat_to_chip = arr
        elif arr.shape == (2, 2):
            M_pat_to_chip[:2, :2] = arr

    M_img_to_chip = M_pat_to_chip @ M_img_to_pat

    # 2. Projector pixel -> chip-space at reference stage_xy_corr = (0, 0),
    # shifted so projector center (pw/2, ph/2) is at local chip-space origin (0, 0).
    M_proj_to_chip_raw = build_projector_to_chip_matrix(
        stage_xy_corr=(0.0, 0.0),
        projector_size=(pw, ph),
        camera_size=camera_size,
        projector_pixel_size=projector_pixel_size,
        camera_pixel_size=camera_pixel_size,
        proj_to_cam_matrix=proj_to_cam_matrix,
        corr_to_chip_matrix=corr_to_chip_matrix,
    )
    proj_center_chip = apply_homography_2d((0.5 * pw, 0.5 * ph), M_proj_to_chip_raw)
    T_shift_to_origin = np.array(
        [
            [1.0, 0.0, -float(proj_center_chip[0])],
            [0.0, 1.0, -float(proj_center_chip[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    M_proj_to_chip_0 = T_shift_to_origin @ M_proj_to_chip_raw

    # 3. Determine projector physical axes and FOV size in chip-space
    edge_midpoints_proj = np.array(
        [
            [0.0, 0.5 * ph],       # left midpoint
            [float(pw), 0.5 * ph], # right midpoint
            [0.5 * pw, 0.0],       # top midpoint
            [0.5 * pw, float(ph)], # bottom midpoint
        ],
        dtype=np.float64,
    )
    edge_midpoints_chip = apply_homography_2d(edge_midpoints_proj, M_proj_to_chip_0)
    vec_x = edge_midpoints_chip[1] - edge_midpoints_chip[0]
    vec_y = edge_midpoints_chip[3] - edge_midpoints_chip[2]

    fov_w_um = max(1e-3, float(np.linalg.norm(vec_x)))
    fov_h_um = max(1e-3, float(np.linalg.norm(vec_y)))
    u_x = vec_x / fov_w_um
    u_y = vec_y / fov_h_um

    # 4. Project the 4 corners of the layer's chip-space footprint onto (u_x, u_y)
    layer_corners_pat = np.array(
        [
            [-0.5 * w_um, -0.5 * h_um],
            [0.5 * w_um, -0.5 * h_um],
            [0.5 * w_um, 0.5 * h_um],
            [-0.5 * w_um, 0.5 * h_um],
        ],
        dtype=np.float64,
    )
    layer_corners_chip = apply_homography_2d(layer_corners_pat, M_pat_to_chip)
    proj_x_vals = layer_corners_chip @ u_x
    proj_y_vals = layer_corners_chip @ u_y
    span_x = float(np.max(proj_x_vals) - np.min(proj_x_vals))
    span_y = float(np.max(proj_y_vals) - np.min(proj_y_vals))

    stride_x = max(1.0, fov_w_um - float(overlap_x_um))
    stride_y = max(1.0, fov_h_um - float(overlap_y_um))

    # Check if the layer corners fit within a single centered projector frame
    fits_single_proj_x = False
    fits_single_proj_y = False
    try:
        M_chip_to_proj_0 = np.linalg.inv(M_proj_to_chip_0)
        corners_in_proj = apply_homography_2d(layer_corners_chip, M_chip_to_proj_0)
        if np.min(corners_in_proj[:, 0]) >= -1e-3 and np.max(corners_in_proj[:, 0]) <= float(pw) + 1e-3:
            fits_single_proj_x = True
        if np.min(corners_in_proj[:, 1]) >= -1e-3 and np.max(corners_in_proj[:, 1]) <= float(ph) + 1e-3:
            fits_single_proj_y = True
    except np.linalg.LinAlgError:
        pass

    if fits_single_proj_x or span_x <= fov_w_um + 1e-6:
        nx = 1
    else:
        nx = int(np.ceil((span_x - fov_w_um) / stride_x)) + 1

    if fits_single_proj_y or span_y <= fov_h_um + 1e-6:
        ny = 1
    else:
        ny = int(np.ceil((span_y - fov_h_um) / stride_y)) + 1

    snake_indices = generate_snake_sequence(nx, ny)
    tiles: List[np.ndarray] = []
    tile_coords_chip: List[Tuple[float, float]] = []

    interp = cv2.INTER_NEAREST if nearest_interp else cv2.INTER_LINEAR

    for x_idx, y_idx in snake_indices:
        dx_proj_um = (float(x_idx) - 0.5 * float(nx - 1)) * stride_x
        dy_proj_um = (float(y_idx) - 0.5 * float(ny - 1)) * stride_y

        rel_chip_vec = dx_proj_um * u_x + dy_proj_um * u_y
        rel_cx = float(rel_chip_vec[0])
        rel_cy = float(rel_chip_vec[1])

        T_tile_shift = np.array(
            [
                [1.0, 0.0, rel_cx],
                [0.0, 1.0, rel_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        M_proj_to_chip_tile = T_tile_shift @ M_proj_to_chip_0
        try:
            M_chip_to_proj_tile = np.linalg.inv(M_proj_to_chip_tile)
        except np.linalg.LinAlgError:
            M_chip_to_proj_tile = np.eye(3, dtype=np.float64)

        M_img_to_proj = M_chip_to_proj_tile @ M_img_to_chip

        warped = cv2.warpPerspective(
            img,
            M_img_to_proj,
            (pw, ph),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        tiles.append(warped)
        tile_coords_chip.append((rel_cx, rel_cy))

    return tiles, tile_coords_chip


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

    x_count = len(x_positions)
    y_count = len(y_positions)

    tile_grid: dict[Tuple[int, int], np.ndarray] = {}
    for y_idx, top in enumerate(y_positions):
        for x_idx, left in enumerate(x_positions):
            tile = img[top : top + tile_height, left : left + tile_width]
            th, tw = tile.shape[:2]
            if tw != tile_width or th != tile_height:
                if img.ndim == 3:
                    bg = np.zeros((tile_height, tile_width, img.shape[2]), dtype=img.dtype)
                else:
                    bg = np.zeros((tile_height, tile_width), dtype=img.dtype)
                bg[:th, :tw] = tile
                tile = bg
            tile_grid[(x_idx, y_idx)] = tile

    snake_coords = generate_snake_sequence(x_count, y_count)
    ordered_tiles: List[np.ndarray] = [tile_grid[c] for c in snake_coords]

    return ordered_tiles, snake_coords

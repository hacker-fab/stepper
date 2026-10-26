from collections.abc import Sequence
import cv2
import numpy as np
from typing import Optional, Tuple

from core.events import ColorMode


def get_aruco_dict(dict_id: int = cv2.aruco.DICT_4X4_250) -> cv2.aruco.Dictionary:
    """Returns the predefined OpenCV ArUco dictionary."""
    return cv2.aruco.getPredefinedDictionary(dict_id)


def generate_ar_tag_grid(
    grid_n: int,
    canvas_size: Tuple[int, int] = (1920, 1080),
    dict_id: int = cv2.aruco.DICT_4X4_250,
) -> np.ndarray:
    """Generates an N x N grid of ArUco tags centered on a canvas for projection.
    
    Args:
        grid_n: Number of tags along each axis (e.g. 2 for 2x2, 4 for 4x4, 8 for 8x8).
        canvas_size: (width, height) of projector canvas.
        dict_id: OpenCV ArUco dictionary identifier.
        
    Returns:
        BGR numpy array of shape (height, width, 3) with dtype uint8.
    """
    width, height = canvas_size
    grid_n = max(1, int(grid_n))
    dictionary = get_aruco_dict(dict_id)

    # 5% border margin around canvas to ensure all tags and quiet zones are well within projector bounds
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y

    cell_w = usable_w / grid_n
    cell_h = usable_h / grid_n
    marker_sz = max(12, int(min(cell_w, cell_h) * 0.65))
    quiet = max(4, int(marker_sz * 0.2))
    box_sz = marker_sz + 2 * quiet

    # Monochrome canvas (black background)
    canvas = np.zeros((height, width), dtype=np.uint8)

    max_markers = 250
    for r in range(grid_n):
        for c in range(grid_n):
            idx = (r * grid_n + c) % max_markers
            cx = int(margin_x + (c + 0.5) * cell_w)
            cy = int(margin_y + (r + 0.5) * cell_h)

            x1 = cx - box_sz // 2
            y1 = cy - box_sz // 2
            x2 = x1 + box_sz
            y2 = y1 + box_sz

            # Ensure inside canvas
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue

            # White quiet zone background
            canvas[y1:y2, x1:x2] = 255

            # Render ArUco marker
            marker_img = cv2.aruco.generateImageMarker(dictionary, idx, marker_sz)
            canvas[y1 + quiet : y1 + quiet + marker_sz, x1 + quiet : x1 + quiet + marker_sz] = marker_img

    # Build 3-channel BGR image
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = canvas
    bgr[:, :, 1] = canvas
    bgr[:, :, 2] = canvas
    return bgr


def detect_ar_tags(
    camera_image: Optional[np.ndarray],
    dict_id: int = cv2.aruco.DICT_4X4_250,
    flip_horizontal: bool = False,
) -> Tuple[int, Sequence, Optional[np.ndarray]]:
    """Detects ArUco tags in the given camera frame (BGR or Grayscale).
    
    Returns:
        (detected_count, corners, ids)
    """
    if camera_image is None or camera_image.size == 0:
        return 0, [], None

    dictionary = get_aruco_dict(dict_id)
    detector = cv2.aruco.ArucoDetector(dictionary)

    if camera_image.ndim == 2:
        gray = camera_image
    elif camera_image.shape[2] == 3:
        gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
    elif camera_image.shape[2] == 4:
        gray = cv2.cvtColor(camera_image, cv2.COLOR_BGRA2GRAY)
    else:
        return 0, [], None

    if flip_horizontal:
        gray = cv2.flip(gray, 1)

    corners, ids, rejected = detector.detectMarkers(gray)
    count = len(ids) if ids is not None else 0
    return count, corners, ids


def compute_focus_score(
    camera_image: Optional[np.ndarray],
    ddepth: int = cv2.CV_64F,
    kernel_size: int = 9,
) -> float:
    """Calculates focus sharpness score via Sobel gradient magnitude variance."""
    if camera_image is None or camera_image.size == 0:
        return 0.0

    img = camera_image.copy()
    if img.shape[0] * 0.5 >= kernel_size and img.shape[1] * 0.5 >= kernel_size:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        return 0.0

    if gray.shape[0] < kernel_size or gray.shape[1] < kernel_size:
        return 0.0

    sobel_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=kernel_size)
    magnitude = np.hypot(sobel_x, sobel_y)
    return float(magnitude.var())

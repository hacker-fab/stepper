import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from lib.edge_artag import (
    EdgeArucoDetector,
    EdgeDetectorParameters,
    detect_edge_markers,
)
from operations.stage_calibration import DEFAULT_ARUCO_DICT


def detect_markers_in_frame(
    frame: Optional[np.ndarray],
    parameters: Optional[EdgeDetectorParameters] = None,
    dictionary: Optional[cv2.aruco.Dictionary] = None,
    debug_output: Optional[Any] = None,
) -> Tuple[int, List[np.ndarray], Optional[np.ndarray]]:
    """Detects ArUco markers using edge_artag without fallback.

    Args:
        frame: BGR or grayscale numpy image.
        parameters: Optional EdgeDetectorParameters for fine-tuning.
        dictionary: Optional cv2.aruco.Dictionary (defaults to DEFAULT_ARUCO_DICT).
        debug_output: Optional dict or list to receive intermediate debug image outputs.

    Returns:
        (count, corners_list, ids_array)
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        if debug_output is not None:
            if isinstance(debug_output, dict):
                debug_output.clear()
            elif isinstance(debug_output, list):
                debug_output.clear()
        return 0, [], None

    dict_obj = dictionary if dictionary is not None else DEFAULT_ARUCO_DICT

    corners, ids, _ = detect_edge_markers(
        frame,
        dictionary=dict_obj,
        parameters=parameters,
        debug_output=debug_output,
    )
    count = len(ids) if ids is not None else 0
    return count, corners, ids


def detect_markers_batch(
    images_or_paths: Sequence[Union[str, np.ndarray]],
    parameters: Optional[EdgeDetectorParameters] = None,
    dictionary: Optional[cv2.aruco.Dictionary] = None,
    max_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Tuple[int, List[np.ndarray], Optional[np.ndarray], Tuple[int, int]]]:
    """Detects markers across multiple images in parallel using ThreadPoolExecutor.

    OpenCV image decoding and thresholding/contour operations release the Python GIL,
    enabling multi-core CPU scaling across threads.

    Args:
        images_or_paths: List of file paths or numpy image arrays.
        parameters: Optional EdgeDetectorParameters.
        dictionary: Optional cv2.aruco.Dictionary.
        max_workers: Max worker threads (defaults to min(32, (os.cpu_count() or 1) + 4)).
        progress_cb: Optional callback receiving (completed_count, total_count).

    Returns:
        List of (count, corners, ids, (width, height)) in the same order as input.
    """
    total = len(images_or_paths)
    if total == 0:
        return []

    workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    results: List[Optional[Tuple[int, List[np.ndarray], Optional[np.ndarray], Tuple[int, int]]]] = [None] * total

    def _process_item(item: Union[str, np.ndarray]) -> Tuple[int, List[np.ndarray], Optional[np.ndarray], Tuple[int, int]]:
        if isinstance(item, str):
            img = cv2.imread(item)
        else:
            img = item
        if img is None or getattr(img, "size", 0) == 0:
            return 0, [], None, (0, 0)
        h, w = img.shape[:2]
        cnt, corners, ids = detect_markers_in_frame(img, parameters=parameters, dictionary=dictionary)
        return cnt, corners, ids, (w, h)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_item, item): idx for idx, item in enumerate(images_or_paths)}
        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[detect_markers_batch] Error processing item {idx}: {e}")
                results[idx] = (0, [], None, (0, 0))
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, total)

    return [r if r is not None else (0, [], None, (0, 0)) for r in results]


def draw_detected_markers(
    image: np.ndarray,
    corners: List[np.ndarray],
    ids: Optional[np.ndarray],
) -> np.ndarray:
    """Draws detected ArUco marker boundaries, corner origins, and IDs on an image copy."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    if ids is None or len(corners) == 0:
        return annotated

    for idx, c in enumerate(corners):
        pts = c.reshape(-1, 2).astype(np.int32)
        # Bounding polygon
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Top-left corner highlight (origin of the tag)
        cv2.circle(annotated, tuple(pts[0]), radius=5, color=(0, 0, 255), thickness=-1)
        # ID label
        mid = int(ids.flatten()[idx])
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.putText(
            annotated,
            f"ID:{mid}",
            (center[0] - 20, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


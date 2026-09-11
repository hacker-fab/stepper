from typing import Callable
from dataclasses import dataclass
from core.operation import Operation, ExecutionContext
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class AlignmentConfig:
    enabled: bool = True
    model_path: str = "ckpts/best.pt"
    right_marker_x: float = 1820.0
    left_marker_x: float = 280.0
    top_marker_y: float = 269.0
    bottom_marker_y: float = 1075.0
    x_scale_factor: float = -1100.0
    y_scale_factor: float = 800.0




def load_alignment_model(model_path: str) -> Optional[Any]:
    """Attempts to load the YOLO alignment model weights.

    Returns the loaded model or None if ultralytics is not installed or loading
    fails.
    """
    try:
        print("Loading YOLO alignment model...")
        from ultralytics import YOLO

        model = YOLO(model_path)
        print("Loaded YOLO alignment model successfully.")
        return model
    except ImportError:
        print(
            "[Alignment] Optional package 'ultralytics' is not installed. "
            "Alignment marker detection is disabled. Install with: pip install '.[alignment]'"
        )
        return None
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None


def detect_alignment_markers(
    model: Any, image: np.ndarray, draw_rectangle: bool = False
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], np.ndarray]:
    """Detects alignment markers across the entire image using a YOLO model."""
    if model is None or image is None:
        return [], (image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8))

    detections = []
    display_image = image.copy()
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (640, 640))
        results = model(resized)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * original_width / 640)
            x2 = int(x2 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            y2 = int(y2 * original_height / 640)
            detections.append(((x1, y1), (x2, y2)))
            print("mark at ", (x1 + x2) / 2, (y1 + y2) / 2)
            if draw_rectangle:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    except Exception as e:
        print(f"Detection failed: {e}")

    return detections, display_image


def detect_alignment_markers_tiling(
    yolo_model: Any,
    image: np.ndarray,
    draw_rectangle: bool = False,
    edge: Optional[Union[str, List[str]]] = None,
    edge_fraction: float = 0.25,
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], np.ndarray]:
    """Detects alignment markers and optionally filters detections by image edge(s).

    edge: 'left', 'right', 'top', or a list like ['left', 'right']. None means markers are expected on all edges.
    """
    if yolo_model is None or image is None:
        return [], (image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8))

    detections = []
    display_image = image.copy()
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (640, 640))
        results = yolo_model(resized)
        boxes = results[0].boxes

        if isinstance(edge, str):
            edge = [edge]  # allow single string or list

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * original_width / 640)
            x2 = int(x2 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            y2 = int(y2 * original_height / 640)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # If edge filtering is enabled
            if edge is not None:
                if "left" in edge and x_center > original_width * edge_fraction:
                    continue
                if "right" in edge and x_center < original_width * (1 - edge_fraction):
                    continue
                if "top" in edge and y_center > original_height * edge_fraction:
                    continue

            detections.append(((x1, y1), (x2, y2)))
            if draw_rectangle:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        print(f"Detected {len(detections)} marker(s)")
    except Exception as e:
        print(f"Detection failed: {e}")

    return detections, display_image


def compute_standard_alignment_offset(
    markers: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    image_width: int,
    image_height: int,
    alignment: AlignmentConfig,
) -> Tuple[float, float]:
    """Computes dx, dy offset using all detected markers against alignment configuration."""
    if not markers or image_width == 0 or image_height == 0:
        return 0.0, 0.0

    dx, dy = 0.0, 0.0
    for xy0, xy1 in markers:
        x0, y0 = xy0
        x1, y1 = xy1
        x = (x0 + x1) / 2.0 / image_width
        y = (y0 + y1) / 2.0 / image_height

        if x > 0.5:
            dx += alignment.x_scale_factor * (alignment.right_marker_x / image_width - x)
        else:
            dx += alignment.x_scale_factor * (alignment.left_marker_x / image_width - x)

        if y > 0.5:
            dy += alignment.y_scale_factor * (alignment.bottom_marker_y / image_height - y)
        else:
            dy += alignment.y_scale_factor * (alignment.top_marker_y / image_height - y)

    dx /= len(markers)
    dy /= len(markers)
    return float(dx), float(dy)


def compute_tiling_alignment_offset(
    markers: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    image_width: int,
    image_height: int,
    alignment: AlignmentConfig,
) -> Tuple[float, float]:
    """Computes dx, dy offset for tiling alignment with edge-specific weighting."""
    if not markers or image_width == 0 or image_height == 0:
        return 0.0, 0.0

    dx, dy = 0.0, 0.0
    count_x, count_y = 0, 0

    for xy0, xy1 in markers:
        x0, y0 = xy0
        x1, y1 = xy1
        x = (x0 + x1) / 2.0 / image_width
        y = (y0 + y1) / 2.0 / image_height

        # Horizontal alignment (left/right markers)
        if x < 0.5:
            dx += alignment.x_scale_factor * (alignment.left_marker_x / image_width - x)
            count_x += 1
        elif x > 0.5:
            dx += alignment.x_scale_factor * (alignment.right_marker_x / image_width - x)
            count_x += 1

        # Vertical alignment (top markers only)
        if y < 0.3:
            dy += alignment.y_scale_factor * (alignment.top_marker_y / image_height - y)
            count_y += 1

    if count_x > 0:
        dx /= count_x
    if count_y > 0:
        dy /= count_y

    return float(dx), float(dy)


class AlignmentOperation(Operation):
    """Detects alignment marks with YOLO and offsets the stage."""

    def __init__(self, config: Optional[AlignmentConfig] = None):
        super().__init__("Optical Alignment")
        self.config = config or AlignmentConfig()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]):
        report_progress(0.1, "Detecting alignment markers...")
        if not self.config.enabled:
            report_progress(1.0, "Alignment disabled in config")
            return

        model = load_alignment_model(self.config.model_path)
        if model is None:
            report_progress(1.0, "YOLO alignment model could not be loaded")
            return

        cam_img = context.camera.get_latest_frame()
        if cam_img is None:
            report_progress(1.0, "No camera frame available for alignment.")
            return

        h, w = cam_img.shape[:2]
        markers, _ = detect_alignment_markers(model, cam_img)
        if not markers:
            report_progress(1.0, "No markers detected")
            return

        report_progress(0.5, "Calculating offset...")
        dx, dy = compute_standard_alignment_offset(markers, w, h, self.config)
        context.stage.move_relative({"x": dx, "y": dy})
        report_progress(1.0, f"Aligned: dx={dx:.2f}µm, dy={dy:.2f}µm")

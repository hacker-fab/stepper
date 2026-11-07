# Hacker Fab
# Stage Calibration Operations:
# Stage 1: Multi-Point Autofocus Sweep across Wafer Surface (Z Focal Plane Mapping)
# Stage 2: Optical Alignment, Distortion Correction, and ArUco Tag Grid Patterning (using Stage 1 Z Surface)
# Stage 3: Chip Pose Estimation, Multi-Tag Camera Scanning, Relative Least Squares Global Reconstruction

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pathlib

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import detect_ar_tags
from lib.edge_artag import (
    EdgeArucoDetector,
    EdgeDetectorParameters,
    detect_edge_markers,
)
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_points_projector_to_camera,
)
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from stage_control.calibration import StageCalibration, StageCalibrationPoint


ARUCO_DICT_CACHE_FILE = pathlib.Path(__file__).parent.parent.parent / 'aruco_dict_6x6_10000.yml'


def _get_cache_path(filename: str = ARUCO_DICT_CACHE_FILE) -> str:
    if os.path.exists(filename):
        return filename
    module_dir_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(module_dir_file):
        return module_dir_file
    return filename


def load_aruco_dict_from_file(file_path: str = ARUCO_DICT_CACHE_FILE) -> Optional[cv2.aruco.Dictionary]:
    """Loads an ArUco dictionary from a FileStorage YAML/XML/JSON file."""
    actual_path = _get_cache_path(file_path)
    if not os.path.exists(actual_path):
        return None
    fs = None
    try:
        read_flag = getattr(cv2, "FILE_STORAGE_READ", getattr(cv2, "FileStorage_READ", 0))
        fs = cv2.FileStorage(actual_path, read_flag)
        if not fs.isOpened():
            return None
        root = fs.root()
        if hasattr(root, "empty") and root.empty():
            return None
        dictionary = cv2.aruco.Dictionary()
        if hasattr(dictionary, "readDictionary"):
            ok = dictionary.readDictionary(root)
            if ok is False:
                return None
            return dictionary
    except Exception as e:
        print(f"[Stage Calibration] Could not load ArUco dictionary from {actual_path}: {e}", flush=True)
        return None
    finally:
        if fs is not None and hasattr(fs, "release"):
            fs.release()
    return None


def save_aruco_dict_to_file(dictionary: cv2.aruco.Dictionary, file_path: str = ARUCO_DICT_CACHE_FILE) -> bool:
    """Saves an ArUco dictionary to a local FileStorage file using writeDictionary."""
    fs = None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        write_flag = getattr(cv2, "FILE_STORAGE_WRITE", getattr(cv2, "FileStorage_WRITE", 1))
        fs = cv2.FileStorage(file_path, write_flag)
        if not fs.isOpened():
            return False
        if hasattr(dictionary, "writeDictionary"):
            dictionary.writeDictionary(fs)
        return True
    except Exception as e:
        print(f"[Stage Calibration] Could not save ArUco dictionary to {file_path}: {e}", flush=True)
        return False
    finally:
        if fs is not None and hasattr(fs, "release"):
            fs.release()


def get_aruco_dict(dictionary: Any = None, cache_file: str = ARUCO_DICT_CACHE_FILE) -> cv2.aruco.Dictionary:
    """Returns a custom ArUco dictionary generated with 5x5 markers and 10000 IDs,
    loading from a local cached FileStorage file if available."""
    dict_type = getattr(cv2.aruco, "Dictionary", None)
    if isinstance(dict_type, type) and isinstance(dictionary, dict_type):
        return dictionary

    # 1. Try loading from cached local file first
    loaded = load_aruco_dict_from_file(cache_file)
    if loaded is not None:
        return loaded

    # 2. Generate custom dictionary (5x5, 10000 IDs)
    if hasattr(cv2.aruco, "Dictionary") and hasattr(cv2.aruco.Dictionary, "create"):
        dictionary = cv2.aruco.Dictionary.create(10000, 6, 12345)
    elif hasattr(cv2.aruco, "extendDictionary"):
        dictionary = cv2.aruco.extendDictionary(10000, 6, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000), 12345)
    elif hasattr(cv2.aruco, "Dictionary_create"):
        dictionary = cv2.aruco.Dictionary_create(10000, 6, 12345)
    else:
        dictionary = cv2.aruco.Dictionary.create(10000, 6, 12345)

    # 3. Save to local file using cv2.FileStorage and writeDictionary
    save_aruco_dict_to_file(dictionary, cache_file)

    return dictionary


DEFAULT_ARUCO_DICT = get_aruco_dict()
# DEFAULT_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)


def compute_calibration_grid_params(
    cal_range: float,
    pixel_scale: float,
    camera_size: Tuple[int, int] = (1920, 1080),
) -> Tuple[int, float]:
    """Computes square grid_size N and step_distance d.

    Ensures step_distance <= 0.4 * min(FOV_x, FOV_y) so that any 2x2 cluster
    of 4 adjacent tags fits within a single camera frame.

    Returns:
        (grid_size, step_distance) where grid_size is N (for N x N grid).
    """
    cam_w, cam_h = camera_size
    fov = min(cam_w, cam_h) * float(pixel_scale)
    target_pitch = max(100.0, 0.4 * fov)

    intervals = max(1, int(round(cal_range / target_pitch)))
    grid_size = intervals + 1
    step_distance = float(cal_range / intervals)
    return grid_size, step_distance


def generate_spiral_grid_indices(nx: int, ny: int) -> List[Tuple[int, int]]:
    """Generates (row, col) grid indices in center-to-edge spiral order.

    Guarantees all (r, c) in [0, ny) x [0, nx) are visited exactly once,
    starting from the center grid cell ((ny - 1) // 2, (nx - 1) // 2).
    """
    total = nx * ny
    if total <= 0:
        return []

    c0 = (nx - 1) // 2
    r0 = (ny - 1) // 2

    indices: List[Tuple[int, int]] = [(r0, c0)]
    visited = {(r0, c0)}

    x, y = 0, 0
    dx, dy = 1, 0
    segment_length = 1
    segment_passed = 0
    turns = 0

    max_steps = max(nx, ny) * max(nx, ny) * 4 + 100
    steps = 0
    while len(indices) < total and steps < max_steps:
        x += dx
        y += dy
        steps += 1

        c = c0 + x
        r = r0 + y
        if 0 <= r < ny and 0 <= c < nx:
            if (r, c) not in visited:
                visited.add((r, c))
                indices.append((r, c))

        segment_passed += 1
        if segment_passed == segment_length:
            segment_passed = 0
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                segment_length += 1

    return indices


def generate_canonical_ar_tag(
    marker_id: int,
    dictionary: cv2.aruco.Dictionary = DEFAULT_ARUCO_DICT,
    size: int = 400,
    quiet_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a square marker image with white quiet zone and returns canonical corners.

    Corners order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    quiet = max(4, int(size * quiet_ratio))
    total_sz = size + 2 * quiet

    canvas = np.zeros((total_sz, total_sz), dtype=np.uint8)
    canvas[:, :] = 255  # white quiet zone background
    marker_img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), size)
    canvas[quiet : quiet + size, quiet : quiet + size] = marker_img

    corners = np.array(
        [
            [0.0, 0.0],
            [float(total_sz), 0.0],
            [float(total_sz), float(total_sz)],
            [0.0, float(total_sz)],
        ],
        dtype=np.float32,
    )
    return canvas, corners


def warp_marker_to_projector(
    marker_id: int,
    target_proj_corners: np.ndarray,
    canvas_size: Tuple[int, int],
    dictionary: cv2.aruco.Dictionary = DEFAULT_ARUCO_DICT,
) -> np.ndarray:
    """Warps an ArUco marker onto the projector canvas to match target_proj_corners."""
    pw, ph = canvas_size
    canonical_img, canonical_corners = generate_canonical_ar_tag(marker_id, dictionary=dictionary, size=300)

    M = cv2.getPerspectiveTransform(canonical_corners, target_proj_corners.astype(np.float32))
    warped_gray = cv2.warpPerspective(canonical_img, M, (pw, ph), flags=cv2.INTER_LINEAR)

    bgr = np.zeros((ph, pw, 3), dtype=np.uint8)
    bgr[:, :, 0] = warped_gray
    bgr[:, :, 1] = warped_gray
    bgr[:, :, 2] = warped_gray
    return bgr


def solve_global_positions_from_relative_measurements(
    grid_size: int,
    measurements: List[Tuple[int, int, float, float]],
    anchor_id: int,
    anchor_pos: Tuple[float, float],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Reconstructs global optical positions using least squares from pairwise relative vectors.

    Args:
        grid_size: N for an N x N grid. Total markers M = N * N.
        measurements: List of (id_A, id_B, dx, dy) where x_B - x_A = dx and y_B - y_A = dy.
        anchor_id: Marker ID to anchor (e.g. 0).
        anchor_pos: (X, Y) coordinate for the anchor.
        nominal_positions: Dict mapping marker_id -> nominal (X, Y) fallback.

    Returns:
        Dict mapping marker_id -> reconstructed (X, Y).
    """
    total_markers = grid_size * grid_size
    if len(measurements) == 0:
        return dict(nominal_positions)

    K = len(measurements)
    A = np.zeros((K + 1, total_markers), dtype=np.float64)
    bx = np.zeros(K + 1, dtype=np.float64)
    by = np.zeros(K + 1, dtype=np.float64)

    for k, (id_a, id_b, dx, dy) in enumerate(measurements):
        if 0 <= id_a < total_markers and 0 <= id_b < total_markers:
            A[k, id_a] = -1.0
            A[k, id_b] = 1.0
            bx[k] = dx
            by[k] = dy

    # Anchor row
    A[K, anchor_id] = 1.0
    bx[K] = anchor_pos[0]
    by[K] = anchor_pos[1]

    sol_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Check for connected markers
    observed = set()
    for id_a, id_b, _, _ in measurements:
        observed.add(id_a)
        observed.add(id_b)
    observed.add(anchor_id)

    results: Dict[int, Tuple[float, float]] = {}
    for m_id in range(total_markers):
        if m_id in observed and not (np.isnan(sol_x[m_id]) or np.isnan(sol_y[m_id])):
            results[m_id] = (float(sol_x[m_id]), float(sol_y[m_id]))
        else:
            results[m_id] = nominal_positions.get(m_id, (0.0, 0.0))

    return results


# =============================================================================
# STAGE 1: Z FOCUS CALIBRATION OPERATION
# =============================================================================


class StageCalibrationZFocusOperation(Operation):
    """Stage 1: Multi-point autofocus sweep across the wafer surface to map the Z focal plane."""

    def __init__(
        self,
        calibration_range: float = 10000.0,
        camera_pixel_size: Optional[float] = None,
        stabilization_delay: float = 1.0,
        output_json_path: Optional[str] = None,
        apply_to_stage: bool = True,
        grid_size: Optional[int] = None,
        step_distance: Optional[float] = None,
        dictionary: cv2.aruco.Dictionary = DEFAULT_ARUCO_DICT,
    ):
        super().__init__("Stage Calibration - Stage 1 (Z Focus)")
        self.calibration_range = float(calibration_range)
        self.camera_pixel_size = float(camera_pixel_size) if camera_pixel_size is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.output_json_path = output_json_path
        self.apply_to_stage = bool(apply_to_stage)
        self.grid_size = int(grid_size) if grid_size is not None else None
        self.step_distance = float(step_distance) if step_distance is not None else None

        self._current_sub_op: Optional[Operation] = None
        self.calibration: Optional[StageCalibration] = None
        self.recorded_points: List[StageCalibrationPoint] = []

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        report_progress: Callable[[float, str], None],
    ) -> Optional[str]:
        self._current_sub_op = op
        err = op.execute(context, report_progress)
        self._current_sub_op = None
        return err

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None or context.camera is None:
            return "Stage or Camera not available in execution context"

        stage = context.stage
        camera = context.camera

        stage.set_calibration(None)

        pixel_scale = (
            self.camera_pixel_size
            if self.camera_pixel_size is not None
            else (
                context.machine_config.stage_calibration.camera_pixel_size
                if context.machine_config is not None
                else 1.0
            )
        )
        output_path = (
            self.output_json_path
            or (
                context.machine_config.stage_calibration.calibration_file_path
                if context.machine_config is not None
                else "stage_calibration.json"
            )
        )

        # Get camera frame dimensions
        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)

        # Calculate grid parameters: N and d
        if self.grid_size is not None and self.step_distance is not None:
            nx = ny = self.grid_size
            step_dist = self.step_distance
        else:
            nx, step_dist = compute_calibration_grid_params(
                cal_range=self.calibration_range,
                pixel_scale=pixel_scale,
                camera_size=(cam_w, cam_h),
            )
            ny = nx

        total_points = nx * ny
        start_x, start_y, start_z = stage.get_raw_position()
        c0 = (nx - 1) // 2
        r0 = (ny - 1) // 2
        spiral_indices = generate_spiral_grid_indices(nx, ny)

        report_progress(0.0, f"Starting Stage 1: Z focus survey on {nx}x{ny} grid (step {step_dist:.1f} µm)...")
        self.recorded_points = []

        for point_idx, (r, c) in enumerate(spiral_indices):
            if self.is_aborted:
                return "Stage 1 calibration aborted"

            prog_start = point_idx / total_points
            prog_end = (point_idx + 1) / total_points
            marker_id = r * nx + c

            target_raw_x = start_x + (c - c0) * step_dist
            target_raw_y = start_y + (r - r0) * step_dist

            report_progress(
                prog_start,
                f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}]): Moving to ({target_raw_x:.1f}, {target_raw_y:.1f}) µm...",
            )
            stage.move_absolute_raw({"x": target_raw_x, "y": target_raw_y}, set_internal_coord=True)
            context.delay_func(self.stabilization_delay)

            # Autofocus at current point (Red focal plane measurement, UV offset=0.0)
            af_op = AutofocusOperation(uv_z_offset=0.0)
            af_reporter = sub_progress(
                report_progress,
                prog_start + 0.1 * (prog_end - prog_start),
                prog_end,
                prefix=f"Point {point_idx + 1}/{total_points} - AF",
            )
            af_err = self._run_sub_op(af_op, context, af_reporter)
            if af_err and self.is_aborted:
                return "Stage 1 calibration aborted during autofocus"

            if af_op.best_red_z is not None:
                z_focus = float(af_op.best_red_z)
            else:
                z_focus = float(stage.get_raw_position()[2])

            pt = StageCalibrationPoint(
                grid_index=(r, c),
                marker_id=marker_id,
                raw_xy=(float(target_raw_x), float(target_raw_y)),
                corr_xy=(float(target_raw_x), float(target_raw_y)),  # Identity mapping until Stage 3
                z_focus=float(z_focus),
            )
            self.recorded_points.append(pt)

        # Sort points by grid_index for deterministic ordering
        self.recorded_points.sort(key=lambda p: (p.grid_index[0], p.grid_index[1]))

        self.calibration = StageCalibration(
            grid_shape=(ny, nx),
            points=self.recorded_points,
            stage_type=type(stage).__name__,
            step_distance=step_dist,
            calibration_range=self.calibration_range,
        )
        self.calibration.save_to_json(output_path)

        if self.apply_to_stage:
            stage.set_calibration(self.calibration)
            print("[Stage Calibration] Applied Z-leveling calibration map to stage controller.")

        report_progress(
            1.0,
            f"Stage 1 complete! Saved Z focal plane with {len(self.recorded_points)} points to {output_path}. "
            "Stage Z-leveling is now active.",
        )
        return None


# =============================================================================
# STAGE 2: PATTERNING OPERATION
# =============================================================================


class StageCalibrationPatternOperation(Operation):
    """Stage 2: Measure optical transform, correct alignment, and pattern the ArUco tag grid using calibrated Z heights."""

    def __init__(
        self,
        input_json_path: Optional[str] = None,
        exposure_time: Optional[float] = None,
        stabilization_delay: float = 1.0,
        autofocus: bool = False,
        grid_size: Optional[int] = None,
        pattern_range: Optional[float] = None,
    ):
        super().__init__("Stage Calibration - Stage 2 (Patterning)")
        self.input_json_path = input_json_path
        self.exposure_time = float(exposure_time) if exposure_time is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.autofocus = bool(autofocus)
        self.grid_size = int(grid_size) if grid_size is not None and int(grid_size) > 0 else None
        self.pattern_range = float(pattern_range) if pattern_range is not None and float(pattern_range) > 0 else None

        self._current_sub_op: Optional[Operation] = None
        self.corrected_proj_corners: Optional[np.ndarray] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        report_progress: Callable[[float, str], None],
    ) -> Optional[str]:
        self._current_sub_op = op
        err = op.execute(context, report_progress)
        self._current_sub_op = None
        return err

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None or context.projector is None or context.camera is None:
            return "Stage, Projector, or Camera not available in execution context"

        input_path = (
            self.input_json_path
            or (
                context.machine_config.stage_calibration.calibration_file_path
                if context.machine_config is not None
                else "stage_calibration.json"
            )
        )

        if not os.path.exists(input_path):
            return f"Stage 1 calibration file not found: {input_path}. Please run Stage 1 (Z Focus) first."

        cal = StageCalibration.load_from_json(input_path)
        if len(cal.points) == 0:
            return f"No points found in calibration file: {input_path}"

        stage = context.stage
        stage.set_calibration(cal)
        projector = context.projector
        camera = context.camera
        pw, ph = projector.projector_size()

        effective_exp_time = (
            self.exposure_time
            if self.exposure_time is not None
            else (
                context.machine_config.exposure.default_exposure_time_ms / 1000.0
                if context.machine_config is not None
                else 8.0
            )
        )

        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)
        u_c, v_c = cam_w / 2.0, cam_h / 2.0

        ny, nx = cal.grid_shape
        step_dist = cal.step_distance or 1000.0
        pts_by_grid: Dict[Tuple[int, int], StageCalibrationPoint] = {
            tuple(p.grid_index): p for p in cal.points
        }

        try:
            # =========================================================================
            # STEP 1: MEASURE OPTICAL TRANSFORM (PROJECTOR TO CAMERA)
            # =========================================================================

            # Initial autofocus is ALWAYS performed before patterning
            report_progress(0.02, f"Performing initial autofocus...")
            af_op = AutofocusOperation(uv_z_offset=0)
            af_err = self._run_sub_op(af_op, context, sub_progress(report_progress, 0.02, 0.05, prefix="Initial AF"))
            if af_err or self.is_aborted:
                return af_err or "Stage 2 calibration aborted during initial autofocus"

            af_start_z = stage.get_raw_position()[2]


            report_progress(0.05, "Step 1/2: Measuring optical transform (Projector -> Camera)...")
            grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
                grid_n=2,
                canvas_size=(pw, ph),
                dict_id=DEFAULT_ARUCO_DICT,
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(ColorMode.RED)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)
            context.delay_func(self.stabilization_delay)

            cal_frame = camera.get_latest_frame()
            det_count, corners, ids = detect_ar_tags(cal_frame, dict_id=DEFAULT_ARUCO_DICT)

            H_proj_to_cam, matched_count = compute_projector_to_camera_homography(corners, ids, tag_coords)
            projector.set_on(False)

            if H_proj_to_cam is None or matched_count < 4:
                return (
                    f"Projector-to-camera optical calibration failed: insufficient tag matches ({matched_count}/4 points). "
                    "Cannot pattern stage grid without calibrated projection."
                )

            try:
                H_cam_to_proj = np.linalg.inv(H_proj_to_cam)
            except np.linalg.LinAlgError as e:
                return f"Failed to invert projector-to-camera homography matrix: {e}"

            # 3-step iterative correction so marker 0 sits with top-left at camera center
            report_progress(0.08, "Iteratively correcting tag projection alignment (3 passes)...")
            tag_sz_cam = min(cam_w, cam_h) * 0.25
            cam_corners = np.array([
                [u_c, v_c],
                [u_c + tag_sz_cam, v_c],
                [u_c + tag_sz_cam, v_c + tag_sz_cam],
                [u_c, v_c + tag_sz_cam],
            ], dtype=np.float64)

            for _ in range(3):
                if self.is_aborted:
                    return "Stage 2 calibration aborted"

                proj_pts = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)
                self.corrected_proj_corners = proj_pts

                test_img = warp_marker_to_projector(
                    marker_id=0,
                    target_proj_corners=proj_pts,
                    canvas_size=(pw, ph),
                    dictionary=DEFAULT_ARUCO_DICT,
                )
                projector.set_generated_image(test_img)
                projector.set_color_mode(ColorMode.RED)
                projector.set_on(True)
                context.delay_func(self.stabilization_delay)

                iter_frame = camera.get_latest_frame()
                c_cnt, c_corners, _ = detect_ar_tags(iter_frame, dict_id=DEFAULT_ARUCO_DICT)
                if c_cnt > 0 and c_corners:
                    detected_tl = c_corners[0].reshape(-1, 2)[0]
                    err = detected_tl - np.array([u_c, v_c])
                    cam_corners[:, 0] -= err[0]
                    cam_corners[:, 1] -= err[1]

                projector.set_on(False)

            self.corrected_proj_corners = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)

            # =========================================================================
            # STEP 2: EXPOSE ARUCO GRID USING CALIBRATED Z FOCUS SURFACE
            # =========================================================================
            report_progress(0.15, f"Step 2/2: Patterning grid using Stage 1 calibrated Z heights...")
            loop_span = 0.85
            base_offset = 0.15

            af_op = AutofocusOperation()
            af_err = self._run_sub_op(af_op, context, sub_progress(report_progress, 0.02, 0.05, prefix="Initial AF"))
            if af_err or self.is_aborted:
                return af_err or "Stage 2 calibration aborted during initial autofocus"


            # Find closest calibration grid node to current stage XY
            cur_raw_x, cur_raw_y = stage.get_raw_position()[:2]
            closest_pt = min(
                cal.points,
                key=lambda p: (p.raw_xy[0] - cur_raw_x) ** 2 + (p.raw_xy[1] - cur_raw_y) ** 2,
            )
            r_center, c_center = closest_pt.grid_index

            # Determine grid size if pattern_range is specified
            eff_grid_size = self.grid_size
            if eff_grid_size is None and self.pattern_range is not None and self.pattern_range > 0:
                eff_grid_size = max(1, int(round(self.pattern_range / step_dist)) + 1)

            if eff_grid_size is not None and eff_grid_size < max(nx, ny):
                n_sub = eff_grid_size
                r_min = max(0, min(ny - n_sub, r_center - n_sub // 2))
                r_max = r_min + n_sub - 1
                c_min = max(0, min(nx - n_sub, c_center - n_sub // 2))
                c_max = c_min + n_sub - 1
                selected_set = {
                    (r, c)
                    for r in range(r_min, r_max + 1)
                    for c in range(c_min, c_max + 1)
                    if (r, c) in pts_by_grid
                }
            else:
                selected_set = set(pts_by_grid.keys())

            spiral_indices = generate_spiral_grid_indices(nx, ny)
            pattern_indices = [(r, c) for (r, c) in spiral_indices if (r, c) in selected_set]
            total_points = len(pattern_indices)

            for point_idx, (r, c) in enumerate(pattern_indices):
                if self.is_aborted:
                    return "Stage 2 calibration aborted"

                prog_frac = base_offset + (point_idx / total_points) * loop_span
                pt = pts_by_grid[(r, c)]
                marker_id = pt.marker_id
                target_raw_x = float(pt.raw_xy[0])
                target_raw_y = float(pt.raw_xy[1])

                report_progress(
                    prog_frac,
                    f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id}): Moving to ({target_raw_x:.1f}, {target_raw_y:.1f})...",
                )
                stage.move_absolute(
                    {"x": target_raw_x, "y": target_raw_y},
                    z_calib_only=True,
                )
                context.delay_func(self.stabilization_delay)

                pt_span = loop_span / total_points
                # If autofocus checkbox is enabled, perform autofocus at every node
                if self.autofocus:
                    report_progress(
                        prog_frac + 0.02 * pt_span,
                        f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id}): Autofocusing...",
                    )
                    node_af_op = AutofocusOperation()
                    af_reporter = sub_progress(
                        report_progress,
                        prog_frac + 0.02 * pt_span,
                        prog_frac + 0.3 * pt_span,
                        prefix=f"AF Point {point_idx + 1}/{total_points}",
                    )
                    af_err = self._run_sub_op(node_af_op, context, af_reporter)
                    if af_err or self.is_aborted:
                        return af_err or "Stage 2 calibration aborted during per-node autofocus"

                # Expose single ArUco marker directly
                tag_img = warp_marker_to_projector(
                    marker_id=marker_id,
                    target_proj_corners=self.corrected_proj_corners,
                    canvas_size=(pw, ph),
                    dictionary=DEFAULT_ARUCO_DICT,
                )
                projector.set_generated_image(tag_img)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                exp_op = ExposureOperation(layer_index=None, exposure_time=effective_exp_time * 1000.0)
                exp_reporter = sub_progress(
                    report_progress,
                    prog_frac + (0.35 if self.autofocus else 0.1) * pt_span,
                    prog_frac + pt_span,
                    prefix=f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id})",
                )
                exp_err = self._run_sub_op(exp_op, context, exp_reporter)
                if exp_err or self.is_aborted:
                    return exp_err or "Stage 2 calibration aborted during exposure"

            report_progress(
                1.0,
                f"Stage 2 complete! {total_points} ArUco markers patterned using calibrated Z heights. "
                "Please develop wafer, mount, and jog stage so at least one tag is visible to run Stage 3.",
            )
            # Move the stage 2mm away from the end position in the direction away from the starting position
            if total_points > 0:
                first_pt = pts_by_grid[pattern_indices[0]]
                last_pt = pts_by_grid[pattern_indices[-1]]
                dx = float(last_pt.raw_xy[0] - first_pt.raw_xy[0])
                dy = float(last_pt.raw_xy[1] - first_pt.raw_xy[1])
                sign_x = 1.0 if dx >= 0 else -1.0
                sign_y = 1.0 if dy >= 0 else -1.0
                stage.move_relative({"x": sign_x * 2000.0, "y": sign_y * 2000.0}, z_calib_only=True)

            return None

        finally:
            cal.save_to_json(input_path)

            projector.set_on(False)
            projector.set_generated_image(None)
            projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)


# =============================================================================
# STAGE 3: SPIRAL IMAGE SCAN OPERATION (HARDWARE ACQUISITION ONLY)
# =============================================================================


class StageCalibrationScanOperation(Operation):
    """Stage 3: Outward spiral image scan with Z-offset tracking, saving all photos to a folder."""

    def __init__(
        self,
        output_dir: str = "",
        scan_range: Optional[float] = None,
        camera_pixel_size: Optional[float] = None,
        stabilization_delay: float = 0.5,
        autofocus: bool = False,
        grid_size: Optional[int] = None,
        input_json_path: Optional[str] = None,
        # Legacy parameter compatibility if called from older signatures
        output_json_path: Optional[str] = None,
        apply_to_stage: bool = False,
        save_images_dir: Optional[str] = None,
        edge_detector_params: Optional[Any] = None,
    ):
        super().__init__("Stage Calibration - Stage 3 (Spiral Image Scan)")
        resolved_dir = save_images_dir or output_dir
        self.output_dir = str(resolved_dir).strip() if resolved_dir else ""
        self.scan_range = float(scan_range) if scan_range is not None and float(scan_range) > 0 else None
        self.camera_pixel_size = float(camera_pixel_size) if camera_pixel_size is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.autofocus = bool(autofocus)
        self.grid_size = int(grid_size) if grid_size is not None and int(grid_size) > 0 else None
        self.input_json_path = input_json_path
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        report_progress: Callable[[float, str], None],
    ) -> Optional[str]:
        self._current_sub_op = op
        err = op.execute(context, report_progress)
        self._current_sub_op = None
        return err

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None or context.camera is None:
            return "Stage or Camera not available in execution context"

        if not self.output_dir:
            return "Output directory is required for saving scanned images"

        os.makedirs(self.output_dir, exist_ok=True)

        stage = context.stage
        camera = context.camera

        # Load Stage 1 calibration onto stage if provided so z_calib_only uses the measured surface
        if self.input_json_path and os.path.exists(self.input_json_path):
            try:
                cal = StageCalibration.load_from_json(self.input_json_path)
                stage.set_calibration(cal)
            except Exception as e:
                print(f"[StageCalibrationScan] Warning: could not load Stage 1 calibration: {e}")

        effective_pixel_scale = (
            self.camera_pixel_size
            if self.camera_pixel_size is not None
            else (
                context.machine_config.stage_calibration.camera_pixel_size
                if context.machine_config is not None
                else 1.0
            )
        )

        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)

        # Step size strictly 0.4 * FOV
        fov_min = min(cam_w, cam_h) * effective_pixel_scale
        step_dist = max(50.0, 0.4 * fov_min)

        eff_range = self.scan_range or 10000.0
        if self.grid_size is not None and self.grid_size > 0:
            nx = ny = self.grid_size
        else:
            nx = ny = max(1, int(round(eff_range / step_dist)) + 1)

        total_points = nx * ny
        start_x, start_y = stage.get_raw_position()[:2]
        c0 = (nx - 1) // 2
        r0 = (ny - 1) // 2

        spiral_indices = generate_spiral_grid_indices(nx, ny)
        manifest: List[Dict[str, Any]] = []

        report_progress(0.0, f"Starting spiral scan on {nx}x{ny} grid ({total_points} photos, step {step_dist:.1f} µm)...")

        try:
            for point_idx, (r, c) in enumerate(spiral_indices):
                if self.is_aborted:
                    return "Stage 3 calibration aborted"

                prog = point_idx / total_points
                dx = (c - c0) * step_dist
                dy = (r - r0) * step_dist
                target_x = start_x + dx
                target_y = start_y + dy

                report_progress(
                    prog,
                    f"Photo {point_idx + 1}/{total_points} (Grid [{r}, {c}]): Moving to ({target_x:.1f}, {target_y:.1f}) µm...",
                )
                stage.move_absolute({"x": target_x, "y": target_y}, z_calib_only=True)
                context.delay_func(self.stabilization_delay)

                # Optional autofocus before each photo
                if self.autofocus:
                    report_progress(prog + 0.02 / total_points, f"Autofocusing at point {point_idx + 1}/{total_points}...")
                    af_op = AutofocusOperation()
                    af_reporter = sub_progress(
                        report_progress,
                        prog + 0.02 / total_points,
                        prog + 0.5 / total_points,
                        prefix=f"AF {point_idx + 1}/{total_points}",
                    )
                    af_err = self._run_sub_op(af_op, context, af_reporter)
                    if af_err and self.is_aborted:
                        return "Stage 3 calibration aborted during autofocus"

                frame = camera.get_latest_frame()
                if frame is None:
                    continue

                img_filename = f"scan_{point_idx:04d}_r{r}_c{c}_x{target_x:.1f}_y{target_y:.1f}.jpg"
                img_path = os.path.join(self.output_dir, img_filename)
                cv2.imwrite(img_path, frame)

                actual_pos = stage.get_raw_position()
                manifest.append({
                    "step": point_idx,
                    "grid_index": [r, c],
                    "filename": img_filename,
                    "target_xy": [target_x, target_y],
                    "actual_raw_pos": list(actual_pos),
                })
        finally:
            # Save manifest
            manifest_path = os.path.join(self.output_dir, "scan_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "version": "1.0",
                        "total_photos": len(manifest),
                        "grid_shape": [ny, nx],
                        "step_distance": step_dist,
                        "fov_min": fov_min,
                        "pixel_size": effective_pixel_scale,
                        "photos": manifest,
                    }, f, indent=2)
            except Exception as e:
                print(f"[StageCalibrationScan] Warning: could not write scan_manifest.json: {e}")

        report_progress(1.0, f"Spiral scan complete! Saved {len(manifest)} photos to {self.output_dir}.")
        return None

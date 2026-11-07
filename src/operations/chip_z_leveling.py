# Hacker Fab
# Chip Z Leveling Operation
# Performs 4-point autofocus around the chip corners to establish a temporary leveling plane

import time
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from operations.autofocus import AutofocusOperation


def fit_plane(points: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Fits a plane Delta_Z = a*X + b*Y + c to measured (X, Y, Delta_Z) points using least squares.

    Returns:
        (a, b, c) plane coefficients.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {len(pts)}")
    # A has shape (N, 3): [X, Y, 1]
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)])
    b = pts[:, 2]
    coef, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


class ChipZLevelingOperation(Operation):
    """Temporary chip Z leveling operation using 4-point autofocus plane fitting.

    Saves current XY coordinate, moves to 4 corners of the chip defined by chip_size,
    runs autofocus at each point, returns to the original coordinate, fits a plane to
    the focal height deviations, and sets the plane offset on the StageCalibration.
    """

    def __init__(
        self,
        chip_size_mm: Optional[float] = None,
        chip_size: Optional[float] = None,
        stabilization_delay: float = 0.5,
    ):
        super().__init__("Chip Z Leveling")
        size = chip_size_mm if chip_size_mm is not None else (chip_size if chip_size is not None else 10.0)
        self.chip_size_mm = float(size)
        self.stabilization_delay = float(stabilization_delay)
        self._current_sub_op: Optional[Operation] = None
        self.fitted_plane: Optional[Tuple[float, float, float]] = None
        self.measured_points: List[Tuple[float, float, float]] = []

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
        if context.stage is None:
            return "Stage controller not available in execution context"

        stage = context.stage
        cal = stage.get_calibration()
        if cal is None:
            msg = "No StageCalibration loaded on StageController. Chip Z leveling requires an active stage calibration."
            if context.warning_callback:
                context.warning_callback(msg)
            return msg

        if self.is_aborted:
            return "Chip Z Leveling aborted"

        # Save initial stage position
        orig_x, orig_y, orig_z = stage.get_position()

        # Clear any previously set chip Z plane so measurements are relative to base calibration
        cal.clear_chip_z_plane()

        # Compute 4 measurement corners (in µm) centered around current XY position
        chip_size_um = self.chip_size_mm * 1000.0
        h = chip_size_um / 2.0

        points = [
            (orig_x - h, orig_y - h),
            (orig_x + h, orig_y - h),
            (orig_x + h, orig_y + h),
            (orig_x - h, orig_y + h),
        ]

        report_progress(0.0, f"Starting Chip Z Leveling for {self.chip_size_mm:.1f} mm chip...")
        self.measured_points = []

        try:
            for idx, (px, py) in enumerate(points):
                if self.is_aborted:
                    return "Chip Z Leveling aborted"

                prog_start = (idx / 4.0) * 0.85
                prog_end = ((idx + 1) / 4.0) * 0.85

                report_progress(
                    prog_start,
                    f"Point {idx + 1}/4: Moving to ({px:.1f}, {py:.1f}) µm...",
                )

                # Move in corrected coordinates with z=0.0 (surface height)
                if not stage.move_absolute({"x": px, "y": py, "z": orig_z}):
                    msg = f"Failed to move stage to point {idx + 1} ({px:.1f}, {py:.1f}) µm"
                    if context.warning_callback:
                        context.warning_callback(msg)
                    return msg

                if self.stabilization_delay > 0:
                    context.delay_func(self.stabilization_delay)

                # Autofocus in red light
                af_op = AutofocusOperation(uv_z_offset=0.0)
                af_reporter = sub_progress(
                    report_progress,
                    prog_start + 0.1 * (prog_end - prog_start),
                    prog_end,
                    prefix=f"Point {idx + 1}/4 - AF",
                )
                af_err = self._run_sub_op(af_op, context, af_reporter)
                if self.is_aborted:
                    return "Chip Z Leveling aborted"
                if af_err is not None:
                    return f"Autofocus failed at point {idx + 1} ({px:.1f}, {py:.1f}): {af_err}"

                if af_op.best_red_z is not None:
                    z_focus_corr = float(af_op.best_red_z)
                else:
                    z_focus_corr = float(stage.get_position()[2])

                # Since stage was moved to z=0.0, the best Z in corrected coords is delta_z
                self.measured_points.append((px, py, z_focus_corr - orig_z))

            # Move back to original point
            report_progress(0.85, "Returning stage to original position...")
            if not stage.move_absolute({"x": orig_x, "y": orig_y, "z": orig_z}):
                msg = "Failed to return stage to original position"
                if context.warning_callback:
                    context.warning_callback(msg)
                return msg

            # Calculate chip Z plane using the 4 measured coordinates
            a, b, c = fit_plane(self.measured_points)
            self.fitted_plane = (a, b, c)

            # Apply temporary plane offset to stage calibration
            cal.set_chip_z_plane((a, b, c))

            report_progress(
                1.0,
                f"Chip Z Leveling complete. Plane: ΔZ = {a:.6e}*X + {b:.6e}*Y + {c:.2f} µm",
            )
            return None

        finally:
            # Ensure stage returns to initial position if aborted or error occurred before returning
            current_pos = stage.get_position()
            if (
                abs(current_pos[0] - orig_x) > 1e-3
                or abs(current_pos[1] - orig_y) > 1e-3
                or abs(current_pos[2] - orig_z) > 1e-3
            ):
                stage.move_absolute({"x": orig_x, "y": orig_y, "z": orig_z})


# Alias matching operation naming convention
chip_z_leveling = ChipZLevelingOperation


class LocateChipMarkerOperation(Operation):
    """Records or updates a chip alignment marker (idx=1 or idx=2) at the primary camera crosshair."""

    def __init__(self, idx: int = 1):
        super().__init__(f"Locate Chip Marker #{idx}")
        self.idx = int(idx)

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.idx not in (1, 2):
            msg = f"Invalid marker index {self.idx}. Must be 1 or 2."
            report_progress(1.0, msg)
            return msg

        if context.stage is None:
            msg = "Stage controller not available in execution context"
            report_progress(1.0, msg)
            return msg

        if context.project is None:
            msg = "Chip project not available in execution context"
            report_progress(1.0, msg)
            return msg

        sx, sy, _ = context.stage.get_position()
        cam_px_size = 1.0
        if context.machine_config is not None:
            cam_px_size = float(context.machine_config.stage_calibration.camera_pixel_size)

        if context.camera is not None:
            dx_um, dy_um = context.camera.get_primary_crosshair_offset_um(cam_px_size)
        else:
            dx_um, dy_um = 0.0, 0.0

        marker_corr_xy = (float(sx + dx_um), float(sy + dy_um))
        msg = context.project.chip_progress.locate_marker(self.idx, marker_corr_xy)
        context.project.invalidate_all_tile_caches()

        if context.event_bus is not None:
            context.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, context.project.chip_progress)

        report_progress(1.0, msg)
        return None


locate_chip_marker = LocateChipMarkerOperation


def generate_marker_pattern(idx: int, size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Generates an optical fiducial marker pattern with meeting triangle wedges and marker ID numerals.

    Features two right-angled triangle wedges meeting at the exact center (top-left and bottom-right),
    and numerals matching the marker index in the other two quadrants.
    """
    pw, ph = int(size[0]), int(size[1])
    img = np.zeros((ph, pw, 3), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    R = int(min(pw, ph) * 0.28)

    # Top-left triangle: right angle corner at (cx, cy), hypotenuse from (cx - R, cy) to (cx, cy - R)
    tri_tl = np.array([[cx, cy], [cx - R, cy], [cx, cy - R]], dtype=np.int32)
    cv2.fillPoly(img, [tri_tl], (255, 255, 255))

    # Bottom-right triangle: right angle corner at (cx, cy), hypotenuse from (cx + R, cy) to (cx, cy + R)
    tri_br = np.array([[cx, cy], [cx + R, cy], [cx, cy + R]], dtype=np.int32)
    cv2.fillPoly(img, [tri_br], (255, 255, 255))

    # Text in Quadrant 1 (top-right) and Quadrant 3 (bottom-left)
    text = str(idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, R / 50.0)
    thickness = max(2, int(font_scale * 3.0))

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Q1 (top-right)
    q1_x = int(cx + R * 0.50 - tw / 2)
    q1_y = int(cy - R * 0.50 + th / 2)
    cv2.putText(img, text, (q1_x, q1_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Q3 (bottom-left)
    q3_x = int(cx - R * 0.50 - tw / 2)
    q3_y = int(cy + R * 0.50 + th / 2)
    cv2.putText(img, text, (q3_x, q3_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


class ExposeMarkerAndRecordOperation(Operation):
    """Operation that generates a fiducial marker pattern, autofocuses, exposes on the projector,
    and records the marker location for the selected index (1 or 2). Aborts if index is already recorded.
    """

    def __init__(
        self,
        idx: int = 1,
        exposure_time: Optional[float] = None,
        uv_z_offset: Optional[float] = None,
    ):
        super().__init__(f"Expose Marker {idx} & Record")
        self.idx = int(idx)
        self.exposure_time = exposure_time
        self.uv_z_offset = uv_z_offset
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.idx not in (1, 2):
            msg = f"Marker index must be 1 or 2, got {self.idx}"
            report_progress(1.0, msg)
            return msg

        if context.stage is None:
            msg = "Stage controller not available in execution context"
            report_progress(1.0, msg)
            return msg

        if context.project is None:
            msg = "Chip project not available in execution context"
            report_progress(1.0, msg)
            return msg

        if context.projector is None:
            msg = "Projector controller not available in execution context"
            report_progress(1.0, msg)
            return msg

        cp = context.project.chip_progress
        # Abort and show error if the idx is already recorded
        if (self.idx == 1 and cp.marker1_chip is not None) or (self.idx == 2 and cp.marker2_chip is not None):
            msg = (
                f"Marker {self.idx} is already recorded. "
                "Please reset chip progress before exposing a new reference marker."
            )
            report_progress(1.0, msg)
            if context.warning_callback:
                context.warning_callback(msg)
            return msg

        # Step 1: Autofocus at current position
        report_progress(0.0, f"Autofocusing before exposing Marker {self.idx}...")
        context.projector.set_color_mode(ColorMode.RED)
        af_op = AutofocusOperation(uv_z_offset=self.uv_z_offset)
        self._current_sub_op = af_op
        try:
            err = af_op.execute(
                context,
                sub_progress(report_progress, 0.0, 0.35, prefix=f"Marker {self.idx} AF"),
            )
        finally:
            self._current_sub_op = None

        if err or self.is_aborted:
            msg = err or "Autofocus aborted"
            report_progress(1.0, msg)
            return msg

        # Step 2: Generate marker pattern centered on the projector display
        proj_size = (
            context.projector.projector_size()
            if hasattr(context.projector, "projector_size")
            else context.project.projector_size
        )
        pattern = generate_marker_pattern(self.idx, proj_size)

        # Step 3: Determine exposure duration and expose
        if self.exposure_time is not None:
            duration_ms = float(self.exposure_time)
        elif context.project and context.project.settings.exposure_time > 0:
            duration_ms = float(context.project.settings.exposure_time)
        elif context.machine_config:
            duration_ms = float(context.machine_config.exposure.default_exposure_time_ms)
        else:
            duration_ms = 8000.0

        coords = context.stage.get_position()
        report_progress(0.35, f"Exposing Marker {self.idx} ({int(duration_ms)} ms)...")

        context.projector.set_on(False)
        context.projector.set_generated_image(pattern)
        context.projector.set_image_source(ProjectorImageSource.GENERATED)
        context.projector.set_color_mode(ColorMode.UV)

        context.delay_func(3)
        context.projector.set_on(True)
        context.projector.wait_for_display(timeout=2.0)

        start_datetime = datetime.now()
        start_t = time.time()
        end_t = start_t + (duration_ms / 1000.0)

        try:
            progress_resolution = min((0.1, duration_ms / 1000.0 / 10))
            while time.time() < end_t:
                if self.is_aborted:
                    break
                elapsed = time.time() - start_t
                pct = min(1.0, max(0.0, elapsed / (duration_ms / 1000.0)))
                report_progress(0.35 + 0.55 * pct, f"Exposing Marker {self.idx}... ({int(pct * 100)}%)")
                context.delay_func(progress_resolution)
        finally:
            context.projector.set_on(False)
            context.projector.wait_for_display(timeout=1.0)
            context.projector.set_color_mode(ColorMode.RED)

        elapsed_ms = (time.time() - start_t) * 1000.0
        if self.is_aborted:
            report_progress(1.0, f"Exposure of Marker {self.idx} aborted")
            return f"Exposure of Marker {self.idx} aborted"

        # Step 4: Compute marker location (center of projector display in corr-stage-space)
        sx, sy, sz = float(coords[0]), float(coords[1]), float(coords[2])
        cam_size = (
            context.camera._last_frame_size
            if context.camera and getattr(context.camera, "_last_frame_size", None)
            else context.project.camera_size
        )
        cam_px = 1.0
        proj_px = 1.0
        proj_to_cam = None
        if context.machine_config is not None:
            cam_px = float(context.machine_config.stage_calibration.camera_pixel_size)
            proj_px = float(context.machine_config.optics.projector_pixel_size)
            proj_to_cam = context.machine_config.optics.proj_to_cam_matrix

        from core.coordinate_system import (
            build_camera_to_corr_stage_matrix,
            compute_projector_footprint_in_chip_space,
            get_proj_to_cam_homography,
        )

        pw, ph = proj_size
        H_proj_to_cam = get_proj_to_cam_homography(
            proj_to_cam_matrix=proj_to_cam,
            projector_size=(pw, ph),
            camera_size=cam_size,
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
        )
        p_center_proj = np.array([pw / 2.0, ph / 2.0, 1.0], dtype=np.float64)
        p_cam = H_proj_to_cam @ p_center_proj
        u_cam = float(p_cam[0] / p_cam[2])
        v_cam = float(p_cam[1] / p_cam[2])

        M_cam_to_corr = build_camera_to_corr_stage_matrix(
            stage_xy_corr=(sx, sy),
            camera_size=cam_size,
            camera_pixel_size=cam_px,
        )
        p_corr = M_cam_to_corr @ np.array([u_cam, v_cam, 1.0])
        marker_corr_xy = (float(p_corr[0]), float(p_corr[1]))

        # Record marker coordinate in chip progress
        msg = cp.locate_marker(self.idx, marker_corr_xy)
        context.project.invalidate_all_tile_caches()

        # Record exposure in history
        footprint_chip = compute_projector_footprint_in_chip_space(
            stage_xy_corr=(sx, sy),
            projector_size=proj_size,
            camera_size=cam_size,
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
            proj_to_cam_matrix=proj_to_cam,
            corr_to_chip_matrix=cp.get_corr_to_chip_matrix(),
        )
        from core.chip_project import ExposureRecord
        record = ExposureRecord(
            coords=(float(marker_corr_xy[0]), float(marker_corr_xy[1]), float(sz)),
            time=start_datetime,
            duration=elapsed_ms,
            aborted=self.is_aborted,
            layer_index=None,
            tile_index=None,
            footprint_chip_corners=footprint_chip,
        )
        context.project.add_exposure_record(record)

        if context.event_bus is not None:
            context.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, cp)
            context.event_bus.emit(Event.EXPOSURE_HISTORY_CHANGED, cp.exposure_history)

        final_msg = f"Exposed Marker {self.idx} and recorded at ({marker_corr_xy[0]:.1f}, {marker_corr_xy[1]:.1f}) µm"
        report_progress(1.0, final_msg)
        return None



from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.coordinate_system import compute_projector_pixel_size_from_tags
from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import (
    detect_ar_tags,
    generate_ar_tag_grid,
)
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
)
from operations.maximize_image_sharpness import MaximizeImageSharpnessOperation


class OpticsCalibrationOperation(Operation):
    """Calibrates the chromatic focal Z offset between Red and UV illumination.

    Process:
    1. User places a bare silicon chip on the stage and sets rough manual focus.
    2. Projects ArUco calibration tags in Red illumination via generated image source.
    3. Checks that tags are detected by the camera; aborts with a warning if not detected.
    4. Sweeps Z using iterative ArUco grids and MaximizeImageSharpnessOperation to find Z_red.
    5. Switches to UV illumination and sweeps Z to find Z_uv.
    6. Computes chromatic offset: ΔZ = Z_uv - Z_red.
    7. Disables projector illumination and reports the offset.
    """

    def __init__(
        self,
        search_range: float = 500.0,
        target_accuracy: float = 0.5,
        grid_sizes: Sequence[int] = (2, 4),
        step_configs: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__("Optics Calibration")
        self.search_range = float(search_range)
        self.target_accuracy = float(target_accuracy)
        self.grid_sizes = list(grid_sizes)
        self.red_best_z: Optional[float] = None
        self.uv_best_z: Optional[float] = None
        self.uv_z_offset: Optional[float] = None
        self._current_sub_op: Optional[Operation] = None

        if step_configs is not None:
            self.step_configs = [(float(sr), float(acc)) for sr, acc in step_configs]
        elif len(self.grid_sizes) == 1:
            self.step_configs = [(self.search_range, self.target_accuracy)]
        elif len(self.grid_sizes) == 2:
            mid_range = max(self.target_accuracy * 4, min(self.search_range / 5.0, 20.0))
            mid_acc = max(self.target_accuracy * 2, 2.0)
            self.step_configs = [
                (self.search_range, mid_acc),
                (mid_range, self.target_accuracy),
            ]
        else:
            self.step_configs = [
                (self.search_range, max(self.target_accuracy * 10, 10.0)),
                (
                    max(self.target_accuracy * 4, min(self.search_range / 10.0, 20.0)),
                    max(self.target_accuracy * 2, 2.0),
                ),
                (max(self.target_accuracy * 2, 4.0), self.target_accuracy),
            ]
            while len(self.step_configs) < len(self.grid_sizes):
                self.step_configs.append(
                    (max(self.target_accuracy * 2, 4.0), self.target_accuracy)
                )

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _detect_tags(
        self, frame: Optional[np.ndarray]
    ) -> Tuple[int, Sequence, Optional[np.ndarray]]:
        return detect_ar_tags(frame)

    def _run_focus_iterations(
        self,
        context: ExecutionContext,
        color_mode: ColorMode,
        progress_start: float,
        progress_end: float,
        report_progress: Callable[[float, str], None],
        initial_tag_check: bool = False,
    ) -> Optional[str]:
        projector = context.projector
        proj_size = projector.projector_size()
        mode_name = "UV" if color_mode == ColorMode.UV else "Red"
        num_grids = len(self.grid_sizes)
        progress_span = progress_end - progress_start

        grid_span = progress_span / max(1, num_grids)

        for idx, grid_n in enumerate(self.grid_sizes):
            if self.is_aborted:
                return "Optics calibration aborted"

            total_tags = grid_n * grid_n
            step_start = progress_start + idx * grid_span
            step_end = step_start + grid_span
            report_progress(
                step_start,
                f"Projecting {grid_n}x{grid_n} ArUco grid ({mode_name})...",
            )

            # 1. Project generated ArUco grid
            grid_img = generate_ar_tag_grid(
                grid_n=grid_n,
                canvas_size=proj_size,
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(color_mode)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)
            context.delay_func(1.0)

            if self.is_aborted:
                return "Optics calibration aborted"

            # 2. Determine search range & threshold for this grid level
            sweep_range, threshold = (
                self.step_configs[idx]
                if idx < len(self.step_configs)
                else self.step_configs[-1]
            )

            # 3. Maximize image sharpness via MaximizeImageSharpnessOperation
            sharp_start = step_start + 0.05 * grid_span
            sub_reporter = sub_progress(
                report_progress,
                sharp_start,
                step_end,
                prefix=f"{mode_name} {grid_n}x{grid_n}",
            )
            sharp_op = MaximizeImageSharpnessOperation(
                z_range=sweep_range,
                threshold=threshold,
            )
            self._current_sub_op = sharp_op
            if self.is_aborted:
                sharp_op.abort()

            err = sharp_op.execute(context, report_progress=sub_reporter)
            self._current_sub_op = None

            if self.is_aborted:
                return "Optics calibration aborted"

            if err is not None:
                return f"Optics calibration failed: {err}"

        return None

    def execute(
        self, context: ExecutionContext, report_progress: Callable[[float, str], None]
    ) -> Optional[str]:
        if self.is_aborted:
            return "Optics calibration aborted"

        report_progress(0.0, "Starting Optics Calibration...")
        projector = context.projector
        assert projector is not None, "Projector not found"

        try:
            # 1. Optimize focus in Red light
            red_err = self._run_focus_iterations(
                context=context,
                color_mode=ColorMode.RED,
                progress_start=0.05,
                progress_end=0.50,
                report_progress=report_progress,
                initial_tag_check=True,
            )
            if red_err is not None:
                return red_err

            self.red_best_z = context.stage.get_position()[2]
            report_progress(0.50, f"Red focus locked at Z = {self.red_best_z:.2f} µm")

            if self.is_aborted:
                return "Optics calibration aborted"

            # 2. Switch to UV illumination
            report_progress(0.52, "Switching to UV illumination...")

            # 3. Optimize focus in UV light
            uv_err = self._run_focus_iterations(
                context=context,
                color_mode=ColorMode.UV,
                progress_start=0.55,
                progress_end=0.95,
                report_progress=report_progress,
                initial_tag_check=False,
            )
            if uv_err is not None:
                return uv_err

            self.uv_best_z = context.stage.get_position()[2]

            # 4. Compute UV - Red Z offset
            self.uv_z_offset = self.uv_best_z - self.red_best_z
            report_progress(
                1.0,
                f"Optics Calibration Complete: Red Z = {self.red_best_z:.2f} µm, "
                f"UV Z = {self.uv_best_z:.2f} µm, Offset ΔZ = {self.uv_z_offset:+.2f} µm",
            )
            return None

        finally:
            # Unconditionally turn off projector and restore source
            if projector is not None:
                projector.set_on(False)
                projector.set_generated_image(None)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)


class ProjectorCameraCalibrationOperation(Operation):
    """Calculates projector pixel size (µm/px) and projector-to-camera homography transform.

    Process:
    1. Projects a 2x2 ArUco grid in Red illumination on the projector.
    2. Captures a frame from the camera and detects the ArUco tags.
    3. Uses known camera_pixel_size, ArUco tag positions in projector pixels, and measured
       camera pixel positions to compute projector_pixel_size (µm/px) and 3x3 H_proj_to_cam.
    4. Stores both in machine_config.optics and saves machine_config.toml.
    """

    def __init__(
        self,
        stabilization_delay: float = 0.8,
        save_config: bool = True,
        config_path: str = "machine_config.toml",
    ):
        super().__init__("Projector-Camera Calibration")
        self.stabilization_delay = float(stabilization_delay)
        self.save_config = bool(save_config)
        self.config_path = config_path
        self.projector_pixel_size: Optional[float] = None
        self.homography: Optional[np.ndarray] = None

    def execute(
        self, context: ExecutionContext, report_progress: Callable[[float, str], None]
    ) -> Optional[str]:
        if context.projector is None or context.camera is None:
            return "Projector or Camera not available in execution context"

        if self.is_aborted:
            return "Projector-Camera calibration aborted"

        projector = context.projector
        camera = context.camera
        pw, ph = projector.projector_size()

        cam_px_size = (
            context.machine_config.stage_calibration.camera_pixel_size
            if context.machine_config is not None
            else 1.0
        )

        try:
            report_progress(0.1, "Projecting 2x2 ArUco calibration grid...")
            grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
                grid_n=2,
                canvas_size=(pw, ph),
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(ColorMode.RED)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)
            projector.wait_for_display(timeout=2.0)

            if self.stabilization_delay > 0:
                context.delay_func(self.stabilization_delay)

            if self.is_aborted:
                return "Projector-Camera calibration aborted"

            report_progress(0.5, "Capturing camera frame and detecting ArUco tags...")
            frame = camera.get_latest_frame()
            if frame is None:
                return "Failed to capture camera frame for Projector-Camera calibration"

            det_count, corners, ids = detect_ar_tags(frame)
            if det_count == 0 or ids is None:
                return (
                    "No ArUco tags detected in camera frame. "
                    "Ensure the substrate is in focus, projector is visible, and camera flip mode is configured."
                )

            report_progress(0.75, f"Detected {det_count} ArUco tags. Computing pixel size and homography...")
            proj_px_size = compute_projector_pixel_size_from_tags(
                detected_corners=corners,
                detected_ids=ids,
                known_tag_coords=tag_coords,
                camera_pixel_size=cam_px_size,
            )
            H_proj_to_cam, matched_pts = compute_projector_to_camera_homography(
                detected_corners=list(corners),
                detected_ids=ids,
                known_tag_coords=tag_coords,
            )

            if proj_px_size is None or H_proj_to_cam is None or matched_pts < 4:
                return (
                    f"Projector-Camera calibration failed: insufficient tag corner matches ({matched_pts} points). "
                    "At least 1 full tag (4 corners) is required."
                )

            self.projector_pixel_size = float(proj_px_size)
            self.homography = H_proj_to_cam

            if context.machine_config is not None:
                context.machine_config.optics.projector_pixel_size = self.projector_pixel_size
                context.machine_config.optics.proj_to_cam_matrix = H_proj_to_cam.tolist()
                if self.save_config:
                    try:
                        context.machine_config.save(self.config_path)
                    except Exception as e:
                        print(f"[ProjectorCameraCalibration] Warning saving config: {e}")

            if context.project is not None:
                context.project.camera_size = (frame.shape[1], frame.shape[0])
                context.project.invalidate_all_tile_caches()

            if context.event_bus is not None and context.machine_config is not None:
                context.event_bus.emit(Event.MACHINE_CONFIG_CHANGED, context.machine_config)
                context.event_bus.emit(Event.EXPOSURE_CONFIG_CHANGED)

            report_progress(
                1.0,
                f"Projector-Camera calibration complete: projector pixel size = {self.projector_pixel_size:.4f} µm/px "
                f"({det_count} tags, {matched_pts} corners matched).",
            )
            return None

        finally:
            projector.set_on(False)
            projector.set_generated_image(None)
            projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)



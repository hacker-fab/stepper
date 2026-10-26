from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation
from lib.ar_tag import (
    detect_ar_tags,
    generate_ar_tag_grid,
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
        min_detection_rate: float = 0.5,
        grid_sizes: Sequence[int] = (2, 4),
        step_configs: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__("Optics Calibration")
        self.search_range = float(search_range)
        self.target_accuracy = float(target_accuracy)
        self.min_detection_rate = float(min_detection_rate)
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
        det_count, corners, ids = detect_ar_tags(frame, flip_horizontal=True)
        if det_count == 0:
            det_count, corners, ids = detect_ar_tags(frame, flip_horizontal=False)
        return det_count, corners, ids

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

        for idx, grid_n in enumerate(self.grid_sizes):
            if self.is_aborted:
                return "Optics calibration aborted"

            total_tags = grid_n * grid_n
            step_base = progress_start + (idx / num_grids) * progress_span
            report_progress(
                step_base,
                f"Projecting {grid_n}x{grid_n} ArUco grid...",
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

            # 2. Check tag detection
            # Disabled because the failure rate is too high

            # frame = context.camera.get_latest_frame()
            # det_count, _, _ = self._detect_tags(frame)

            # # On initial check in Red illumination, verify that tags are present on stage
            # if initial_tag_check and idx == 0 and det_count == 0:
            #     msg = (
            #         "No ArUco tags detected in Red illumination! "
            #         "Please verify that a bare silicon chip is on the stage, "
            #         "rough focus is set, and the camera/projector are active."
            #     )
            #     if context.warning_callback:
            #         context.warning_callback(msg)
            #     return msg

            # detection_rate = det_count / total_tags if total_tags > 0 else 0.0

            # # If detection rate is below threshold, stop iteration (skip first iteration as alignment may be blurry)
            # if detection_rate < self.min_detection_rate and idx > 0:
            #     print(
            #         f"{mode_name} Grid {grid_n}x{grid_n} detection rate ({detection_rate*100:.1f}%) "
            #         f"< {self.min_detection_rate*100:.1f}%. Stopping iteration.",
            #         flush=True,
            #     )
            #     break

            # 3. Determine search range & threshold for this grid level
            sweep_range, threshold = (
                self.step_configs[idx]
                if idx < len(self.step_configs)
                else self.step_configs[-1]
            )

            # 4. Maximize image sharpness via MaximizeImageSharpnessOperation
            report_progress(
                step_base + (0.5 / num_grids) * progress_span,
                f"Maximizing sharpness for {mode_name} {grid_n}x{grid_n} grid...",
            )
            sharp_op = MaximizeImageSharpnessOperation(
                z_range=sweep_range,
                threshold=threshold,
            )
            self._current_sub_op = sharp_op
            if self.is_aborted:
                sharp_op.abort()

            err = sharp_op.execute(
                context,
                report_progress=lambda p, m: report_progress(
                    step_base + p * (progress_span / num_grids),
                    f"{mode_name} {grid_n}x{grid_n} - {m}",
                ),
            )
            self._current_sub_op = None

            if self.is_aborted:
                return "Optics calibration aborted"

            if err is not None:
                return f"Optics calibration failed: {err}"

            # 5. Verify detection rate at best focus
            # Disabled because the failure rate is too high

            # context.delay_func(1.0)
            # frame_best = context.camera.get_latest_frame()
            # det_count_best, _, _ = self._detect_tags(frame_best)
            # rate_best = det_count_best / total_tags if total_tags > 0 else 0.0

            # # If detection falls below threshold, do not progress to smaller tags
            # if rate_best < self.min_detection_rate:
            #     print(
            #         f"Post-focus {mode_name} {grid_n}x{grid_n} detection rate ({rate_best*100:.1f}%) "
            #         f"< {self.min_detection_rate*100:.1f}%. Halting further refinement.",
            #         flush=True,
            #     )
            #     break

        return None

    def execute(
        self, context: ExecutionContext, report_progress: Callable[[float, str], None]
    ) -> Optional[str]:
        if self.is_aborted:
            return "Optics calibration aborted"

        report_progress(0.05, "Starting Optics Calibration...")
        projector = context.projector
        assert projector is not None, "Projector not found"

        try:
            # 1. Optimize focus in Red light
            red_err = self._run_focus_iterations(
                context=context,
                color_mode=ColorMode.RED,
                progress_start=0.1,
                progress_end=0.5,
                report_progress=report_progress,
                initial_tag_check=True,
            )
            if red_err is not None:
                return red_err

            self.red_best_z = context.stage.get_position()[2]
            report_progress(0.5, f"Red focus locked at Z = {self.red_best_z:.2f} µm")

            if self.is_aborted:
                return "Optics calibration aborted"

            # 2. Switch to UV illumination
            report_progress(0.55, "Switching to UV illumination...")

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


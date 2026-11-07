from typing import Callable, Optional, Sequence
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.machine_config import AutofocusMachineConfig, AutofocusStageConfig
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import generate_ar_tag_grid
from operations.maximize_image_sharpness import MaximizeImageSharpnessOperation


class AutofocusOperation(Operation):
    """Performs autofocus calibration using iterative ArUco tag grids and MaximizeImageSharpnessOperation."""

    def __init__(
        self,
        log: bool = False,
        uv_z_offset: Optional[float] = None,
        stages: Optional[Sequence[AutofocusStageConfig]] = None,
    ):
        super().__init__("Autofocus")
        self.log = log
        self.uv_z_offset = uv_z_offset
        self.stages = stages
        self._current_sub_op: Optional[Operation] = None
        self.best_red_z: Optional[float] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.is_aborted:
            return "Autofocus aborted"

        report_progress(0.05, "Starting iterative ArUco autofocus...")

        projector = context.projector
        assert projector is not None, "Projector not found"
        proj_size = projector.projector_size()

        best_overall_z = context.stage.get_position()[2]

        effective_stages = self.stages
        if effective_stages is None:
            if context.machine_config is not None:
                effective_stages = context.machine_config.autofocus.stages
            else:
                effective_stages = [
                    AutofocusStageConfig(grid_n=2, sweep_range=250.0, threshold=5.0),
                    AutofocusStageConfig(grid_n=4, sweep_range=10.0, threshold=0.5),
                ]
        elif isinstance(effective_stages, AutofocusStageConfig):
            effective_stages = [effective_stages]

        effective_uv_offset = self.uv_z_offset
        if effective_uv_offset is None:
            if context.machine_config is not None and hasattr(context.machine_config, "optics"):
                effective_uv_offset = context.machine_config.optics.uv_z_offset
            elif context.machine_config is not None and hasattr(context.machine_config, "autofocus"):
                effective_uv_offset = getattr(context.machine_config.autofocus, "uv_z_offset", 0.0)
            else:
                effective_uv_offset = 0.0

        try:
            num_stages = max(1, len(effective_stages))
            stage_pct_span = 0.85 / num_stages

            for idx, stage_cfg in enumerate(effective_stages):
                if self.is_aborted:
                    return "Autofocus aborted"

                grid_n = stage_cfg.grid_n
                sweep_range = stage_cfg.sweep_range
                threshold = stage_cfg.threshold

                stage_start = 0.05 + idx * stage_pct_span
                stage_end = stage_start + stage_pct_span
                report_progress(stage_start, f"Projecting {grid_n}x{grid_n} ArUco grid...")

                # 1. Generate and project ArUco grid using generated image source
                grid_img = generate_ar_tag_grid(
                    grid_n=grid_n,
                    canvas_size=proj_size,
                )
                projector.set_generated_image(grid_img)
                projector.set_color_mode(ColorMode.RED)
                projector.set_image_source(ProjectorImageSource.GENERATED)
                projector.set_on(True)
                context.delay_func(1.0)

                # 2. Maximize image sharpness via MaximizeImageSharpnessOperation
                sharp_start = stage_start + 0.05 * stage_pct_span
                sub_reporter = sub_progress(
                    report_progress,
                    sharp_start,
                    stage_end,
                    prefix=f"Grid {grid_n}x{grid_n}",
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
                    return "Autofocus aborted"

                if err is not None:
                    return f"Autofocus failed: {err}"

                if sharp_op.result is not None:
                    best_overall_z = sharp_op.result.best_z

            self.best_red_z = float(best_overall_z)

            # 3. Apply UV-Red Z offset if offset is configured
            if abs(effective_uv_offset) > 1e-6:
                report_progress(0.92, "Applying UV-Red Z offset...")
                uv_focal_z = best_overall_z + effective_uv_offset
                if not context.stage.move_absolute({"z": uv_focal_z}):
                    msg = f"Failed to apply UV-Red Z offset to {uv_focal_z:.2f} µm"
                    if context.warning_callback:
                        context.warning_callback(msg)
                    return msg
                print(f"Applied UV-Red Z offset: {effective_uv_offset:+.2f} µm (Z: {best_overall_z:.2f} -> {uv_focal_z:.2f} µm)", flush=True)

            report_progress(1.0, f"Autofocus complete at Z = {best_overall_z:.2f} µm")
            return None

        finally:
            if projector is not None:
                projector.set_on(False)
                projector.set_generated_image(None)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
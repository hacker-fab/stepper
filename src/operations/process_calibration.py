from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.gen_calibration_pattern import generate_litho_target
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation


def generate_spiral_offsets(count: int, step_distance: float) -> List[Tuple[float, float]]:
    """Generates `count` (dx, dy) coordinate offsets in spiral order starting from center (0, 0).
    
    Order:
      0: (0, 0)
      1: (+d, 0)
      2: (+d, +d)
      3: (0, +d)
      4: (-d, +d)
      ...
    """
    if count <= 0:
        return []
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    if count == 1:
        return offsets

    x, y = 0, 0
    dx, dy = 1, 0
    segment_length = 1
    segment_passed = 0
    turns = 0

    while len(offsets) < count:
        x += dx
        y += dy
        offsets.append((float(x * step_distance), float(y * step_distance)))
        segment_passed += 1

        if segment_passed == segment_length:
            segment_passed = 0
            # Turn 90 degrees CCW: (1,0) -> (0,1) -> (-1,0) -> (0,-1)
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                segment_length += 1

    return offsets[:count]


class ProcessCalibrationOperation(Operation):
    """Automates process calibration (FEM) across an array of positions in a 2D spiral order.
    
    At each position:
      1. Moves stage to spiral location (starting at center for step 0).
      2. Performs autofocus (using configured autofocus offset).
      3. Applies Z sweep offset on top of the autofocus position.
      4. Generates calibration pattern fitting the projector resolution with exposure length and Z offset as title.
      5. Sets generated pattern to projector and exposes for the configured duration.
    """

    def __init__(
        self,
        min_exposure: Optional[float] = None,
        max_exposure: Optional[float] = None,
        sweep_steps: int = 5,
        motion_distance: float = 1000.0,
        min_z_offset: float = 0.0,
        max_z_offset: float = 0.0,
        z_steps: int = 1,
        uv_z_offset: Optional[float] = None,
    ):
        super().__init__("Process Calibration")
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.sweep_steps = sweep_steps
        self.motion_distance = motion_distance
        self.min_z_offset = min_z_offset
        self.max_z_offset = max_z_offset
        self.z_steps = z_steps
        self.uv_z_offset = uv_z_offset
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        progress_cb: Callable[[float, str], None],
    ) -> Optional[str]:
        if self.is_aborted:
            return "Aborted"
        self._current_sub_op = op
        try:
            return op.execute(context, progress_cb)
        finally:
            self._current_sub_op = None

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.is_aborted:
            return "Process calibration aborted"

        projector = context.projector
        stage = context.stage
        if projector is None or stage is None:
            return "Projector or stage not found in context"

        default_exp_s = (
            context.machine_config.exposure.default_exposure_time_ms / 1000.0
            if context.machine_config is not None
            else 5.0
        )
        min_exp = self.min_exposure if self.min_exposure is not None else default_exp_s
        max_exp = self.max_exposure if self.max_exposure is not None else default_exp_s

        total_exp_steps = max(1, int(self.sweep_steps))
        if total_exp_steps == 1:
            exposures = [float(min_exp)]
        else:
            exposures = [float(e) for e in np.linspace(min_exp, max_exp, total_exp_steps)]

        total_z_steps = max(1, int(self.z_steps))
        if total_z_steps == 1:
            z_offsets = [float(self.min_z_offset)]
        else:
            z_offsets = [float(z) for z in np.linspace(self.min_z_offset, self.max_z_offset, total_z_steps)]

        # 2D Focus-Exposure Matrix combinations (exposure x z_offset)
        pairs: List[Tuple[float, float]] = [
            (exp, z_off) for exp in exposures for z_off in z_offsets
        ]
        total_steps = len(pairs)
        offsets = generate_spiral_offsets(total_steps, self.motion_distance)

        # Record starting position
        start_x, start_y, _ = stage.get_position()
        pw, ph = projector.projector_size()
        square_size = min(pw, ph)

        report_progress(0.0, f"Starting process calibration ({total_steps} steps: {total_exp_steps} exp x {total_z_steps} z)...")

        last_error: Optional[str] = None
        try:
            for step_idx, ((exp_s, z_off), (off_x, off_y)) in enumerate(zip(pairs, offsets)):
                if self.is_aborted:
                    break

                pct_base = step_idx / total_steps
                pct_step = 1.0 / total_steps
                step_num = step_idx + 1

                # 1. Move stage (step 0 starts at current position, subsequent steps move to spiral offset)
                if step_idx > 0:
                    target_x = start_x + off_x
                    target_y = start_y + off_y
                    report_progress(
                        pct_base,
                        f"Step {step_num}/{total_steps}: Moving to spiral pos ({target_x:.1f}, {target_y:.1f}) µm...",
                    )
                    jog_reporter = sub_progress(
                        report_progress,
                        pct_base,
                        pct_base + 0.15 * pct_step,
                        prefix=f"Step {step_num}/{total_steps}",
                    )
                    jog_op = JogOperation({"x": target_x, "y": target_y}, relative=False)
                    err = self._run_sub_op(jog_op, context, jog_reporter)
                    if err or self.is_aborted:
                        last_error = err
                        break

                # 2. Autofocus
                report_progress(
                    pct_base + 0.15 * pct_step,
                    f"Step {step_num}/{total_steps}: Autofocusing...",
                )
                af_op = AutofocusOperation(uv_z_offset=self.uv_z_offset)
                af_reporter = sub_progress(
                    report_progress,
                    pct_base + 0.15 * pct_step,
                    pct_base + 0.50 * pct_step,
                    prefix=f"Step {step_num}/{total_steps} - AF",
                )
                err = self._run_sub_op(af_op, context, af_reporter)
                if err or self.is_aborted:
                    if err:
                        print(f"Autofocus failed at step {step_num}: {err}, skipping to next step")
                        continue
                    break

                # 3. Apply Z sweep offset on top of autofocus offset
                if abs(z_off) > 1e-6:
                    curr_pos = stage.get_position()
                    target_z = curr_pos[2] + z_off
                    report_progress(
                        pct_base + 0.52 * pct_step,
                        f"Step {step_num}/{total_steps}: Applying Z offset {z_off:+.2f} µm (Z: {curr_pos[2]:.2f} -> {target_z:.2f} µm)...",
                    )
                    if not stage.move_absolute({"z": target_z}):
                        msg = f"Failed to apply Z offset {z_off:+.2f} µm at step {step_num}"
                        if context.warning_callback:
                            context.warning_callback(msg)
                        return msg
                    context.delay_func(1)

                # 4. Generate pattern fitting projector resolution with exposure length and Z offset as title
                report_progress(
                    pct_base + 0.55 * pct_step,
                    f"Step {step_num}/{total_steps}: Generating pattern for {exp_s:.4g}s, {z_off:+.2f}µm...",
                )
                if abs(z_off) > 1e-6 or total_z_steps > 1:
                    title = f"{exp_s:.4g}s | {z_off:+.2f}µm"
                else:
                    title = f"{exp_s:.4g}s"
                pattern_sq = generate_litho_target(
                    size=square_size,
                    main_text=title,
                )

                # Fit pattern into projector canvas
                canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
                if pattern_sq.ndim == 2:
                    pattern_bgr = cv2.cvtColor(pattern_sq, cv2.COLOR_GRAY2BGR)
                else:
                    pattern_bgr = pattern_sq

                y_offset = (ph - square_size) // 2
                x_offset = (pw - square_size) // 2
                canvas[y_offset : y_offset + square_size, x_offset : x_offset + square_size] = pattern_bgr

                # 5. Set generated image and expose
                projector.set_generated_image(canvas)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                duration_ms = exp_s * 1000.0
                report_progress(
                    pct_base + 0.6 * pct_step,
                    f"Step {step_num}/{total_steps}: Exposing {exp_s:g}s ({int(duration_ms)} ms)...",
                )
                exp_op = ExposureOperation(layer_index=None, exposure_time=duration_ms)
                exp_reporter = sub_progress(
                    report_progress,
                    pct_base + 0.60 * pct_step,
                    pct_base + 1.0 * pct_step,
                    prefix=f"Step {step_num}/{total_steps}",
                )
                err = self._run_sub_op(exp_op, context, exp_reporter)
                if err or self.is_aborted:
                    last_error = err
                    break

        finally:
            if projector is not None:
                projector.set_on(False)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

        if self.is_aborted:
            report_progress(1.0, "Process calibration aborted")
            return "Process calibration aborted"
        elif last_error is not None:
            msg = f"Process calibration failed: {last_error}"
            report_progress(1.0, msg)
            return msg
        else:
            report_progress(1.0, "Process calibration complete")
            return None

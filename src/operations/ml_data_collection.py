from datetime import datetime
from enum import auto
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource, StrAutoEnum
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import detect_ar_tags
from lib.cross_marker import generate_cross_pattern
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_marker_to_camera_space,
    warp_projector_image_to_camera,
)
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation
from operations.process_calibration import generate_spiral_offsets


class PatternSource(StrAutoEnum):
    """Pattern source options for ML data collection."""

    GENERATED_MARKER = auto()
    IMAGE_FOLDER = auto()


class MLDataCollectionOperation(Operation):
    """Automates ML dataset generation with projector-to-camera calibration, spiral patterning, and ground-truth annotation."""

    def __init__(
        self,
        pattern_source: str = PatternSource.GENERATED_MARKER.value,
        image_folder: str = "",
        total_patterns: int = 10,
        pattern_gap: float = 1000.0,
        target_scale_pct: float = 8.0,
        scale_jitter_pct: float = 2.0,
        marker_count: int = 20,
        min_exposure: Optional[float] = None,
        max_exposure: Optional[float] = None,
        min_thickness: float = 0.2,
        max_thickness: float = 1.0,
        stabilization_delay: float = 1.0,
        save_directory: str = "stepper_captures/ml_data",
        grid_n: int = 2,
        uv_z_offset: Optional[float] = None,
        exposure_time: Optional[float] = None,
    ):
        super().__init__("ML Data Collection")
        raw_source = pattern_source
        self.pattern_source = raw_source.value if hasattr(raw_source, "value") else str(raw_source)
        self.image_folder = str(image_folder)
        self.total_patterns = int(total_patterns)
        self.pattern_gap = float(pattern_gap)
        self.target_scale_pct = float(target_scale_pct)
        self.scale_jitter_pct = float(scale_jitter_pct)
        self.marker_count = int(marker_count)
        if exposure_time is not None:
            self.min_exposure = float(exposure_time)
            self.max_exposure = float(exposure_time)
        else:
            self.min_exposure = float(min_exposure) if min_exposure is not None else None
            self.max_exposure = float(max_exposure) if max_exposure is not None else None
        self.min_thickness = float(min_thickness)
        self.max_thickness = float(max_thickness)
        self.stabilization_delay = float(stabilization_delay)
        self.save_directory = str(save_directory)
        self.grid_n = int(grid_n)
        self.uv_z_offset = uv_z_offset

        self._current_sub_op: Optional[Operation] = None
        self.homography: Optional[np.ndarray] = None
        self.dataset_records: List[Dict[str, Any]] = []

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

    def execute(
        self, context: ExecutionContext, report_progress: Callable[[float, str], None]
    ) -> Optional[str]:
        if self.is_aborted:
            return "ML data collection aborted"

        projector = context.projector
        stage = context.stage
        camera = context.camera
        if projector is None or stage is None or camera is None:
            return "Projector, stage, or camera not found in context"

        pw, ph = projector.projector_size()
        os.makedirs(self.save_directory, exist_ok=True)

        effective_uv_offset = (
            self.uv_z_offset
            if self.uv_z_offset is not None
            else (
                (
                    context.machine_config.optics.uv_z_offset
                    if hasattr(context.machine_config, "optics")
                    else getattr(context.machine_config.autofocus, "uv_z_offset", 0.0)
                )
                if context.machine_config is not None
                else 0.0
            )
        )

        default_exp_s = (
            context.machine_config.exposure.default_exposure_time_ms / 1000.0
            if context.machine_config is not None
            else 8.0
        )
        eff_min_exp = self.min_exposure if self.min_exposure is not None else default_exp_s
        eff_max_exp = self.max_exposure if self.max_exposure is not None else default_exp_s

        # Validate image folder if image folder mode
        image_paths: List[str] = []
        if self.pattern_source == PatternSource.IMAGE_FOLDER.value:
            if not os.path.isdir(self.image_folder):
                return f"Image folder does not exist or is not a directory: {self.image_folder}"
            for fname in sorted(os.listdir(self.image_folder)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(os.path.join(self.image_folder, fname))
            if not image_paths:
                return f"No valid image files found in folder: {self.image_folder}"

        report_progress(0.0, "Starting ML data collection...")

        try:
            # =========================================================================
            # STEP 1: INITIAL CALIBRATION (Homography estimation)
            # =========================================================================
            report_progress(0.05, "Step 1: Autofocusing without UV Z offset for calibration...")

            # 1.1 Autofocus without UV offset
            af_cal_op = AutofocusOperation(uv_z_offset=0.0)
            af_reporter = sub_progress(report_progress, 0.05, 0.15, prefix="Initial AF")
            err = self._run_sub_op(
                af_cal_op,
                context,
                af_reporter,
            )
            if err or self.is_aborted:
                return err or "ML data collection aborted during initial AF"

            init_red_z = stage.get_position()[2]

            # 1.2 Project ArUco tag grid in RED
            report_progress(0.16, f"Projecting {self.grid_n}x{self.grid_n} ArUco calibration grid in Red...")
            grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
                grid_n=self.grid_n,
                canvas_size=(pw, ph),
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(ColorMode.RED)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)

            context.delay_func(self.stabilization_delay)

            if self.is_aborted:
                return "ML data collection aborted"

            # 1.3 Detect ArUco markers and compute homography
            report_progress(0.20, "Detecting ArUco markers for projector-to-camera alignment...")
            cal_frame = camera.get_latest_frame()
            if cal_frame is None:
                return "Failed to acquire camera frame for calibration"

            det_count, corners, ids = detect_ar_tags(cal_frame)

            homography, matched_count = compute_projector_to_camera_homography(
                corners, ids, tag_coords
            )

            if homography is None or matched_count < 4:
                return (
                    f"Projector-to-camera calibration failed: matched {matched_count} points "
                    f"(minimum 4 required). Ensure the substrate is in focus and tags are visible."
                )

            self.homography = homography
            report_progress(0.25, "Calibration complete. Projector-to-camera homography established.")

            # Turn off projector after calibration
            projector.set_on(False)

            # =========================================================================
            # STEP 2: SPIRAL PATTERNING & DATA ACQUISITION
            # =========================================================================
            total_steps = max(1, self.total_patterns)
            spiral_offsets = generate_spiral_offsets(total_steps, self.pattern_gap)

            start_x, start_y, _ = stage.get_position()
            available_images = list(image_paths)

            step_pct_span = 0.73 / total_steps

            for step_idx in range(total_steps):
                if self.is_aborted:
                    break

                step_num = step_idx + 1
                base_pct = 0.25 + step_idx * step_pct_span
                off_x, off_y = spiral_offsets[step_idx]

                # 2.1 Move stage along spiral
                if step_idx > 0:
                    target_x = start_x + off_x
                    target_y = start_y + off_y
                    report_progress(
                        base_pct,
                        f"Pattern {step_num}/{total_steps}: Moving stage to ({target_x:.1f}, {target_y:.1f}) µm...",
                    )
                    jog_reporter = sub_progress(
                        report_progress,
                        base_pct,
                        base_pct + 0.08 * step_pct_span,
                        prefix=f"Pattern {step_num}/{total_steps}",
                    )
                    jog_op = JogOperation({"x": target_x, "y": target_y}, relative=False)
                    err = self._run_sub_op(jog_op, context, jog_reporter)
                    if err or self.is_aborted:
                        return err or "ML data collection aborted during stage movement"

                # 2.2 Autofocus with configured UV offset
                report_progress(
                    base_pct + 0.10 * step_pct_span,
                    f"Pattern {step_num}/{total_steps}: Autofocusing...",
                )
                af_op = AutofocusOperation(uv_z_offset=effective_uv_offset)
                af_reporter = sub_progress(
                    report_progress,
                    base_pct + 0.10 * step_pct_span,
                    base_pct + 0.40 * step_pct_span,
                    prefix=f"Pattern {step_num}/{total_steps} - AF",
                )
                err = self._run_sub_op(af_op, context, af_reporter)
                if err or self.is_aborted:
                    if err:
                        print(f"Autofocus failed at step {step_num}: {err}, skipping", flush=True)
                        continue
                    else:
                        print("ML data collection aborted during autofocus", flush=True)
                        return "ML data collection aborted during autofocus"

                # Determine red focus Z to return to after exposure
                if af_op.best_red_z is not None:
                    red_focus_z = af_op.best_red_z
                else:
                    red_focus_z = stage.get_position()[2] - effective_uv_offset

                # 2.3 Generate or load pattern
                selected_image_path: Optional[str] = None
                markers_spec = None
                if self.pattern_source == PatternSource.IMAGE_FOLDER.value:
                    if not available_images:
                        available_images = list(image_paths)
                        random.shuffle(available_images)
                    selected_image_path = available_images.pop()

                    report_progress(
                        base_pct + 0.40 * step_pct_span,
                        f"Pattern {step_num}/{total_steps}: Loading image ({os.path.basename(selected_image_path)})...",
                    )
                    loaded_img = cv2.imread(selected_image_path, cv2.IMREAD_COLOR)
                    if loaded_img is None:
                        return f"Failed to load image: {selected_image_path}"
                    pattern_canvas = cv2.resize(loaded_img, (pw, ph), interpolation=cv2.INTER_LINEAR)
                    pattern_canvas = cv2.cvtColor(pattern_canvas, cv2.COLOR_BGR2GRAY)
                    pattern_canvas = cv2.threshold(pattern_canvas, 20, 255, cv2.THRESH_BINARY)[1]
                else:
                    report_progress(
                        base_pct + 0.40 * step_pct_span,
                        f"Pattern {step_num}/{total_steps}: Generating cross pattern...",
                    )
                    pattern_canvas, markers_spec = generate_cross_pattern(
                        canvas_size=(pw, ph),
                        target_scale_pct=self.target_scale_pct,
                        scale_jitter_pct=self.scale_jitter_pct,
                        marker_count=self.marker_count,
                        min_thickness=self.min_thickness,
                        max_thickness=self.max_thickness,
                    )

                # 2.4 Expose pattern in UV
                projector.set_generated_image(pattern_canvas)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                exp_min = min(eff_min_exp, eff_max_exp)
                exp_max = max(eff_min_exp, eff_max_exp)
                step_exposure_sec = float(random.uniform(exp_min, exp_max))
                duration_ms = step_exposure_sec * 1000.0

                report_progress(
                    base_pct + 0.50 * step_pct_span,
                    f"Pattern {step_num}/{total_steps}: Exposing pattern ({step_exposure_sec:.2f}s)...",
                )
                exp_op = ExposureOperation(layer_index=None, exposure_time=duration_ms)
                exp_reporter = sub_progress(
                    report_progress,
                    base_pct + 0.50 * step_pct_span,
                    base_pct + 0.75 * step_pct_span,
                    prefix=f"Pattern {step_num}/{total_steps} - Exposure",
                )
                err = self._run_sub_op(exp_op, context, exp_reporter)
                if err or self.is_aborted:
                    return err or "ML data collection aborted during exposure"

                # 2.5 Return stage to Red focus Z position
                report_progress(
                    base_pct + 0.77 * step_pct_span,
                    f"Pattern {step_num}/{total_steps}: Returning to Red focus Z ({red_focus_z:.2f} µm)...",
                )
                if not stage.move_absolute({"z": red_focus_z}):
                    msg = f"Failed to return stage to Red focus Z = {red_focus_z:.2f} µm"
                    if context.warning_callback:
                        context.warning_callback(msg)
                    return msg

                # 2.6 Switch projector to solid Red illumination
                projector.set_image_source(ProjectorImageSource.SOLID)
                projector.set_color_mode(ColorMode.RED)
                projector.set_on(True)

                # 2.7 Wait for image to stabilize
                report_progress(
                    base_pct + 0.82 * step_pct_span,
                    f"Pattern {step_num}/{total_steps}: Waiting for stabilization...",
                )
                context.delay_func(self.stabilization_delay)

                if self.is_aborted:
                    break

                # 2.8 Capture photo (average 30 frames to reduce noise)
                report_progress(
                    base_pct + 0.84 * step_pct_span,
                    f"Pattern {step_num}/{total_steps}: Capturing and averaging frames...",
                )
                captured_frames = []
                for _ in range(30):
                    if self.is_aborted:
                        break
                    frame = camera.get_latest_frame()
                    if frame is None:
                        return f"Failed to capture camera frame at step {step_num}"
                    captured_frames.append(frame)
                    context.delay_func(0.033)

                if self.is_aborted:
                    break

                captured_frame = np.mean(captured_frames, axis=0).round().astype(np.uint8)

                cam_h, cam_w = captured_frame.shape[:2]

                # 2.9 Save captured photo
                img_filename = f"pattern_{step_num:04d}.png"
                img_path = os.path.join(self.save_directory, img_filename)
                cv2.imwrite(img_path, captured_frame)

                # 2.10 Warp and save projected ground-truth pattern
                gt_filename = f"pattern_{step_num:04d}_projected.png"
                gt_path = os.path.join(self.save_directory, gt_filename)
                warped_gt = warp_projector_image_to_camera(
                    pattern_canvas, homography, (cam_w, cam_h)
                )
                cv2.imwrite(gt_path, warped_gt)

                # 2.11 Transform all marker coordinates to camera space if generated markers
                transformed_markers = None
                if markers_spec is not None:
                    transformed_markers = [
                        transform_marker_to_camera_space(m, homography)
                        for m in markers_spec
                    ]

                # 2.12 Write individual JSON annotation
                json_filename = f"pattern_{step_num:04d}.json"
                json_path = os.path.join(self.save_directory, json_filename)
                curr_pos = stage.get_position()
                record: Dict[str, Any] = {
                    "image_filename": img_filename,
                    "projected_gt_filename": gt_filename,
                    "step_index": step_num,
                    "exposure_time": round(step_exposure_sec, 3),
                    "pattern_source": self.pattern_source,
                    "stage_position": {
                        "x": float(curr_pos[0]),
                        "y": float(curr_pos[1]),
                        "z_captured": float(curr_pos[2]),
                        "z_red_focus": float(red_focus_z),
                    },
                }
                if self.pattern_source == PatternSource.IMAGE_FOLDER.value:
                    if selected_image_path is not None:
                        record["source_image"] = os.path.basename(selected_image_path)
                else:
                    if transformed_markers is not None:
                        record["marker_count"] = len(transformed_markers)
                        record["markers"] = transformed_markers

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

                self.dataset_records.append(record)

            # 2.13 Write master dataset metadata file
            dataset_meta_path = os.path.join(self.save_directory, "dataset_metadata.json")
            master_metadata = {
                "dataset_name": "ML Data Collection",
                "timestamp": datetime.now().isoformat(),
                "projector_size": [pw, ph],
                "homography_matrix": homography.tolist(),
                "total_patterns_collected": len(self.dataset_records),
                "config": {
                    "pattern_source": self.pattern_source,
                    "image_folder": self.image_folder if self.pattern_source == PatternSource.IMAGE_FOLDER.value else None,
                    "total_patterns": self.total_patterns,
                    "pattern_gap": self.pattern_gap,
                    "target_scale_pct": self.target_scale_pct,
                    "scale_jitter_pct": self.scale_jitter_pct,
                    "marker_count": self.marker_count,
                    "min_thickness": self.min_thickness,
                    "max_thickness": self.max_thickness,
                    "min_exposure": eff_min_exp,
                    "max_exposure": eff_max_exp,
                    "stabilization_delay": self.stabilization_delay,
                },
                "records": self.dataset_records,
            }
            with open(dataset_meta_path, "w", encoding="utf-8") as f:
                json.dump(master_metadata, f, indent=2)

            report_progress(1.0, f"ML Data Collection Complete: {len(self.dataset_records)} patterns collected.")
            return None

        finally:
            if projector is not None:
                projector.set_on(False)
                projector.set_generated_image(None)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

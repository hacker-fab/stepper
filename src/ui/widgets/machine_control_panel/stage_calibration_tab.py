import json
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations.stage_calibration import (
    StageCalibrationPatternOperation,
    StageCalibrationScanOperation,
    StageCalibrationZFocusOperation,
    compute_calibration_grid_params,
)
from ui.bridge import QtEngineBridge
from ui.widgets.stage_calibration_inspector import StageCalibrationInspectorWindow


class StageCalibrationTabWidget(QScrollArea):
    """Tab 5: 3-Stage Stage Calibration (Stage 1: Z Focus, Stage 2: Pattern Grid, Stage 3: Optical Scan)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self._active_op: Optional[object] = None
        self._inspector_window: Optional[StageCalibrationInspectorWindow] = None

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 1. Instructions
        info_box = QGroupBox("Stage Calibration Procedure (3 Stages)")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>3-Stage Coordinate & Focal Surface Calibration</b><br><br>"
            "<b>Stage 1 (Z Focus Calibration):</b><br>"
            "• Multi-point autofocus sweep across the wafer surface.<br>"
            "• Maps wafer focal plane into a height map and enables stage Z-leveling.<br><br>"
            "<b>Stage 2 (Patterning):</b><br>"
            "• Measures optical transform (Projector -> Camera) and corrects alignment.<br>"
            "• Uses calibrated Stage 1 Z heights to expose small ArUco tags in UV.<br>"
            "<i>After Stage 2 completes, develop the wafer and remount.</i><br><br>"
            "<b>Stage 3 (Optical Scan & Global Reconstruction):</b><br>"
            "• Jog stage until at least one tag is visible in camera view.<br>"
            "• Estimates wafer pose, scans tag clusters, and reconstructs coordinates via Least Squares."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        main_layout.addWidget(info_box)

        default_cal_file = (
            self.engine.machine_config.stage_calibration.calibration_file_path
            if hasattr(self.engine, "machine_config")
            else "stage_calibration.json"
        )

        # 2. Stage 1 Group
        s1_box = QGroupBox("Stage 1: Z Focus Calibration")
        s1_layout = QVBoxLayout(s1_box)
        s1_layout.setSpacing(6)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Calibration Range:"))
        self.spin_cal_range = QDoubleSpinBox()
        self.spin_cal_range.setRange(500.0, 100000.0)
        self.spin_cal_range.setValue(10000.0)
        self.spin_cal_range.setSingleStep(1000.0)
        self.spin_cal_range.setSuffix(" µm")
        self.spin_cal_range.valueChanged.connect(self._update_computed_grid_label)
        range_row.addWidget(self.spin_cal_range, 1)
        s1_layout.addLayout(range_row)

        self.lbl_grid_info = QLabel("Computed Grid: Calculating...")
        self.lbl_grid_info.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        s1_layout.addWidget(self.lbl_grid_info)

        s1_layout.addWidget(QLabel("Stage 1 Output JSON:"))
        s1_file_row = QHBoxLayout()
        self.txt_s1_file = QLineEdit(os.path.abspath(default_cal_file))
        s1_file_row.addWidget(self.txt_s1_file, 1)
        self.btn_browse_s1 = QPushButton("Browse...")
        self.btn_browse_s1.clicked.connect(self._on_browse_s1_clicked)
        s1_file_row.addWidget(self.btn_browse_s1)
        s1_layout.addLayout(s1_file_row)

        self.btn_start_s1 = QPushButton("Run Stage 1 (Z Focus Calibration)")
        self.btn_start_s1.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_s1.clicked.connect(self._on_start_s1_clicked)
        s1_layout.addWidget(self.btn_start_s1)

        main_layout.addWidget(s1_box)

        # 3. Stage 2 Group
        s2_box = QGroupBox("Stage 2: Pattern Grid (Uses Stage 1 Z Heights)")
        s2_layout = QVBoxLayout(s2_box)
        s2_layout.setSpacing(6)

        s2_layout.addWidget(QLabel("Stage 1 Input JSON:"))
        s2_in_row = QHBoxLayout()
        self.txt_s2_input = QLineEdit(os.path.abspath(default_cal_file))
        s2_in_row.addWidget(self.txt_s2_input, 1)
        self.btn_browse_s2_input = QPushButton("Browse...")
        self.btn_browse_s2_input.clicked.connect(self._on_browse_s2_input_clicked)
        s2_in_row.addWidget(self.btn_browse_s2_input)
        s2_layout.addLayout(s2_in_row)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure Time:"))
        self.spin_exposure_time = QDoubleSpinBox()
        self.spin_exposure_time.setRange(0.1, 120.0)
        default_exp_s = (
            self.engine.machine_config.exposure.default_exposure_time_ms / 1000.0
            if hasattr(self.engine, "machine_config")
            else 8.0
        )
        self.spin_exposure_time.setValue(default_exp_s)
        self.spin_exposure_time.setSingleStep(1.0)
        self.spin_exposure_time.setSuffix(" s")
        exp_row.addWidget(self.spin_exposure_time, 1)
        s2_layout.addLayout(exp_row)

        self.chk_s2_autofocus = QCheckBox("Perform autofocus on every node")
        self.chk_s2_autofocus.setChecked(False)
        self.chk_s2_autofocus.setToolTip(
            "Initial autofocus is always performed at the start. "
            "Check this to also run autofocus at every individual grid point (slower)."
        )
        s2_layout.addWidget(self.chk_s2_autofocus)

        s2_range_row = QHBoxLayout()
        s2_range_row.addWidget(QLabel("Pattern Range (0 for full):"))
        self.spin_s2_range = QDoubleSpinBox()
        self.spin_s2_range.setRange(0.0, 100000.0)
        self.spin_s2_range.setValue(0.0)
        self.spin_s2_range.setSingleStep(1000.0)
        self.spin_s2_range.setSuffix(" µm")
        self.spin_s2_range.setSpecialValueText("Full (from Stage 1)")
        self.spin_s2_range.valueChanged.connect(self._update_s2_grid_label)
        s2_range_row.addWidget(self.spin_s2_range, 1)
        s2_layout.addLayout(s2_range_row)

        self.lbl_s2_grid_info = QLabel("Computed Grid: Full (from Stage 1)")
        self.lbl_s2_grid_info.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        s2_layout.addWidget(self.lbl_s2_grid_info)

        self.btn_start_s2 = QPushButton("Run Stage 2 (Expose ArUco Grid)")
        self.btn_start_s2.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_s2.clicked.connect(self._on_start_s2_clicked)
        s2_layout.addWidget(self.btn_start_s2)

        main_layout.addWidget(s2_box)

        # 4. Stage 3 Group
        s3_box = QGroupBox("Stage 3: Spiral Image Scan")
        s3_layout = QVBoxLayout(s3_box)
        s3_layout.setSpacing(6)

        s3_layout.addWidget(QLabel("Stage 1 Input JSON (for calibrated Z-leveling):"))
        s3_in_row = QHBoxLayout()
        self.txt_s3_input = QLineEdit(os.path.abspath(default_cal_file))
        s3_in_row.addWidget(self.txt_s3_input, 1)
        self.btn_browse_s3_input = QPushButton("Browse...")
        self.btn_browse_s3_input.clicked.connect(self._on_browse_s3_input_clicked)
        s3_in_row.addWidget(self.btn_browse_s3_input)
        s3_layout.addLayout(s3_in_row)

        s3_layout.addWidget(QLabel("Output Folder for Scanned Photos:"))
        s3_out_row = QHBoxLayout()
        default_scan_dir = os.path.abspath("scanned_calibration_photos")
        self.txt_s3_output_dir = QLineEdit(default_scan_dir)
        s3_out_row.addWidget(self.txt_s3_output_dir, 1)
        self.btn_browse_s3_output_dir = QPushButton("Browse...")
        self.btn_browse_s3_output_dir.clicked.connect(self._on_browse_s3_output_dir_clicked)
        s3_out_row.addWidget(self.btn_browse_s3_output_dir)
        s3_layout.addLayout(s3_out_row)

        s3_range_row = QHBoxLayout()
        s3_range_row.addWidget(QLabel("Scan Range (0 for full):"))
        self.spin_s3_range = QDoubleSpinBox()
        self.spin_s3_range.setRange(0.0, 100000.0)
        self.spin_s3_range.setValue(0.0)
        self.spin_s3_range.setSingleStep(1000.0)
        self.spin_s3_range.setSuffix(" µm")
        self.spin_s3_range.setSpecialValueText("Full (from Stage 1/2)")
        self.spin_s3_range.valueChanged.connect(self._update_s3_grid_label)
        s3_range_row.addWidget(self.spin_s3_range, 1)
        s3_layout.addLayout(s3_range_row)

        self.lbl_s3_grid_info = QLabel("Scan Grid: Full (from Stage 1/2)")
        self.lbl_s3_grid_info.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        s3_layout.addWidget(self.lbl_s3_grid_info)

        self.chk_s3_autofocus = QCheckBox("Perform autofocus before every photo taken")
        self.chk_s3_autofocus.setChecked(False)
        self.chk_s3_autofocus.setToolTip(
            "Check this to run autofocus at every spiral scan position instead of relying purely on Stage 1 Z-leveling."
        )
        s3_layout.addWidget(self.chk_s3_autofocus)

        self.btn_start_s3 = QPushButton("Run Stage 3 (Spiral Image Scan)")
        self.btn_start_s3.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_s3.clicked.connect(self._on_start_s3_clicked)
        s3_layout.addWidget(self.btn_start_s3)

        main_layout.addWidget(s3_box)

        # 5. Status & Controls
        status_box = QGroupBox("Status & Controls")
        status_layout = QVBoxLayout(status_box)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_status)

        self.btn_abort = QPushButton("Abort Operation")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        status_layout.addWidget(self.btn_abort)

        main_layout.addWidget(status_box)

        # 6. Standalone Inspector & Reconstruction Tool
        inspector_box = QGroupBox("Reconstruction & Analysis")
        inspector_layout = QVBoxLayout(inspector_box)
        inspector_desc = QLabel(
            "Open the standalone tool to inspect scanned photos, tune ArUco edge detection parameters with live preview, "
            "perform least-squares coordinate reconstruction, and export the final stage calibration."
        )
        inspector_desc.setWordWrap(True)
        inspector_layout.addWidget(inspector_desc)

        self.btn_open_inspector = QPushButton("Open Calibration Inspector & Reconstruction Tool...")
        self.btn_open_inspector.setStyleSheet("font-weight: bold; font-size: 13px; padding: 10px; background-color: #2b5c8f;")
        self.btn_open_inspector.clicked.connect(self._on_open_inspector_clicked)
        inspector_layout.addWidget(self.btn_open_inspector)

        main_layout.addWidget(inspector_box)
        main_layout.addStretch(1)

        self.setWidget(container)

        self.bridge.operation_progress.connect(self._on_progress)
        self.bridge.operation_finished.connect(self._on_operation_finished)
        self.bridge.machine_config_changed.connect(lambda *_: self._update_computed_grid_label())

        self._update_computed_grid_label()

    def _update_computed_grid_label(self):
        cal_range = self.spin_cal_range.value()
        pixel_scale = (
            self.engine.machine_config.stage_calibration.camera_pixel_size
            if hasattr(self.engine, "machine_config")
            else 1.0
        )

        cam_w, cam_h = 1920, 1080
        if self.engine.camera is not None:
            frame = self.engine.camera.get_latest_frame()
            if frame is not None:
                cam_h, cam_w = frame.shape[:2]

        n, d = compute_calibration_grid_params(cal_range, pixel_scale, (cam_w, cam_h))
        fov_um = min(cam_w, cam_h) * pixel_scale
        self.lbl_grid_info.setText(
            f"Computed Grid: {n} x {n} ({n*n} tags) | Step: {d:.1f} µm (FOV: {fov_um:.1f} µm)"
        )

    def _get_step_distance_for_path(self, json_path: str, fallback_range: float) -> float:
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                d = data.get("step_distance")
                if d and float(d) > 0:
                    return float(d)
            except Exception:
                pass
        pixel_scale = (
            self.engine.machine_config.stage_calibration.camera_pixel_size
            if hasattr(self.engine, "machine_config")
            else 1.0
        )
        cam_w, cam_h = 1920, 1080
        if self.engine.camera is not None:
            frame = self.engine.camera.get_latest_frame()
            if frame is not None:
                cam_h, cam_w = frame.shape[:2]
        _, d = compute_calibration_grid_params(max(1000.0, fallback_range), pixel_scale, (cam_w, cam_h))
        return d

    def _update_s2_grid_label(self):
        val = self.spin_s2_range.value()
        if val <= 0.0:
            self.lbl_s2_grid_info.setText("Computed Grid: Full (from Stage 1)")
            return
        step_dist = self._get_step_distance_for_path(self.txt_s2_input.text().strip(), val)
        n = max(1, int(round(val / step_dist)) + 1)
        self.lbl_s2_grid_info.setText(f"Computed Grid: {n} x {n} ({n*n} tags) | Step: {step_dist:.1f} µm")

    def _update_s3_grid_label(self):
        val = self.spin_s3_range.value()
        if val <= 0.0:
            self.lbl_s3_grid_info.setText("Scan Grid: Full (from Stage 1/2)")
            return
        step_dist = self._get_step_distance_for_path(self.txt_s3_input.text().strip(), val)
        n = max(1, int(round(val / step_dist)) + 1)
        self.lbl_s3_grid_info.setText(f"Scan Grid: {n} x {n} ({n*n} tags) | Step: {step_dist:.1f} µm")

    def _on_browse_s1_clicked(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Stage 1 Output JSON", self.txt_s1_file.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_s1_file.setText(path)

    def _on_browse_s2_input_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stage 1 Input JSON", self.txt_s2_input.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_s2_input.setText(path)
            self._update_s2_grid_label()

    def _on_browse_s3_input_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stage Calibration Input JSON", self.txt_s3_input.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_s3_input.setText(path)
            self._update_s3_grid_label()

    def _on_browse_s3_output_dir_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Scanned Photos", self.txt_s3_output_dir.text()
        )
        if dir_path:
            self.txt_s3_output_dir.setText(dir_path)

    def _on_open_inspector_clicked(self):
        pixel_scale = (
            self.engine.machine_config.stage_calibration.camera_pixel_size
            if hasattr(self.engine, "machine_config")
            else 1.0
        )
        images_dir = self.txt_s3_output_dir.text().strip()
        stage1_path = self.txt_s3_input.text().strip() or self.txt_s1_file.text().strip()

        if self._inspector_window is None:
            self._inspector_window = StageCalibrationInspectorWindow(
                initial_images_dir=images_dir,
                initial_stage1_json=stage1_path,
                camera_pixel_size=pixel_scale,
                engine=self.engine,
            )
        else:
            if images_dir and os.path.isdir(images_dir):
                self._inspector_window.txt_images_dir.setText(images_dir)
                self._inspector_window._load_images_directory(images_dir)
            if stage1_path and os.path.exists(stage1_path):
                self._inspector_window.txt_stage1_json.setText(stage1_path)

        self._inspector_window.show()
        self._inspector_window.raise_()
        self._inspector_window.activateWindow()

    def _on_start_s1_clicked(self):
        pixel_scale = (
            self.engine.machine_config.stage_calibration.camera_pixel_size
            if hasattr(self.engine, "machine_config")
            else 1.0
        )
        op = StageCalibrationZFocusOperation(
            calibration_range=self.spin_cal_range.value(),
            camera_pixel_size=pixel_scale,
            output_json_path=self.txt_s1_file.text().strip(),
            apply_to_stage=True,
        )
        self._active_op = op
        self.lbl_status.setText("Status: Starting Stage 1 (Z Focus)...")
        self.btn_abort.setEnabled(True)
        self.bridge.start_operation(op)

    def _on_start_s2_clicked(self):
        s2_range = self.spin_s2_range.value()
        op = StageCalibrationPatternOperation(
            input_json_path=self.txt_s2_input.text().strip(),
            exposure_time=self.spin_exposure_time.value(),
            autofocus=self.chk_s2_autofocus.isChecked(),
            pattern_range=s2_range if s2_range > 0 else None,
        )
        self._active_op = op
        self.lbl_status.setText("Status: Starting Stage 2 (Patterning)...")
        self.btn_abort.setEnabled(True)
        self.bridge.start_operation(op)

    def _on_start_s3_clicked(self):
        pixel_scale = (
            self.engine.machine_config.stage_calibration.camera_pixel_size
            if hasattr(self.engine, "machine_config")
            else 1.0
        )
        s3_range = self.spin_s3_range.value()
        output_dir = self.txt_s3_output_dir.text().strip()
        if not output_dir:
            self.lbl_status.setText("Status: Error - Output folder for scanned photos is required.")
            return

        op = StageCalibrationScanOperation(
            output_dir=output_dir,
            input_json_path=self.txt_s3_input.text().strip(),
            camera_pixel_size=pixel_scale,
            autofocus=self.chk_s3_autofocus.isChecked(),
            scan_range=s3_range if s3_range > 0 else None,
        )
        self._active_op = op
        self.lbl_status.setText("Status: Starting Stage 3 (Spiral Image Scan)...")
        self.btn_abort.setEnabled(True)
        self.bridge.start_operation(op)

    def _on_abort_clicked(self):
        if self._active_op is not None and hasattr(self._active_op, "abort"):
            self._active_op.abort()
        self.bridge.abort_operation()

    def _on_progress(self, pct: float, msg: str):
        if self._active_op is not None:
            self.lbl_status.setText(f"Status: {msg}")

    def _on_operation_finished(self, op_or_name=None, err=None):
        if self._active_op is not None:
            self.btn_abort.setEnabled(False)
            if err:
                self.lbl_status.setText(f"Status: Error - {err}")
            else:
                self.lbl_status.setText("Status: Operation completed successfully!")
            self._active_op = None

    def update_lock_state(self, is_busy: bool):
        self.btn_start_s1.setEnabled(not is_busy)
        self.btn_start_s2.setEnabled(not is_busy)
        self.btn_start_s3.setEnabled(not is_busy)
        self.spin_cal_range.setEnabled(not is_busy)
        self.spin_exposure_time.setEnabled(not is_busy)
        self.txt_s1_file.setEnabled(not is_busy)
        self.btn_browse_s1.setEnabled(not is_busy)
        self.txt_s2_input.setEnabled(not is_busy)
        self.btn_browse_s2_input.setEnabled(not is_busy)
        self.txt_s3_input.setEnabled(not is_busy)
        self.btn_browse_s3_input.setEnabled(not is_busy)
        self.txt_s3_output_dir.setEnabled(not is_busy)
        self.btn_browse_s3_output_dir.setEnabled(not is_busy)
        self.chk_s2_autofocus.setEnabled(not is_busy)
        self.chk_s3_autofocus.setEnabled(not is_busy)
        self.spin_s2_range.setEnabled(not is_busy)
        self.spin_s3_range.setEnabled(not is_busy)

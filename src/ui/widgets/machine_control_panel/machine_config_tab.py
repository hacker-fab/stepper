from __future__ import annotations

import os
from typing import Any, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from core.events import Event
from core.machine_config import (
    AutofocusMachineConfig,
    AutofocusStageConfig,
    MachineConfig,
    OpticsMachineConfig,
)
from ui.bridge import QtEngineBridge


class MachineConfigTabWidget(QScrollArea):
    """Dedicated tab for global machine calibration values, autofocus stages, and defaults."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setWidgetResizable(True)
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ---------------------------------------------------------------------
        # 1. Optics Configuration
        # ---------------------------------------------------------------------
        optics_box = QGroupBox("Optics Configuration")
        optics_layout = QFormLayout(optics_box)
        optics_layout.setContentsMargins(8, 8, 8, 8)
        optics_layout.setSpacing(6)

        self.spin_uv_offset = QDoubleSpinBox()
        self.spin_uv_offset.setRange(-1000.0, 1000.0)
        self.spin_uv_offset.setDecimals(2)
        self.spin_uv_offset.setSingleStep(0.5)
        self.spin_uv_offset.setSuffix(" µm")
        self.spin_uv_offset.valueChanged.connect(self._on_ui_changed)
        optics_layout.addRow("UV-Red Z Offset:", self.spin_uv_offset)

        self.spin_settle_delay = QDoubleSpinBox()
        self.spin_settle_delay.setRange(0.0, 10.0)
        self.spin_settle_delay.setDecimals(2)
        self.spin_settle_delay.setSingleStep(0.05)
        self.spin_settle_delay.setSuffix(" s")
        self.spin_settle_delay.valueChanged.connect(self._on_ui_changed)
        optics_layout.addRow("Settle Delay:", self.spin_settle_delay)

        self.combo_camera_flip = QComboBox()
        self.combo_camera_flip.addItems(["none", "horizontal", "vertical", "rotate_180"])
        self.combo_camera_flip.currentTextChanged.connect(self._on_ui_changed)
        optics_layout.addRow("Camera Flip Mode:", self.combo_camera_flip)

        self.spin_projector_pixel_size = QDoubleSpinBox()
        self.spin_projector_pixel_size.setRange(0.001, 100.0)
        self.spin_projector_pixel_size.setDecimals(4)
        self.spin_projector_pixel_size.setSingleStep(0.05)
        self.spin_projector_pixel_size.setSuffix(" µm/px")
        self.spin_projector_pixel_size.valueChanged.connect(self._on_ui_changed)
        optics_layout.addRow("Projector Pixel Size:", self.spin_projector_pixel_size)

        self.lbl_proj_cam_status = QLabel("Status: Uncalibrated (Default Scaling)")
        self.lbl_proj_cam_status.setStyleSheet("color: #888888; font-size: 11px;")
        optics_layout.addRow("Proj→Cam Transform:", self.lbl_proj_cam_status)

        main_layout.addWidget(optics_box)

        # ---------------------------------------------------------------------
        # 2. Autofocus Progression Stages
        # ---------------------------------------------------------------------
        af_box = QGroupBox("Autofocus Progression Stages")
        af_layout = QVBoxLayout(af_box)
        af_layout.setContentsMargins(8, 8, 8, 8)
        af_layout.setSpacing(6)

        self.table_af_stages = QTableWidget(0, 3)
        self.table_af_stages.setHorizontalHeaderLabels([
            "Grid Dimension (NxN)",
            "Sweep Range (±µm)",
            "Convergence Threshold (µm)",
        ])
        self.table_af_stages.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_af_stages.setFixedHeight(120)
        af_layout.addWidget(self.table_af_stages)

        btn_row_af = QHBoxLayout()
        self.btn_add_stage = QPushButton("+ Add Stage")
        self.btn_add_stage.clicked.connect(self._on_add_stage_clicked)
        btn_row_af.addWidget(self.btn_add_stage)

        self.btn_remove_stage = QPushButton("- Remove Stage")
        self.btn_remove_stage.clicked.connect(self._on_remove_stage_clicked)
        btn_row_af.addWidget(self.btn_remove_stage)
        btn_row_af.addStretch()
        af_layout.addLayout(btn_row_af)

        main_layout.addWidget(af_box)

        # ---------------------------------------------------------------------
        # 3. Exposure Defaults
        # ---------------------------------------------------------------------
        exp_box = QGroupBox("Exposure Defaults")
        exp_layout = QFormLayout(exp_box)
        exp_layout.setContentsMargins(8, 8, 8, 8)
        exp_layout.setSpacing(6)

        self.spin_default_exposure = QDoubleSpinBox()
        self.spin_default_exposure.setRange(1.0, 120000.0)
        self.spin_default_exposure.setDecimals(0)
        self.spin_default_exposure.setSingleStep(500.0)
        self.spin_default_exposure.setSuffix(" ms")
        self.spin_default_exposure.valueChanged.connect(self._on_ui_changed)
        exp_layout.addRow("Default Exposure Time:", self.spin_default_exposure)

        main_layout.addWidget(exp_box)

        # ---------------------------------------------------------------------
        # 4. Stage Coordinate Calibration
        # ---------------------------------------------------------------------
        stage_cal_box = QGroupBox("Stage Coordinate Calibration")
        stage_cal_layout = QFormLayout(stage_cal_box)
        stage_cal_layout.setContentsMargins(8, 8, 8, 8)
        stage_cal_layout.setSpacing(6)

        file_row = QHBoxLayout()
        self.txt_cal_file = QLineEdit()
        self.txt_cal_file.textChanged.connect(self._on_ui_changed)
        self.btn_browse_cal = QPushButton("Browse...")
        self.btn_browse_cal.clicked.connect(self._on_browse_cal_clicked)
        file_row.addWidget(self.txt_cal_file)
        file_row.addWidget(self.btn_browse_cal)
        stage_cal_layout.addRow("Calibration File:", file_row)

        self.lbl_cal_status = QLabel("Status: Raw (No Calibration)")
        self.lbl_cal_status.setStyleSheet("color: #888888; font-size: 11px;")
        stage_cal_layout.addRow("Active State:", self.lbl_cal_status)

        self.spin_camera_pixel_size = QDoubleSpinBox()
        self.spin_camera_pixel_size.setRange(0.001, 100.0)
        self.spin_camera_pixel_size.setDecimals(3)
        self.spin_camera_pixel_size.setSingleStep(0.05)
        self.spin_camera_pixel_size.setSuffix(" µm/px")
        self.spin_camera_pixel_size.valueChanged.connect(self._on_ui_changed)
        stage_cal_layout.addRow("Camera Pixel Size:", self.spin_camera_pixel_size)

        self.spin_stage_rotation = QDoubleSpinBox()
        self.spin_stage_rotation.setRange(-180.0, 180.0)
        self.spin_stage_rotation.setDecimals(3)
        self.spin_stage_rotation.setSingleStep(0.1)
        self.spin_stage_rotation.setSuffix(" °")
        self.spin_stage_rotation.valueChanged.connect(self._on_ui_changed)
        stage_cal_layout.addRow("Additional Stage Rotation:", self.spin_stage_rotation)

        self.spin_stage_skew = QDoubleSpinBox()
        self.spin_stage_skew.setRange(-1.0, 1.0)
        self.spin_stage_skew.setDecimals(5)
        self.spin_stage_skew.setSingleStep(0.0001)
        self.spin_stage_skew.setSuffix(" dy/dx")
        self.spin_stage_skew.valueChanged.connect(self._on_ui_changed)
        stage_cal_layout.addRow("Additional Stage Skew:", self.spin_stage_skew)

        self.spin_stage_scale_x = QDoubleSpinBox()
        self.spin_stage_scale_x.setRange(0.1, 10.0)
        self.spin_stage_scale_x.setDecimals(4)
        self.spin_stage_scale_x.setSingleStep(0.001)
        self.spin_stage_scale_x.setValue(1.0)
        self.spin_stage_scale_x.valueChanged.connect(self._on_ui_changed)
        stage_cal_layout.addRow("Stage Motion Scale X:", self.spin_stage_scale_x)

        self.spin_stage_scale_y = QDoubleSpinBox()
        self.spin_stage_scale_y.setRange(0.1, 10.0)
        self.spin_stage_scale_y.setDecimals(4)
        self.spin_stage_scale_y.setSingleStep(0.001)
        self.spin_stage_scale_y.setValue(1.0)
        self.spin_stage_scale_y.valueChanged.connect(self._on_ui_changed)
        stage_cal_layout.addRow("Stage Motion Scale Y:", self.spin_stage_scale_y)

        cal_btn_row = QHBoxLayout()
        self.btn_apply_stage_cal = QPushButton("Load / Apply Calibration")
        self.btn_apply_stage_cal.clicked.connect(self._on_apply_stage_cal_clicked)
        cal_btn_row.addWidget(self.btn_apply_stage_cal)

        self.btn_clear_stage_cal = QPushButton("Disable Calibration")
        self.btn_clear_stage_cal.clicked.connect(self._on_clear_stage_cal_clicked)
        cal_btn_row.addWidget(self.btn_clear_stage_cal)
        cal_btn_row.addStretch()
        stage_cal_layout.addRow("", cal_btn_row)

        main_layout.addWidget(stage_cal_box)

        # ---------------------------------------------------------------------
        # 5. Config Actions (Save / Load / Reload)
        # ---------------------------------------------------------------------
        actions_box = QGroupBox("Configuration Persistence")
        actions_layout = QHBoxLayout(actions_box)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(10)

        self.btn_save_config = QPushButton("Save Machine Config")
        self.btn_save_config.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_save_config.clicked.connect(self._on_save_config_clicked)
        actions_layout.addWidget(self.btn_save_config)

        self.btn_load_config = QPushButton("Load Config File...")
        self.btn_load_config.clicked.connect(self._on_load_config_clicked)
        actions_layout.addWidget(self.btn_load_config)

        self.btn_reload_defaults = QPushButton("Reload Defaults")
        self.btn_reload_defaults.clicked.connect(self._on_reload_defaults_clicked)
        actions_layout.addWidget(self.btn_reload_defaults)

        main_layout.addWidget(actions_box)
        main_layout.addStretch()

        self.setWidget(container)

        # Populate controls from engine's machine config
        self._syncing = False
        self.sync_from_config(self.engine.machine_config)

        # Connect bridge change signal
        self.bridge.machine_config_changed.connect(self.sync_from_config)

    def sync_from_config(self, config: MachineConfig):
        """Populates all UI controls from a MachineConfig instance."""
        self._syncing = True
        try:
            opt = config.optics
            self.spin_uv_offset.setValue(opt.uv_z_offset)
            self.spin_settle_delay.setValue(opt.settle_delay)
            idx_flip = self.combo_camera_flip.findText(str(opt.camera_flip_mode))
            if idx_flip >= 0:
                self.combo_camera_flip.setCurrentIndex(idx_flip)
            self.spin_projector_pixel_size.setValue(opt.projector_pixel_size)
            if opt.proj_to_cam_matrix is not None:
                self.lbl_proj_cam_status.setText("Active: Calibrated (3×3 Homography)")
                self.lbl_proj_cam_status.setStyleSheet("color: #55ff55; font-weight: bold; font-size: 11px;")
            else:
                self.lbl_proj_cam_status.setText("Status: Uncalibrated (Default Scaling)")
                self.lbl_proj_cam_status.setStyleSheet("color: #888888; font-size: 11px;")

            exp = config.exposure
            self.spin_default_exposure.setValue(exp.default_exposure_time_ms)

            sc = config.stage_calibration
            self.txt_cal_file.setText(sc.calibration_file_path)
            self.spin_camera_pixel_size.setValue(sc.camera_pixel_size)
            self.spin_stage_rotation.setValue(sc.additional_stage_rotation_deg)
            self.spin_stage_skew.setValue(sc.additional_stage_skew)
            self.spin_stage_scale_x.setValue(sc.additional_stage_scale_x)
            self.spin_stage_scale_y.setValue(sc.additional_stage_scale_y)

            # Populate stages table
            self._populate_stages_table(config.autofocus.stages)

            # Update stage calibration status label
            self._refresh_calibration_status()
        finally:
            self._syncing = False

    def _populate_stages_table(self, stages: List[AutofocusStageConfig]):
        self.table_af_stages.blockSignals(True)
        self.table_af_stages.setRowCount(len(stages))
        for r, s in enumerate(stages):
            item_grid = QTableWidgetItem(str(s.grid_n))
            item_grid.setTextAlignment(Qt.AlignCenter)
            item_range = QTableWidgetItem(f"{s.sweep_range:.1f}")
            item_range.setTextAlignment(Qt.AlignCenter)
            item_thresh = QTableWidgetItem(f"{s.threshold:.2f}")
            item_thresh.setTextAlignment(Qt.AlignCenter)

            self.table_af_stages.setItem(r, 0, item_grid)
            self.table_af_stages.setItem(r, 1, item_range)
            self.table_af_stages.setItem(r, 2, item_thresh)
        self.table_af_stages.blockSignals(False)

    def _collect_stages_from_table(self) -> List[AutofocusStageConfig]:
        stages = []
        for r in range(self.table_af_stages.rowCount()):
            try:
                g_item = self.table_af_stages.item(r, 0)
                r_item = self.table_af_stages.item(r, 1)
                t_item = self.table_af_stages.item(r, 2)
                g = int(g_item.text()) if g_item else 2
                sr = float(r_item.text()) if r_item else 250.0
                th = float(t_item.text()) if t_item else 5.0
                stages.append(AutofocusStageConfig(grid_n=g, sweep_range=sr, threshold=th))
            except Exception:
                pass
        return stages or [
            AutofocusStageConfig(grid_n=2, sweep_range=250.0, threshold=5.0),
            AutofocusStageConfig(grid_n=4, sweep_range=10.0, threshold=0.5),
        ]

    def _refresh_calibration_status(self):
        cal = self.engine.stage.get_calibration() if self.engine.stage else None
        if cal is not None and cal.enabled:
            cal_file = self.txt_cal_file.text().strip()
            name = f" ({os.path.basename(cal_file)})" if cal_file else ""
            cal_type = "Full XY + Z" if cal.has_xy_calibration else "Z-Leveling Only"
            self.lbl_cal_status.setText(f"Active: {cal_type}{name}")
            self.lbl_cal_status.setStyleSheet("color: #55ff55; font-weight: bold; font-size: 11px;")
            self.btn_clear_stage_cal.setEnabled(True)
        else:
            self.lbl_cal_status.setText("Status: Raw (No Calibration)")
            self.lbl_cal_status.setStyleSheet("color: #888888; font-size: 11px;")
            self.btn_clear_stage_cal.setEnabled(False)

    def _on_ui_changed(self):
        if self._syncing:
            return
        cfg = self.engine.machine_config
        cfg.optics.uv_z_offset = self.spin_uv_offset.value()
        cfg.optics.settle_delay = self.spin_settle_delay.value()
        cfg.optics.camera_flip_mode = self.combo_camera_flip.currentText()
        cfg.optics.projector_pixel_size = self.spin_projector_pixel_size.value()
        if self.engine.camera is not None:
            self.engine.camera.set_flip_mode(cfg.optics.camera_flip_mode)

        cfg.autofocus.stages = self._collect_stages_from_table()

        cfg.exposure.default_exposure_time_ms = self.spin_default_exposure.value()

        cfg.stage_calibration.calibration_file_path = self.txt_cal_file.text().strip()
        cfg.stage_calibration.camera_pixel_size = self.spin_camera_pixel_size.value()
        cfg.stage_calibration.additional_stage_rotation_deg = self.spin_stage_rotation.value()
        cfg.stage_calibration.additional_stage_skew = self.spin_stage_skew.value()
        cfg.stage_calibration.additional_stage_scale_x = self.spin_stage_scale_x.value()
        cfg.stage_calibration.additional_stage_scale_y = self.spin_stage_scale_y.value()
        if self.engine.stage is not None:
            cal = self.engine.stage.get_calibration()
            if cal is not None:
                cal.set_additional_rotation(cfg.stage_calibration.additional_stage_rotation_deg)
                cal.set_additional_skew(cfg.stage_calibration.additional_stage_skew)
                cal.set_additional_scale(
                    cfg.stage_calibration.additional_stage_scale_x,
                    cfg.stage_calibration.additional_stage_scale_y,
                )
        if self.engine.project is not None:
            self.engine.project.invalidate_all_tile_caches()

    def _on_add_stage_clicked(self):
        stages = self._collect_stages_from_table()
        last = stages[-1] if stages else AutofocusStageConfig()
        new_grid = min(16, last.grid_n * 2)
        new_range = max(1.0, last.sweep_range / 5.0)
        new_thresh = max(0.1, last.threshold / 2.0)
        stages.append(AutofocusStageConfig(grid_n=new_grid, sweep_range=new_range, threshold=new_thresh))
        self.engine.machine_config.autofocus.stages = stages
        self._populate_stages_table(stages)

    def _on_remove_stage_clicked(self):
        stages = self._collect_stages_from_table()
        if len(stages) > 1:
            stages.pop()
            self.engine.machine_config.autofocus.stages = stages
            self._populate_stages_table(stages)

    def _on_browse_cal_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Stage Calibration JSON",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self.txt_cal_file.setText(path)
            self._on_ui_changed()

    def _on_apply_stage_cal_clicked(self):
        path = self.txt_cal_file.text().strip()
        if not path or not os.path.exists(path):
            self.lbl_cal_status.setText(f"File not found: {path}")
            self.lbl_cal_status.setStyleSheet("color: #ff5555; font-size: 11px;")
            return

        try:
            from stage_control.calibration import StageCalibration
            cal = StageCalibration.load_from_json(path)
            rot_deg = self.spin_stage_rotation.value()
            if abs(rot_deg) > 1e-6:
                cal.set_additional_rotation(rot_deg)
            skew = self.spin_stage_skew.value()
            if abs(skew) > 1e-9:
                cal.set_additional_skew(skew)
            scale_x = self.spin_stage_scale_x.value()
            scale_y = self.spin_stage_scale_y.value()
            if abs(scale_x - 1.0) > 1e-7 or abs(scale_y - 1.0) > 1e-7:
                cal.set_additional_scale(scale_x, scale_y)
            if self.engine.stage is not None:
                self.engine.stage.set_calibration(cal)
            self._refresh_calibration_status()
        except Exception as e:
            self.lbl_cal_status.setText(f"Error loading calibration: {e}")
            self.lbl_cal_status.setStyleSheet("color: #ff5555; font-size: 11px;")

    def _on_clear_stage_cal_clicked(self):
        if self.engine.stage is not None:
            self.engine.stage.set_calibration(None)
        self._refresh_calibration_status()

    def _on_save_config_clicked(self):
        self._on_ui_changed()
        self.engine.machine_config.autofocus.stages = self._collect_stages_from_table()
        self.engine.save_machine_config("machine_config.toml")
        self.bridge.status_message.emit("Machine configuration saved to machine_config.toml")

    def _on_load_config_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Machine Configuration",
            "",
            "TOML files (*.toml);;All files (*.*)",
        )
        if path:
            try:
                self.engine.load_machine_config(path)
                self.bridge.status_message.emit(f"Machine configuration loaded from {os.path.basename(path)}")
            except Exception as e:
                self.bridge.status_message.emit(f"Error loading configuration: {e}")

    def _on_reload_defaults_clicked(self):
        try:
            self.engine.load_machine_config("machine_config.toml")
            self.bridge.status_message.emit("Reloaded configuration from machine_config.toml")
        except Exception as e:
            self.bridge.status_message.emit(f"Error reloading defaults: {e}")

    def update_lock_state(self, is_busy: bool):
        """Locks editable controls while an operation is in progress."""
        for widget in [
            self.spin_uv_offset,
            self.spin_settle_delay,
            self.combo_camera_flip,
            self.spin_projector_pixel_size,
            self.table_af_stages,
            self.btn_add_stage,
            self.btn_remove_stage,
            self.spin_default_exposure,
            self.txt_cal_file,
            self.btn_browse_cal,
            self.spin_camera_pixel_size,
            self.spin_stage_rotation,
            self.spin_stage_skew,
            self.spin_stage_scale_x,
            self.spin_stage_scale_y,
            self.btn_apply_stage_cal,
            self.btn_clear_stage_cal,
            self.btn_save_config,
            self.btn_load_config,
            self.btn_reload_defaults,
        ]:
            widget.setEnabled(not is_busy)

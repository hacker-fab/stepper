import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.edge_artag import EdgeDetectorParameters
from lib.stage_calibration_reconstruction.detector import detect_markers_in_frame
from lib.stage_calibration_reconstruction.pipeline import (
    ReconstructionDiagnostics,
    ReconstructionPipeline,
)
from stage_control.calibration import StageCalibration
from ui.widgets.stage_calibration_inspector.distortion_view import DistortionView
from ui.widgets.stage_calibration_inspector.preview_viewport import PreviewViewport


class PipelineWorker(QThread):
    """Background worker thread for running the full ReconstructionPipeline."""

    progress_updated = Signal(float, str)
    finished_reconstruction = Signal(object, object, str)  # (cal, diag, error_msg)

    def __init__(
        self,
        images_dir: str,
        stage1_json: str,
        pixel_size: float,
        params: EdgeDetectorParameters,
        complementary_images_dir: Optional[str] = None,
        additional_stage_rotation_deg: float = 0.0,
        additional_stage_skew: float = 0.0,
        additional_stage_scale_x: float = 1.0,
        additional_stage_scale_y: float = 1.0,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.stage1_json = stage1_json
        self.pixel_size = pixel_size
        self.params = params
        self.complementary_images_dir = complementary_images_dir
        self.additional_stage_rotation_deg = float(additional_stage_rotation_deg)
        self.additional_stage_skew = float(additional_stage_skew)
        self.additional_stage_scale_x = float(additional_stage_scale_x)
        self.additional_stage_scale_y = float(additional_stage_scale_y)

    def run(self):
        pipeline = ReconstructionPipeline(
            images_dir=self.images_dir,
            stage1_json_path=self.stage1_json,
            camera_pixel_size=self.pixel_size,
            detector_params=self.params,
            complementary_images_dir=self.complementary_images_dir,
            additional_stage_rotation_deg=self.additional_stage_rotation_deg,
            additional_stage_skew=self.additional_stage_skew,
            additional_stage_scale_x=self.additional_stage_scale_x,
            additional_stage_scale_y=self.additional_stage_scale_y,
        )
        cal, diag, err = pipeline.run(
            progress_cb=lambda pct, msg: self.progress_updated.emit(pct, msg)
        )
        self.finished_reconstruction.emit(cal, diag, err if err else "")


class StageCalibrationInspectorWindow(QMainWindow):
    """Standalone Top-Level Window for Stage Calibration Inspection, Live ArUco Tuning, and Least-Squares Reconstruction."""

    def __init__(
        self,
        initial_images_dir: str = "",
        initial_stage1_json: str = "",
        camera_pixel_size: float = 1.0,
        engine: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Stage Calibration Inspector & Reconstruction Tool")
        self.resize(1320, 850)
        self.setMinimumSize(960, 600)

        # State
        self._current_frame: Optional[np.ndarray] = None
        self._current_file_path: Optional[str] = None
        self._last_calibration: Optional[StageCalibration] = None
        self._worker_thread: Optional[PipelineWorker] = None
        self._current_debug_images: Dict[str, np.ndarray] = {}
        self._last_detected_corners: List[np.ndarray] = []
        self._last_detected_ids: Optional[np.ndarray] = None

        # Build UI
        self._init_ui(initial_images_dir, initial_stage1_json, camera_pixel_size)

        # Auto-load if valid directory provided
        if initial_images_dir and os.path.isdir(initial_images_dir):
            self._load_images_directory(initial_images_dir)

    def _init_ui(self, initial_images_dir: str, initial_stage1_json: str, camera_pixel_size: float):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. Top Configuration Bar
        top_bar = QGroupBox("Inputs & Configuration")
        top_layout = QVBoxLayout(top_bar)
        top_layout.setSpacing(4)

        # Row 1: Image folder
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scanned Photos Folder:"))
        self.txt_images_dir = QLineEdit(initial_images_dir)
        row1.addWidget(self.txt_images_dir, 1)
        self.btn_browse_dir = QPushButton("Browse...")
        self.btn_browse_dir.clicked.connect(self._on_browse_images_dir)
        row1.addWidget(self.btn_browse_dir)
        self.btn_reload_dir = QPushButton("Load / Reload")
        self.btn_reload_dir.clicked.connect(lambda: self._load_images_directory(self.txt_images_dir.text().strip()))
        row1.addWidget(self.btn_reload_dir)
        top_layout.addLayout(row1)

        # Row 1b: Complementary images folder (optional)
        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("Complementary Photos (Optional):"))
        self.txt_comp_images_dir = QLineEdit()
        self.txt_comp_images_dir.setPlaceholderText("Optional directory with additional photos of the same chip...")
        row1b.addWidget(self.txt_comp_images_dir, 1)
        self.btn_browse_comp_dir = QPushButton("Browse...")
        self.btn_browse_comp_dir.clicked.connect(self._on_browse_comp_images_dir)
        row1b.addWidget(self.btn_browse_comp_dir)
        self.btn_clear_comp_dir = QPushButton("Clear")
        self.btn_clear_comp_dir.clicked.connect(lambda: self.txt_comp_images_dir.clear())
        row1b.addWidget(self.btn_clear_comp_dir)
        top_layout.addLayout(row1b)

        # Row 2: Stage 1 JSON + Camera pixel size + Flip H
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Stage 1 Calibration JSON:"))
        self.txt_stage1_json = QLineEdit(initial_stage1_json)
        row2.addWidget(self.txt_stage1_json, 1)
        self.btn_browse_s1 = QPushButton("Browse...")
        self.btn_browse_s1.clicked.connect(self._on_browse_stage1_json)
        row2.addWidget(self.btn_browse_s1)

        row2.addWidget(QLabel("Pixel Size (µm/px):"))
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.01, 1000.0)
        self.spin_pixel_size.setValue(float(camera_pixel_size))
        self.spin_pixel_size.setSingleStep(0.1)
        self.spin_pixel_size.setDecimals(4)
        row2.addWidget(self.spin_pixel_size)

        init_rot = 0.0
        init_skew = 0.0
        init_scale_x = 1.0
        init_scale_y = 1.0
        if self.engine is not None and getattr(self.engine, "machine_config", None) is not None:
            init_rot = getattr(self.engine.machine_config.stage_calibration, "additional_stage_rotation_deg", 0.0)
            init_skew = getattr(self.engine.machine_config.stage_calibration, "additional_stage_skew", 0.0)
            init_scale_x = getattr(self.engine.machine_config.stage_calibration, "additional_stage_scale_x", 1.0)
            init_scale_y = getattr(self.engine.machine_config.stage_calibration, "additional_stage_scale_y", 1.0)

        row2.addWidget(QLabel("Stage Rot (°):"))
        self.spin_stage_rotation = QDoubleSpinBox()
        self.spin_stage_rotation.setRange(-180.0, 180.0)
        self.spin_stage_rotation.setValue(float(init_rot))
        self.spin_stage_rotation.setDecimals(3)
        self.spin_stage_rotation.setSingleStep(0.1)
        row2.addWidget(self.spin_stage_rotation)

        row2.addWidget(QLabel("Skew (dy/dx):"))
        self.spin_stage_skew = QDoubleSpinBox()
        self.spin_stage_skew.setRange(-1.0, 1.0)
        self.spin_stage_skew.setValue(float(init_skew))
        self.spin_stage_skew.setDecimals(5)
        self.spin_stage_skew.setSingleStep(0.0001)
        row2.addWidget(self.spin_stage_skew)

        row2.addWidget(QLabel("Scale X:"))
        self.spin_stage_scale_x = QDoubleSpinBox()
        self.spin_stage_scale_x.setRange(0.1, 10.0)
        self.spin_stage_scale_x.setValue(float(init_scale_x))
        self.spin_stage_scale_x.setDecimals(4)
        self.spin_stage_scale_x.setSingleStep(0.001)
        row2.addWidget(self.spin_stage_scale_x)

        row2.addWidget(QLabel("Scale Y:"))
        self.spin_stage_scale_y = QDoubleSpinBox()
        self.spin_stage_scale_y.setRange(0.1, 10.0)
        self.spin_stage_scale_y.setValue(float(init_scale_y))
        self.spin_stage_scale_y.setDecimals(4)
        self.spin_stage_scale_y.setSingleStep(0.001)
        row2.addWidget(self.spin_stage_scale_y)

        top_layout.addLayout(row2)
        main_layout.addWidget(top_bar)

        # 2. Central Splitter (Left: List + Tuning, Right: Tabs View)
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel (List of Images + Parameters + Reprocess button)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel("Scanned Images:"))
        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self._on_image_selection_changed)
        left_layout.addWidget(self.list_images, 2)

        # Detection Parameters Group
        param_box = QGroupBox("ArUco Detector Tuning (Live Preview)")
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFrameShape(QScrollArea.NoFrame)
        param_widget = QWidget()
        param_form = QFormLayout(param_widget)
        param_form.setContentsMargins(4, 4, 4, 4)
        param_form.setSpacing(4)

        # Fine adaptive threshold block size
        self.spin_fine_block = QSpinBox()
        self.spin_fine_block.setRange(3, 101)
        self.spin_fine_block.setSingleStep(2)
        self.spin_fine_block.setValue(15)
        self.spin_fine_block.valueChanged.connect(self._on_block_size_changed)
        param_form.addRow("Fine Block Size:", self.spin_fine_block)

        # Fine C
        self.spin_fine_c = QDoubleSpinBox()
        self.spin_fine_c.setRange(0.0, 50.0)
        self.spin_fine_c.setValue(7.0)
        self.spin_fine_c.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Fine C:", self.spin_fine_c)

        # Wide adaptive threshold block size
        self.spin_wide_block = QSpinBox()
        self.spin_wide_block.setRange(5, 201)
        self.spin_wide_block.setSingleStep(2)
        self.spin_wide_block.setValue(61)
        self.spin_wide_block.valueChanged.connect(self._on_block_size_changed)
        param_form.addRow("Wide Block Size:", self.spin_wide_block)

        # Wide C
        self.spin_wide_c = QDoubleSpinBox()
        self.spin_wide_c.setRange(0.0, 50.0)
        self.spin_wide_c.setValue(14.0)
        self.spin_wide_c.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Wide C:", self.spin_wide_c)

        # Min / Max marker area
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(10.0, 500000.0)
        self.spin_min_area.setValue(1500.0)
        self.spin_min_area.setSingleStep(200.0)
        self.spin_min_area.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Min Marker Area:", self.spin_min_area)

        self.spin_max_area = QDoubleSpinBox()
        self.spin_max_area.setRange(500.0, 2000000.0)
        self.spin_max_area.setValue(200000.0)
        self.spin_max_area.setSingleStep(5000.0)
        self.spin_max_area.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Max Marker Area:", self.spin_max_area)

        # Max center edge density
        self.spin_center_density = QDoubleSpinBox()
        self.spin_center_density.setRange(0.01, 1.0)
        self.spin_center_density.setValue(0.16)
        self.spin_center_density.setSingleStep(0.02)
        self.spin_center_density.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Center Edge Max:", self.spin_center_density)

        # Marker border scale
        self.spin_border_scale = QDoubleSpinBox()
        self.spin_border_scale.setRange(0.50, 1.50)
        self.spin_border_scale.setDecimals(3)
        self.spin_border_scale.setSingleStep(0.01)
        self.spin_border_scale.setValue(1.0)
        self.spin_border_scale.valueChanged.connect(self._on_parameter_changed)
        param_form.addRow("Marker Border Scale:", self.spin_border_scale)

        # Subpixel refinement
        self.chk_subpixel = QCheckBox("Subpixel Corner Refine")
        self.chk_subpixel.setChecked(True)
        self.chk_subpixel.toggled.connect(self._on_parameter_changed)
        param_form.addRow(self.chk_subpixel)

        # Reset button
        btn_reset_params = QPushButton("Reset Defaults")
        btn_reset_params.clicked.connect(self._reset_detector_parameters)
        param_form.addRow(btn_reset_params)

        param_scroll.setWidget(param_widget)
        param_box_layout = QVBoxLayout(param_box)
        param_box_layout.setContentsMargins(4, 4, 4, 4)
        param_box_layout.addWidget(param_scroll)
        left_layout.addWidget(param_box, 3)

        # Reprocess All button
        self.btn_reprocess_all = QPushButton("Reprocess All Images")
        self.btn_reprocess_all.setStyleSheet("font-weight: bold; padding: 8px; font-size: 13px; background-color: #2b5c8f;")
        self.btn_reprocess_all.clicked.connect(self._on_reprocess_all_clicked)
        left_layout.addWidget(self.btn_reprocess_all)

        splitter.addWidget(left_panel)

        # Right Panel (Tabs)
        self.tabs = QTabWidget()

        # Tab 1: Image Viewport & Controls
        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        tab1_layout.setContentsMargins(4, 4, 4, 4)
        tab1_layout.setSpacing(4)

        tab1_bar = QHBoxLayout()
        self.lbl_image_info = QLabel("No image selected")
        self.lbl_image_info.setStyleSheet("font-weight: bold; color: #4da6ff;")
        tab1_bar.addWidget(self.lbl_image_info, 1)

        tab1_bar.addWidget(QLabel("View:"))
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItem("Input Image", "input")
        self.combo_view_mode.addItem("Grayscale Image", "gray")
        self.combo_view_mode.addItem("Detected Squares & Inner Grid", "candidate_quads")
        self.combo_view_mode.addItem("Fine Adaptive Threshold", "e_fine")
        self.combo_view_mode.addItem("Wide Adaptive Threshold", "e_wide")
        self.combo_view_mode.addItem("After Fine & Wide Thresholding", "edges_decode")
        self.combo_view_mode.currentIndexChanged.connect(self._on_view_mode_changed)
        tab1_bar.addWidget(self.combo_view_mode)

        self.chk_overlay_markers = QCheckBox("Overlay Markers")
        self.chk_overlay_markers.setChecked(True)
        self.chk_overlay_markers.toggled.connect(self._on_view_mode_changed)
        tab1_bar.addWidget(self.chk_overlay_markers)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(lambda: self.viewport.fit_to_window())
        tab1_bar.addWidget(btn_fit)

        btn_1x = QPushButton("1:1")
        btn_1x.clicked.connect(lambda: self.viewport.reset_zoom())
        tab1_bar.addWidget(btn_1x)

        btn_zin = QPushButton("+")
        btn_zin.clicked.connect(lambda: self.viewport.zoom_in())
        tab1_bar.addWidget(btn_zin)

        btn_zout = QPushButton("-")
        btn_zout.clicked.connect(lambda: self.viewport.zoom_out())
        tab1_bar.addWidget(btn_zout)
        tab1_layout.addLayout(tab1_bar)

        self.viewport = PreviewViewport()
        tab1_layout.addWidget(self.viewport, 1)
        self.tabs.addTab(tab1_widget, "Image Preview & Detection")

        # Tab 2: Reconstruction & Distortion Map
        tab2_widget = QWidget()
        tab2_layout = QVBoxLayout(tab2_widget)
        tab2_layout.setContentsMargins(4, 4, 4, 4)
        tab2_layout.setSpacing(4)

        # Diagnostics metrics
        diag_box = QGroupBox("Reconstruction Diagnostics")
        diag_layout = QHBoxLayout(diag_box)
        diag_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_diag_tags = QLabel("Unique Tags: 0 / 0")
        self.lbl_diag_constraints = QLabel("Pairwise Constraints: 0")
        self.lbl_diag_rot = QLabel("Wafer Rotation: 0.00°")
        self.lbl_diag_residual = QLabel("Residual Mean: 0.00 µm (Max: 0.00 µm)")

        diag_layout.addWidget(self.lbl_diag_tags)
        diag_layout.addWidget(self.lbl_diag_constraints)
        diag_layout.addWidget(self.lbl_diag_rot)
        diag_layout.addWidget(self.lbl_diag_residual)
        tab2_layout.addWidget(diag_box)

        # Distortion Map Controls
        dist_ctrl_bar = QHBoxLayout()
        dist_ctrl_bar.addWidget(QLabel("Vector Magnification:"))
        self.spin_vector_mag = QDoubleSpinBox()
        self.spin_vector_mag.setRange(1.0, 500.0)
        self.spin_vector_mag.setValue(20.0)
        self.spin_vector_mag.setSuffix("x")
        self.spin_vector_mag.valueChanged.connect(lambda val: self.distortion_view.set_magnification(val))
        dist_ctrl_bar.addWidget(self.spin_vector_mag)

        btn_fit_dist = QPushButton("Fit Grid View")
        btn_fit_dist.clicked.connect(lambda: self.distortion_view.fit_view())
        dist_ctrl_bar.addWidget(btn_fit_dist)
        dist_ctrl_bar.addStretch(1)
        tab2_layout.addLayout(dist_ctrl_bar)

        self.distortion_view = DistortionView()
        tab2_layout.addWidget(self.distortion_view, 1)
        self.tabs.addTab(tab2_widget, "Reconstruction & Distortion Map")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, 1)

        # 3. Bottom Status Bar & Action Controls
        bottom_bar = QGroupBox("Export & Application")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 6, 6, 6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        bottom_layout.addWidget(self.progress_bar, 1)

        self.lbl_status = QLabel("Ready")
        bottom_layout.addWidget(self.lbl_status, 2)

        self.btn_save_cal = QPushButton("Save Calibration JSON...")
        self.btn_save_cal.setEnabled(False)
        self.btn_save_cal.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_save_cal.clicked.connect(self._on_save_calibration_clicked)
        bottom_layout.addWidget(self.btn_save_cal)

        self.btn_apply_stage = QPushButton("Apply to Stage Controller")
        self.btn_apply_stage.setEnabled(False)
        self.btn_apply_stage.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_apply_stage.clicked.connect(self._on_apply_stage_clicked)
        bottom_layout.addWidget(self.btn_apply_stage)

        main_layout.addWidget(bottom_bar)
        self.setCentralWidget(main_widget)

    def _get_current_detector_parameters(self) -> EdgeDetectorParameters:
        fine_b = int(self.spin_fine_block.value())
        if fine_b % 2 == 0:
            fine_b += 1
        wide_b = int(self.spin_wide_block.value())
        if wide_b % 2 == 0:
            wide_b += 1

        params = EdgeDetectorParameters()
        params.adaptive_thresh_fine_block_size = fine_b
        params.adaptive_thresh_fine_C = float(self.spin_fine_c.value())
        params.adaptive_thresh_wide_block_size = wide_b
        params.adaptive_thresh_wide_C = float(self.spin_wide_c.value())
        params.min_marker_area = float(self.spin_min_area.value())
        params.max_marker_area = float(self.spin_max_area.value())
        params.max_center_edge_density = float(self.spin_center_density.value())
        params.marker_border_scale = float(self.spin_border_scale.value())
        params.enable_subpixel_refinement = bool(self.chk_subpixel.isChecked())
        return params

    def _reset_detector_parameters(self):
        defs = EdgeDetectorParameters()
        self.spin_fine_block.setValue(defs.adaptive_thresh_fine_block_size)
        self.spin_fine_c.setValue(defs.adaptive_thresh_fine_C)
        self.spin_wide_block.setValue(defs.adaptive_thresh_wide_block_size)
        self.spin_wide_c.setValue(defs.adaptive_thresh_wide_C)
        self.spin_min_area.setValue(defs.min_marker_area)
        self.spin_max_area.setValue(defs.max_marker_area)
        self.spin_center_density.setValue(defs.max_center_edge_density)
        self.spin_border_scale.setValue(defs.marker_border_scale)
        self.chk_subpixel.setChecked(defs.enable_subpixel_refinement)

    def _on_block_size_changed(self, val: int):
        # Force block size to odd numbers
        sender = self.sender()
        if isinstance(sender, QSpinBox) and val % 2 == 0:
            sender.setValue(val + 1)
            return
        self._on_parameter_changed()

    def _on_parameter_changed(self):
        """Live updates detection ONLY on the currently previewed image."""
        if self._current_frame is None:
            return

        params = self._get_current_detector_parameters()
        cnt, corners, ids = detect_markers_in_frame(
            self._current_frame,
            parameters=params,
            debug_output=self._current_debug_images,
        )

        self._last_detected_corners = corners
        self._last_detected_ids = ids

        id_list = ids.flatten().tolist() if ids is not None else []
        fname = os.path.basename(self._current_file_path or "")
        self.lbl_image_info.setText(
            f"Image: {fname} | Detected: {cnt} tags | IDs: {id_list}"
        )
        self._update_viewport_display(fit_view=False)

    def _on_view_mode_changed(self):
        """Switches displayed intermediate debug image."""
        self._update_viewport_display(fit_view=False)

    def _update_viewport_display(self, fit_view: bool = False):
        if self._current_frame is None:
            self.viewport.display_frame(None)
            return

        mode_key = self.combo_view_mode.currentData()
        display_img = self._current_debug_images.get(mode_key)

        if display_img is None:
            display_img = self._current_frame

        overlay = bool(self.chk_overlay_markers.isChecked())
        self.viewport.display_frame(
            display_img,
            corners=self._last_detected_corners,
            ids=self._last_detected_ids,
            overlay_markers=overlay,
            fit_view=fit_view,
        )

    def _on_browse_images_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Scanned Photos Folder", self.txt_images_dir.text()
        )
        if path:
            self.txt_images_dir.setText(path)
            self._load_images_directory(path)

    def _on_browse_comp_images_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Complementary Photos Folder (Optional)",
            self.txt_comp_images_dir.text() or self.txt_images_dir.text(),
        )
        if path:
            self.txt_comp_images_dir.setText(path)

    def _on_browse_stage1_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stage 1 Calibration JSON", self.txt_stage1_json.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_stage1_json.setText(path)

    def _load_images_directory(self, folder_path: str):
        self.list_images.clear()
        if not os.path.isdir(folder_path):
            self.lbl_status.setText(f"Folder not found: {folder_path}")
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f.lower())[1] in valid_exts
        ])

        if not files:
            self.lbl_status.setText(f"No image files found in {folder_path}")
            return

        for f in files:
            item = QListWidgetItem(f)
            item.setData(Qt.UserRole, os.path.join(folder_path, f))
            self.list_images.addItem(item)

        self.lbl_status.setText(f"Loaded {len(files)} images from folder.")
        if self.list_images.count() > 0:
            self.list_images.setCurrentRow(0)

    def _on_image_selection_changed(self, row: int):
        if row < 0 or row >= self.list_images.count():
            return
        item = self.list_images.item(row)
        file_path = item.data(Qt.UserRole)
        if not file_path or not os.path.exists(file_path):
            return

        self._current_file_path = file_path
        img = cv2.imread(file_path)
        self._current_frame = img

        if img is None:
            self.lbl_image_info.setText(f"Failed to read image: {os.path.basename(file_path)}")
            self.viewport.display_frame(None)
            return

        params = self._get_current_detector_parameters()
        cnt, corners, ids = detect_markers_in_frame(
            img,
            parameters=params,
            debug_output=self._current_debug_images,
        )

        self._last_detected_corners = corners
        self._last_detected_ids = ids

        id_list = ids.flatten().tolist() if ids is not None else []
        self.lbl_image_info.setText(
            f"Image: {os.path.basename(file_path)} ({img.shape[1]}x{img.shape[0]}) | Detected: {cnt} tags | IDs: {id_list}"
        )
        self._update_viewport_display(fit_view=True)

    def _on_reprocess_all_clicked(self):
        images_dir = self.txt_images_dir.text().strip()
        comp_dir = self.txt_comp_images_dir.text().strip() or None
        stage1_json = self.txt_stage1_json.text().strip()

        if not images_dir or not os.path.isdir(images_dir):
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid Scanned Photos folder.")
            return

        if comp_dir and not os.path.isdir(comp_dir):
            QMessageBox.warning(self, "Invalid Directory", "Complementary folder does not exist or is not a directory.")
            return

        if not stage1_json or not os.path.exists(stage1_json):
            QMessageBox.warning(self, "Missing Stage 1 File", "Please select a valid Stage 1 JSON file.")
            return

        params = self._get_current_detector_parameters()
        pixel_size = float(self.spin_pixel_size.value())

        self.btn_reprocess_all.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Processing all images via ReconstructionPipeline...")

        self._worker_thread = PipelineWorker(
            images_dir=images_dir,
            stage1_json=stage1_json,
            pixel_size=pixel_size,
            params=params,
            complementary_images_dir=comp_dir,
            additional_stage_rotation_deg=float(self.spin_stage_rotation.value()),
            additional_stage_skew=float(self.spin_stage_skew.value()),
            additional_stage_scale_x=float(self.spin_stage_scale_x.value()),
            additional_stage_scale_y=float(self.spin_stage_scale_y.value()),
        )
        self._worker_thread.progress_updated.connect(self._on_worker_progress)
        self._worker_thread.finished_reconstruction.connect(self._on_worker_finished)
        self._worker_thread.start()

    def _on_worker_progress(self, pct: float, msg: str):
        self.progress_bar.setValue(int(pct * 100))
        self.lbl_status.setText(msg)

    def _on_worker_finished(self, cal: Optional[StageCalibration], diag: ReconstructionDiagnostics, err: str):
        self.btn_reprocess_all.setEnabled(True)
        self.progress_bar.setVisible(False)

        if err or cal is None:
            self.lbl_status.setText(f"Reconstruction failed: {err}")
            QMessageBox.critical(self, "Reconstruction Failed", f"Failed to reconstruct calibration:\n{err}")
            return

        self._last_calibration = cal
        self.btn_save_cal.setEnabled(True)
        if self.engine is not None and getattr(self.engine, "stage", None) is not None:
            self.btn_apply_stage.setEnabled(True)

        # Update List items with detection results
        for idx in range(self.list_images.count()):
            item = self.list_images.item(idx)
            fname = os.path.basename(item.data(Qt.UserRole))
            if fname in diag.per_image_results:
                info = diag.per_image_results[fname]
                cnt = info.get("detected_count", 0)
                ids = info.get("ids", [])
                item.setText(f"{fname}  [{cnt} tags: {ids}]")

        # Update diagnostics display
        self.lbl_diag_tags.setText(f"Unique Tags: {diag.unique_tags_found} / {diag.nominal_tags_count}")
        if diag.complementary_pairwise_constraints > 0:
            self.lbl_diag_constraints.setText(
                f"Pairwise Constraints: {diag.total_pairwise_constraints} ({diag.primary_pairwise_constraints} primary + {diag.complementary_pairwise_constraints} comp, rot: {diag.complementary_relative_rotation_deg:.2f}°)"
            )
        else:
            self.lbl_diag_constraints.setText(f"Pairwise Constraints: {diag.total_pairwise_constraints}")
        self.lbl_diag_rot.setText(f"Wafer Rotation: {diag.estimated_rotation_deg:.3f}°")
        self.lbl_diag_residual.setText(f"Residual Mean: {diag.mean_residual_um:.2f} µm (Max: {diag.max_residual_um:.2f} µm)")

        # Update Distortion view
        self.distortion_view.set_calibration(cal, detected_markers=diag.detected_marker_ids)
        self.tabs.setCurrentIndex(1)  # Switch to distortion map tab
        self.lbl_status.setText("Reconstruction completed successfully! Calibration ready to export.")

    def _on_save_calibration_clicked(self):
        if self._last_calibration is None:
            return
        default_name = os.path.join(
            os.path.dirname(self.txt_stage1_json.text().strip() or "."),
            "stage_calibration_reconstructed.json",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstructed Stage Calibration JSON", default_name, "JSON Files (*.json)"
        )
        if path:
            try:
                self._last_calibration.save_to_json(path)
                self.lbl_status.setText(f"Saved calibration JSON to {path}")
                QMessageBox.information(self, "Saved", f"Calibration saved successfully to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save calibration file:\n{e}")

    def _on_apply_stage_clicked(self):
        if self._last_calibration is None:
            return
        if self.engine is not None and getattr(self.engine, "stage", None) is not None:
            try:
                self.engine.stage.set_calibration(self._last_calibration)
                self.lbl_status.setText("Applied reconstructed calibration directly to active Stage Controller.")
                QMessageBox.information(
                    self, "Calibration Applied",
                    "The reconstructed optical stage calibration map has been applied directly to the active Stage Controller."
                )
            except Exception as e:
                QMessageBox.critical(self, "Application Error", f"Failed to apply calibration to stage:\n{e}")


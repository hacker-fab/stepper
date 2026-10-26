import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations.ml_data_collection import (
    MLDataCollectionConfig,
    MLDataCollectionOperation,
    PatternSource,
)
from ui.bridge import QtEngineBridge


class MLDataCollectionTabWidget(QScrollArea):
    """Tab 4: Automated ML data collection with projector calibration and spiral patterning."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self._active_op: Optional[MLDataCollectionOperation] = None

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 1. Info / Instructions
        info_box = QGroupBox("ML Data Collection Procedure")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>Automated ML Dataset Generation</b><br><br>"
            "1. Performs initial autofocus without UV offset.<br>"
            "2. Projects a 2x2 ArUco calibration grid in Red light to compute relative projector-to-camera alignment.<br>"
            "3. Moves in a 2D spiral pattern from the current position.<br>"
            "4. At each step, autofocuses, exposes the selected pattern (generated cross markers or randomly chosen image from folder scaled to full projector resolution) in UV, then returns to Red focus Z under solid Red illumination.<br>"
            "5. Saves the camera capture, transformed ground-truth pattern, and annotations (including marker coordinates if applicable)."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        main_layout.addWidget(info_box)

        # 2. Pattern Source Selection
        src_box = QGroupBox("Pattern Source")
        src_layout = QVBoxLayout(src_box)
        src_layout.setContentsMargins(6, 6, 6, 6)
        src_layout.setSpacing(6)

        self.src_btn_group = QButtonGroup(self)
        src_radio_row = QHBoxLayout()
        self.radio_src_marker = QRadioButton("Generated Cross Markers")
        self.radio_src_marker.setChecked(True)
        self.radio_src_folder = QRadioButton("Image Folder")
        self.src_btn_group.addButton(self.radio_src_marker)
        self.src_btn_group.addButton(self.radio_src_folder)
        src_radio_row.addWidget(self.radio_src_marker)
        src_radio_row.addWidget(self.radio_src_folder)
        src_radio_row.addStretch()
        src_layout.addLayout(src_radio_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Folder:"))
        self.txt_image_folder = QLineEdit()
        self.txt_image_folder.setPlaceholderText("Select folder with images...")
        self.txt_image_folder.setEnabled(False)
        self.btn_browse_image_folder = QPushButton("Browse...")
        self.btn_browse_image_folder.setEnabled(False)
        self.btn_browse_image_folder.clicked.connect(self._on_browse_image_folder_clicked)
        folder_row.addWidget(self.txt_image_folder)
        folder_row.addWidget(self.btn_browse_image_folder)
        src_layout.addLayout(folder_row)

        self.radio_src_marker.toggled.connect(self._on_source_changed)

        main_layout.addWidget(src_box)

        # 3. Marker Configuration
        self.marker_box = QGroupBox("Marker Parameters")
        marker_grid = QGridLayout(self.marker_box)

        # Target Scale (%)
        marker_grid.addWidget(QLabel("Target Scale (%):"), 0, 0)
        self.spin_target_scale = QDoubleSpinBox()
        self.spin_target_scale.setRange(0.5, 50.0)
        self.spin_target_scale.setDecimals(1)
        self.spin_target_scale.setSingleStep(0.5)
        self.spin_target_scale.setValue(8.0)
        self.spin_target_scale.setSuffix(" %")
        marker_grid.addWidget(self.spin_target_scale, 0, 1)

        # Scale Jitter (+/- %)
        marker_grid.addWidget(QLabel("Scale Jitter (± %):"), 1, 0)
        self.spin_scale_jitter = QDoubleSpinBox()
        self.spin_scale_jitter.setRange(0.0, 20.0)
        self.spin_scale_jitter.setDecimals(1)
        self.spin_scale_jitter.setSingleStep(0.5)
        self.spin_scale_jitter.setValue(2.0)
        self.spin_scale_jitter.setSuffix(" %")
        marker_grid.addWidget(self.spin_scale_jitter, 1, 1)

        # Markers Per Pattern
        marker_grid.addWidget(QLabel("Markers Per Pattern:"), 2, 0)
        self.spin_marker_count = QSpinBox()
        self.spin_marker_count.setRange(1, 200)
        self.spin_marker_count.setSingleStep(1)
        self.spin_marker_count.setValue(20)
        marker_grid.addWidget(self.spin_marker_count, 2, 1)

        # Min Thickness
        marker_grid.addWidget(QLabel("Min Thickness:"), 3, 0)
        self.spin_min_thickness = QDoubleSpinBox()
        self.spin_min_thickness.setRange(0.0, 1.0)
        self.spin_min_thickness.setDecimals(2)
        self.spin_min_thickness.setSingleStep(0.05)
        self.spin_min_thickness.setValue(0.2)
        marker_grid.addWidget(self.spin_min_thickness, 3, 1)

        # Max Thickness
        marker_grid.addWidget(QLabel("Max Thickness:"), 4, 0)
        self.spin_max_thickness = QDoubleSpinBox()
        self.spin_max_thickness.setRange(0.0, 1.0)
        self.spin_max_thickness.setDecimals(2)
        self.spin_max_thickness.setSingleStep(0.05)
        self.spin_max_thickness.setValue(1.0)
        marker_grid.addWidget(self.spin_max_thickness, 4, 1)

        main_layout.addWidget(self.marker_box)

        # 4. Patterning & Exposure
        pattern_box = QGroupBox("Patterning & Capture")
        pattern_grid = QGridLayout(pattern_box)

        # Total Patterns
        pattern_grid.addWidget(QLabel("Total Patterns:"), 0, 0)
        self.spin_total_patterns = QSpinBox()
        self.spin_total_patterns.setRange(1, 100)
        self.spin_total_patterns.setSingleStep(1)
        self.spin_total_patterns.setValue(10)
        pattern_grid.addWidget(self.spin_total_patterns, 0, 1)

        # Pattern Gap (Spiral Step)
        pattern_grid.addWidget(QLabel("Pattern Gap:"), 1, 0)
        self.spin_pattern_gap = QDoubleSpinBox()
        self.spin_pattern_gap.setRange(10.0, 50000.0)
        self.spin_pattern_gap.setDecimals(1)
        self.spin_pattern_gap.setSingleStep(100.0)
        self.spin_pattern_gap.setValue(1000.0)
        self.spin_pattern_gap.setSuffix(" µm")
        pattern_grid.addWidget(self.spin_pattern_gap, 1, 1)

        # Min Exposure
        pattern_grid.addWidget(QLabel("Min Exposure:"), 2, 0)
        self.spin_min_exposure = QDoubleSpinBox()
        self.spin_min_exposure.setRange(0.05, 300.0)
        self.spin_min_exposure.setDecimals(2)
        self.spin_min_exposure.setSingleStep(0.5)
        self.spin_min_exposure.setValue(8.0)
        self.spin_min_exposure.setSuffix(" s")
        pattern_grid.addWidget(self.spin_min_exposure, 2, 1)

        # Max Exposure
        pattern_grid.addWidget(QLabel("Max Exposure:"), 3, 0)
        self.spin_max_exposure = QDoubleSpinBox()
        self.spin_max_exposure.setRange(0.05, 300.0)
        self.spin_max_exposure.setDecimals(2)
        self.spin_max_exposure.setSingleStep(0.5)
        self.spin_max_exposure.setValue(12.0)
        self.spin_max_exposure.setSuffix(" s")
        pattern_grid.addWidget(self.spin_max_exposure, 3, 1)

        # Stabilization Delay
        pattern_grid.addWidget(QLabel("Stabilization Delay:"), 4, 0)
        self.spin_stabilization_delay = QDoubleSpinBox()
        self.spin_stabilization_delay.setRange(0.1, 10.0)
        self.spin_stabilization_delay.setDecimals(1)
        self.spin_stabilization_delay.setSingleStep(0.5)
        self.spin_stabilization_delay.setValue(1.0)
        self.spin_stabilization_delay.setSuffix(" s")
        pattern_grid.addWidget(self.spin_stabilization_delay, 4, 1)

        main_layout.addWidget(pattern_box)

        # 5. Output Folder
        output_box = QGroupBox("Dataset Output Folder")
        output_layout = QHBoxLayout(output_box)

        default_dir = os.path.abspath("stepper_captures/ml_data")
        self.txt_save_dir = QLineEdit(default_dir)
        output_layout.addWidget(self.txt_save_dir)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        output_layout.addWidget(self.btn_browse)

        main_layout.addWidget(output_box)

        # 6. Start button & Status
        action_box = QGroupBox("Execution")
        action_layout = QVBoxLayout(action_box)

        self.btn_start = QPushButton("Start ML Data Collection")
        self.btn_start.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start.clicked.connect(self._on_start_clicked)
        action_layout.addWidget(self.btn_start)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #666; font-size: 11px;")
        action_layout.addWidget(self.lbl_status)

        main_layout.addWidget(action_box)
        main_layout.addStretch()

        self.setWidget(container)

    def _on_source_changed(self):
        is_folder = self.radio_src_folder.isChecked()
        self.txt_image_folder.setEnabled(is_folder)
        self.btn_browse_image_folder.setEnabled(is_folder)
        self.marker_box.setEnabled(not is_folder)
        self.spin_target_scale.setEnabled(not is_folder)
        self.spin_scale_jitter.setEnabled(not is_folder)
        self.spin_marker_count.setEnabled(not is_folder)
        self.spin_min_thickness.setEnabled(not is_folder)
        self.spin_max_thickness.setEnabled(not is_folder)

    def _on_browse_image_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Pattern Images Folder",
            self.txt_image_folder.text() or os.getcwd(),
        )
        if folder:
            self.txt_image_folder.setText(folder)

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ML Data Output Folder",
            self.txt_save_dir.text() or os.getcwd(),
        )
        if folder:
            self.txt_save_dir.setText(folder)

    def _on_start_clicked(self):
        save_dir = self.txt_save_dir.text().strip()
        if not save_dir:
            save_dir = os.path.abspath("stepper_captures/ml_data")
            self.txt_save_dir.setText(save_dir)

        is_folder = self.radio_src_folder.isChecked()
        pattern_source = (
            PatternSource.IMAGE_FOLDER.value
            if is_folder
            else PatternSource.GENERATED_MARKER.value
        )
        image_folder = self.txt_image_folder.text().strip() if is_folder else ""

        if is_folder:
            if not image_folder or not os.path.isdir(image_folder):
                self.lbl_status.setText("Status: Error - Please select a valid image folder")
                return

        config = MLDataCollectionConfig(
            pattern_source=pattern_source,
            image_folder=image_folder,
            total_patterns=self.spin_total_patterns.value(),
            pattern_gap=self.spin_pattern_gap.value(),
            target_scale_pct=self.spin_target_scale.value(),
            scale_jitter_pct=self.spin_scale_jitter.value(),
            marker_count=self.spin_marker_count.value(),
            min_thickness=self.spin_min_thickness.value(),
            max_thickness=self.spin_max_thickness.value(),
            min_exposure=self.spin_min_exposure.value(),
            max_exposure=self.spin_max_exposure.value(),
            stabilization_delay=self.spin_stabilization_delay.value(),
            save_directory=save_dir,
            grid_n=2,
            autofocus_config=getattr(self.engine, "autofocus_config", None),
        )

        op = MLDataCollectionOperation(config=config)
        self._active_op = op
        self.lbl_status.setText("Status: Running ML data collection...")
        self.bridge.start_operation(op)

    def on_operation_finished(self, op_or_name=None, err=None):
        from core.operation import Operation

        op = op_or_name if isinstance(op_or_name, Operation) else getattr(self, "_active_op", None)
        if isinstance(op, MLDataCollectionOperation) or op_or_name == "ML Data Collection":
            if err is not None:
                self.lbl_status.setText(f"Status: Failed - {err}")
            else:
                self.lbl_status.setText("Status: Complete")

    def update_lock_state(self, is_busy: bool):
        self.btn_start.setEnabled(not is_busy)
        self.radio_src_marker.setEnabled(not is_busy)
        self.radio_src_folder.setEnabled(not is_busy)

        is_folder = self.radio_src_folder.isChecked()
        self.txt_image_folder.setEnabled(not is_busy and is_folder)
        self.btn_browse_image_folder.setEnabled(not is_busy and is_folder)

        self.marker_box.setEnabled(not is_busy and not is_folder)
        self.spin_target_scale.setEnabled(not is_busy and not is_folder)
        self.spin_scale_jitter.setEnabled(not is_busy and not is_folder)
        self.spin_marker_count.setEnabled(not is_busy and not is_folder)
        self.spin_min_thickness.setEnabled(not is_busy and not is_folder)
        self.spin_max_thickness.setEnabled(not is_busy and not is_folder)

        self.spin_total_patterns.setEnabled(not is_busy)
        self.spin_pattern_gap.setEnabled(not is_busy)
        self.spin_min_exposure.setEnabled(not is_busy)
        self.spin_max_exposure.setEnabled(not is_busy)
        self.spin_stabilization_delay.setEnabled(not is_busy)
        self.txt_save_dir.setEnabled(not is_busy)
        self.btn_browse.setEnabled(not is_busy)

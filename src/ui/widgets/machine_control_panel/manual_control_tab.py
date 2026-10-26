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
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from core.events import ColorMode, ProjectorImageSource
from operations import (
    AlignmentOperation,
    AutofocusOperation,
    HomeOperation,
    JogOperation,
    MaximizeImageSharpnessOperation,
)
from ui.bridge import QtEngineBridge


class ManualControlTabWidget(QScrollArea):
    """Tab 1: Machine motion, manual jog, sharpness, autofocus, and projector control."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.current_jog_step = 100.0  # Default 100 µm

        container = QWidget()
        manual_layout = QVBoxLayout(container)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(8)

        # 1. Coordinate Readout
        pos_box = QGroupBox("Stage Coordinates (µm)")
        pos_layout = QGridLayout(pos_box)
        pos_layout.setContentsMargins(6, 6, 6, 6)

        pos_layout.addWidget(QLabel("<b>X:</b>"), 0, 0)
        self.lbl_pos_x = QLabel("0.0")
        self.lbl_pos_x.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self.lbl_pos_x, 0, 1)

        pos_layout.addWidget(QLabel("<b>Y:</b>"), 0, 2)
        self.lbl_pos_y = QLabel("0.0")
        self.lbl_pos_y.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self.lbl_pos_y, 0, 3)

        pos_layout.addWidget(QLabel("<b>Z:</b>"), 0, 4)
        self.lbl_pos_z = QLabel("0.0")
        self.lbl_pos_z.setStyleSheet("font-family: monospace; font-size: 13px;")
        pos_layout.addWidget(self.lbl_pos_z, 0, 5)

        manual_layout.addWidget(pos_box)

        # 2. Jog Controls
        jog_box = QGroupBox("Jog Stage")
        jog_layout = QVBoxLayout(jog_box)
        jog_layout.setContentsMargins(6, 6, 6, 6)
        jog_layout.setSpacing(6)

        # Step size selector
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self.step_btn_group = QButtonGroup(self)
        for label_text, val in [("1 µm", 1.0), ("10 µm", 10.0), ("100 µm", 100.0), ("1 mm", 1000.0)]:
            rb = QRadioButton(label_text)
            if val == 100.0:
                rb.setChecked(True)
            self.step_btn_group.addButton(rb)
            rb.toggled.connect(lambda checked, v=val: self._on_step_changed(checked, v))
            step_row.addWidget(rb)
        jog_layout.addLayout(step_row)

        # Directional Grid
        grid = QGridLayout()
        grid.setSpacing(4)

        self.btn_y_pos = QPushButton("▲ +Y")
        self.btn_y_pos.clicked.connect(lambda: self._jog(0, self.current_jog_step, 0))
        grid.addWidget(self.btn_y_pos, 0, 1)

        self.btn_x_neg = QPushButton("◀ -X")
        self.btn_x_neg.clicked.connect(lambda: self._jog(-self.current_jog_step, 0, 0))
        grid.addWidget(self.btn_x_neg, 1, 0)

        self.btn_home = QPushButton("⌂ Home")
        self.btn_home.setStyleSheet("font-weight: bold;")
        self.btn_home.clicked.connect(self._on_home_clicked)
        grid.addWidget(self.btn_home, 1, 1)

        self.btn_x_pos = QPushButton("+X ▶")
        self.btn_x_pos.clicked.connect(lambda: self._jog(self.current_jog_step, 0, 0))
        grid.addWidget(self.btn_x_pos, 1, 2)

        self.btn_y_neg = QPushButton("▼ -Y")
        self.btn_y_neg.clicked.connect(lambda: self._jog(0, -self.current_jog_step, 0))
        grid.addWidget(self.btn_y_neg, 2, 1)

        # Z buttons
        self.btn_z_pos = QPushButton("Z+ (Up)")
        self.btn_z_pos.clicked.connect(lambda: self._jog(0, 0, self.current_jog_step))
        grid.addWidget(self.btn_z_pos, 0, 3)

        self.btn_z_neg = QPushButton("Z- (Down)")
        self.btn_z_neg.clicked.connect(lambda: self._jog(0, 0, -self.current_jog_step))
        grid.addWidget(self.btn_z_neg, 2, 3)

        jog_layout.addLayout(grid)
        manual_layout.addWidget(jog_box)

        # 3. Projector Illumination & Image Source Controls
        proj_box = QGroupBox("Projector Control")
        proj_layout = QVBoxLayout(proj_box)
        proj_layout.setContentsMargins(6, 6, 6, 6)
        proj_layout.setSpacing(6)

        # Projector Output Power / Shutter Button
        self.btn_projector_power = QPushButton("Turn Projector Output ON")
        self.btn_projector_power.setCheckable(True)
        self.btn_projector_power.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_projector_power.clicked.connect(self._on_projector_power_clicked)
        proj_layout.addWidget(self.btn_projector_power)

        # Color Mode Radio Buttons
        color_group_box = QGroupBox("Color Mode")
        color_layout = QHBoxLayout(color_group_box)
        color_layout.setContentsMargins(4, 4, 4, 4)
        self.color_btn_group = QButtonGroup(self)

        self.radio_color_red = QRadioButton("Red")
        self.radio_color_uv = QRadioButton("UV")
        if self.engine.projector.color_mode == ColorMode.RED:
            self.radio_color_red.setChecked(True)
        else:
            self.radio_color_uv.setChecked(True)

        self.color_btn_group.addButton(self.radio_color_red)
        self.color_btn_group.addButton(self.radio_color_uv)

        color_layout.addWidget(self.radio_color_red)
        color_layout.addWidget(self.radio_color_uv)
        proj_layout.addWidget(color_group_box)

        self.radio_color_red.toggled.connect(self._on_color_mode_toggled)
        self.radio_color_uv.toggled.connect(self._on_color_mode_toggled)

        # Image Source Radio Buttons
        src_group_box = QGroupBox("Image Source")
        src_layout = QVBoxLayout(src_group_box)
        src_layout.setContentsMargins(4, 4, 4, 4)
        self.src_btn_group = QButtonGroup(self)

        src_radio_row = QHBoxLayout()
        self.radio_src_active = QRadioButton("Active Layer")
        self.radio_src_active.setChecked(True)
        self.radio_src_custom = QRadioButton("Custom File")
        self.radio_src_solid = QRadioButton("Solid")
        self.src_btn_group.addButton(self.radio_src_active)
        self.src_btn_group.addButton(self.radio_src_custom)
        self.src_btn_group.addButton(self.radio_src_solid)

        src_radio_row.addWidget(self.radio_src_active)
        src_radio_row.addWidget(self.radio_src_custom)
        src_radio_row.addWidget(self.radio_src_solid)
        src_layout.addLayout(src_radio_row)

        # Custom file row
        self.custom_file_row = QHBoxLayout()
        self.txt_custom_file = QLineEdit()
        self.txt_custom_file.setPlaceholderText("Select custom image file...")
        self.txt_custom_file.setEnabled(False)
        self.btn_browse_custom = QPushButton("Browse...")
        self.btn_browse_custom.setEnabled(False)
        self.btn_browse_custom.clicked.connect(self._on_browse_custom_clicked)
        self.custom_file_row.addWidget(self.txt_custom_file)
        self.custom_file_row.addWidget(self.btn_browse_custom)
        src_layout.addLayout(self.custom_file_row)

        self.radio_src_active.toggled.connect(self._on_image_source_toggled)
        self.radio_src_custom.toggled.connect(self._on_image_source_toggled)
        self.radio_src_solid.toggled.connect(self._on_image_source_toggled)

        proj_layout.addWidget(src_group_box)

        # Projector Brightness
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.spin_projector_brightness = QDoubleSpinBox()
        self.spin_projector_brightness.setRange(0.0, 1.0)
        self.spin_projector_brightness.setDecimals(2)
        self.spin_projector_brightness.setSingleStep(0.05)
        init_brightness = getattr(self.engine.projector, "brightness", 1.0)
        self.spin_projector_brightness.setValue(init_brightness)
        self.spin_projector_brightness.valueChanged.connect(self._on_brightness_changed)
        brightness_layout.addWidget(self.spin_projector_brightness)
        proj_layout.addLayout(brightness_layout)

        manual_layout.addWidget(proj_box)

        # 4. Maximize Image Sharpness Group Box
        sharp_box = QGroupBox("Maximize Image Sharpness")
        sharp_layout = QVBoxLayout(sharp_box)
        sharp_layout.setContentsMargins(6, 6, 6, 6)
        sharp_layout.setSpacing(6)

        sharp_range_layout = QHBoxLayout()
        sharp_range_layout.addWidget(QLabel("Z Search Range (±µm):"))
        self.spin_sharp_z_range = QDoubleSpinBox()
        self.spin_sharp_z_range.setRange(0.1, 1000.0)
        self.spin_sharp_z_range.setDecimals(1)
        self.spin_sharp_z_range.setSingleStep(1.0)
        self.spin_sharp_z_range.setValue(20.0)
        sharp_range_layout.addWidget(self.spin_sharp_z_range)
        sharp_layout.addLayout(sharp_range_layout)

        self.btn_maximize_sharpness = QPushButton("Maximize Image Sharpness")
        self.btn_maximize_sharpness.clicked.connect(self._on_maximize_sharpness_clicked)
        sharp_layout.addWidget(self.btn_maximize_sharpness)
        manual_layout.addWidget(sharp_box)

        # 5. Autofocus Group Box
        af_box = QGroupBox("Autofocus")
        af_layout = QVBoxLayout(af_box)
        af_layout.setContentsMargins(6, 6, 6, 6)
        af_layout.setSpacing(6)

        af_offset_layout = QHBoxLayout()
        af_offset_layout.addWidget(QLabel("UV-Red Z Offset (µm):"))
        self.spin_uv_offset = QDoubleSpinBox()
        self.spin_uv_offset.setRange(-1000.0, 1000.0)
        self.spin_uv_offset.setDecimals(2)
        self.spin_uv_offset.setSingleStep(1.0)
        init_offset = 0.0
        if hasattr(self.engine, "autofocus_config") and self.engine.autofocus_config is not None:
            init_offset = getattr(self.engine.autofocus_config, "uv_z_offset", 0.0)
        self.spin_uv_offset.setValue(init_offset)
        self.spin_uv_offset.valueChanged.connect(self._on_uv_offset_changed)
        af_offset_layout.addWidget(self.spin_uv_offset)
        af_layout.addLayout(af_offset_layout)

        self.btn_autofocus = QPushButton("Run Autofocus")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        af_layout.addWidget(self.btn_autofocus)
        manual_layout.addWidget(af_box)

        # 6. Alignment Group Box
        align_box = QGroupBox("Alignment")
        align_layout = QVBoxLayout(align_box)
        align_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_align = QPushButton("Align to Marks")
        self.btn_align.clicked.connect(self._on_align_clicked)
        align_layout.addWidget(self.btn_align)
        manual_layout.addWidget(align_box)

        manual_layout.addStretch()

        self.setWidget(container)

    def _on_step_changed(self, checked: bool, val: float):
        if checked:
            self.current_jog_step = val

    def _on_uv_offset_changed(self, val: float):
        if hasattr(self.engine, "autofocus_config") and self.engine.autofocus_config is not None:
            self.engine.autofocus_config.uv_z_offset = val

    def _jog(self, dx: float, dy: float, dz: float):
        op = JogOperation({"x": dx, "y": dy, "z": dz}, relative=True)
        self.bridge.start_operation(op)

    def _on_home_clicked(self):
        op = HomeOperation()
        self.bridge.start_operation(op)

    def _on_maximize_sharpness_clicked(self):
        op = MaximizeImageSharpnessOperation(
            z_range=self.spin_sharp_z_range.value(),
        )
        self.bridge.start_operation(op)

    def _on_autofocus_clicked(self):
        cfg = getattr(self.engine, "autofocus_config", None)
        if cfg is not None:
            cfg.uv_z_offset = self.spin_uv_offset.value()
        op = AutofocusOperation(config=cfg)
        self.bridge.start_operation(op)

    def _on_align_clicked(self):
        op = AlignmentOperation(config=getattr(self.engine, "alignment_config", None))
        self.bridge.start_operation(op)

    def _on_projector_power_clicked(self):
        is_on = self.btn_projector_power.isChecked()
        self.engine.projector.set_on(is_on)

    def _sync_projector_on_off(self, is_on: bool):
        self.btn_projector_power.blockSignals(True)
        self.btn_projector_power.setChecked(is_on)
        if is_on:
            self.btn_projector_power.setText("Turn Projector Output OFF")
            self.btn_projector_power.setStyleSheet("font-weight: bold; padding: 6px; background-color: #4CAF50; color: white;")
        else:
            self.btn_projector_power.setText("Turn Projector Output ON")
            self.btn_projector_power.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_projector_power.blockSignals(False)

    def _on_color_mode_toggled(self):
        if self.radio_color_red.isChecked():
            self.engine.projector.set_color_mode(ColorMode.RED)
        elif self.radio_color_uv.isChecked():
            self.engine.projector.set_color_mode(ColorMode.UV)

    def _sync_color_mode(self, mode: ColorMode):
        self.radio_color_red.blockSignals(True)
        self.radio_color_uv.blockSignals(True)
        if mode == ColorMode.RED:
            self.radio_color_red.setChecked(True)
        elif mode == ColorMode.UV:
            self.radio_color_uv.setChecked(True)
        self.radio_color_red.blockSignals(False)
        self.radio_color_uv.blockSignals(False)

    def _on_image_source_toggled(self):
        is_custom = self.radio_src_custom.isChecked()
        self.txt_custom_file.setEnabled(is_custom)
        self.btn_browse_custom.setEnabled(is_custom)
        if is_custom:
            path = self.txt_custom_file.text().strip() or None
            self.engine.projector.set_image_source(ProjectorImageSource.CUSTOM_FILE, path)
        elif self.radio_src_solid.isChecked():
            self.engine.projector.set_image_source(ProjectorImageSource.SOLID)
        else:
            self.engine.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

    def _on_browse_custom_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.bmp *.tif)")
        if path:
            self.txt_custom_file.setText(path)
            self.engine.projector.set_image_source(ProjectorImageSource.CUSTOM_FILE, path)

    def _sync_image_source(self, src: ProjectorImageSource):
        self.radio_src_active.blockSignals(True)
        self.radio_src_custom.blockSignals(True)
        self.radio_src_solid.blockSignals(True)
        if src == ProjectorImageSource.CUSTOM_FILE:
            self.radio_src_custom.setChecked(True)
            self.txt_custom_file.setEnabled(True)
            self.btn_browse_custom.setEnabled(True)
        elif src == ProjectorImageSource.SOLID:
            self.radio_src_solid.setChecked(True)
            self.txt_custom_file.setEnabled(False)
            self.btn_browse_custom.setEnabled(False)
        elif src == ProjectorImageSource.ACTIVE_LAYER:
            self.radio_src_active.setChecked(True)
            self.txt_custom_file.setEnabled(False)
            self.btn_browse_custom.setEnabled(False)
        elif src == ProjectorImageSource.GENERATED:
            self.src_btn_group.setExclusive(False)
            self.radio_src_active.setChecked(False)
            self.radio_src_custom.setChecked(False)
            self.radio_src_solid.setChecked(False)
            self.src_btn_group.setExclusive(True)
            self.txt_custom_file.setEnabled(False)
            self.btn_browse_custom.setEnabled(False)
        self.radio_src_active.blockSignals(False)
        self.radio_src_custom.blockSignals(False)
        self.radio_src_solid.blockSignals(False)

    def _on_brightness_changed(self, val: float):
        self.engine.projector.set_brightness(val)

    def _sync_brightness(self, val: float):
        self.spin_projector_brightness.blockSignals(True)
        self.spin_projector_brightness.setValue(val)
        self.spin_projector_brightness.blockSignals(False)

    def _on_pos_changed(self, coords: tuple):
        x, y, z = coords
        self.lbl_pos_x.setText(f"{x:.1f}")
        self.lbl_pos_y.setText(f"{y:.1f}")
        self.lbl_pos_z.setText(f"{z:.1f}")

    def update_lock_state(self, is_busy: bool):
        for btn in [
            self.btn_y_pos,
            self.btn_y_neg,
            self.btn_x_pos,
            self.btn_x_neg,
            self.btn_z_pos,
            self.btn_z_neg,
            self.btn_home,
            self.btn_maximize_sharpness,
            self.spin_sharp_z_range,
            self.btn_autofocus,
            self.btn_align,
            self.btn_projector_power,
            self.radio_color_red,
            self.radio_color_uv,
            self.radio_src_active,
            self.radio_src_custom,
            self.radio_src_solid,
            self.spin_projector_brightness,
            self.spin_uv_offset,
        ]:
            btn.setEnabled(not is_busy)

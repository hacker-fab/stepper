from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
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
)
from ui.bridge import QtEngineBridge


class MachineControlPanelWidget(QWidget):
    """Right dock panel: Machine motion control, autofocus, and projector mode switching."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.current_jog_step = 100.0  # Default 100 µm

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

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

        layout.addWidget(pos_box)

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
        self.btn_home.setStyleSheet("font-weight: bold; background-color: #3f3f46;")
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
        layout.addWidget(jog_box)

        # 3. Autofocus & Alignment
        af_box = QGroupBox("Autofocus & Alignment")
        af_layout = QVBoxLayout(af_box)
        af_layout.setContentsMargins(6, 6, 6, 6)

        af_btn_row = QHBoxLayout()
        self.btn_autofocus = QPushButton("Run Autofocus")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        af_btn_row.addWidget(self.btn_autofocus)

        self.btn_align = QPushButton("Align to Marks")
        self.btn_align.clicked.connect(self._on_align_clicked)
        af_btn_row.addWidget(self.btn_align)
        af_layout.addLayout(af_btn_row)

        layout.addWidget(af_box)

        # 4. Projector Illumination & Image Source Controls
        proj_box = QGroupBox("Projector Control")
        proj_layout = QVBoxLayout(proj_box)
        proj_layout.setContentsMargins(6, 6, 6, 6)
        proj_layout.setSpacing(6)

        # Color Mode Radio Buttons
        color_group_box = QGroupBox("Color Mode")
        color_layout = QHBoxLayout(color_group_box)
        color_layout.setContentsMargins(4, 4, 4, 4)
        self.color_btn_group = QButtonGroup(self)

        self.radio_color_disable = QRadioButton("Disable (Black)")
        self.radio_color_disable.setChecked(True)
        self.radio_color_red = QRadioButton("Red")
        self.radio_color_uv = QRadioButton("UV")

        self.color_btn_group.addButton(self.radio_color_disable)
        self.color_btn_group.addButton(self.radio_color_red)
        self.color_btn_group.addButton(self.radio_color_uv)

        color_layout.addWidget(self.radio_color_disable)
        color_layout.addWidget(self.radio_color_red)
        color_layout.addWidget(self.radio_color_uv)
        proj_layout.addWidget(color_group_box)

        self.radio_color_disable.toggled.connect(self._on_color_mode_toggled)
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
        self.src_btn_group.addButton(self.radio_src_active)
        self.src_btn_group.addButton(self.radio_src_custom)
        src_radio_row.addWidget(self.radio_src_active)
        src_radio_row.addWidget(self.radio_src_custom)
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

        proj_layout.addWidget(src_group_box)
        layout.addWidget(proj_box)
        layout.addStretch()

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_pos_changed)
        self.bridge.projector_color_mode_changed.connect(self._sync_color_mode)
        self.bridge.projector_image_source_changed.connect(self._sync_image_source)
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_aborted.connect(lambda *_: self._update_lock_state())
        self._update_lock_state()

    def _on_color_mode_toggled(self):
        if self.radio_color_disable.isChecked():
            self.engine.projector.set_color_mode(ColorMode.DISABLE)
        elif self.radio_color_red.isChecked():
            self.engine.projector.set_color_mode(ColorMode.RED)
        elif self.radio_color_uv.isChecked():
            self.engine.projector.set_color_mode(ColorMode.UV)

    def _on_image_source_toggled(self):
        is_custom = self.radio_src_custom.isChecked()
        self.txt_custom_file.setEnabled(is_custom)
        self.btn_browse_custom.setEnabled(is_custom)
        if is_custom:
            path = self.txt_custom_file.text().strip() or None
            self.engine.projector.set_image_source(ProjectorImageSource.CUSTOM_FILE, path)
        else:
            self.engine.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

    def _on_browse_custom_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.bmp *.tif)")
        if path:
            self.txt_custom_file.setText(path)
            self.engine.projector.set_image_source(ProjectorImageSource.CUSTOM_FILE, path)

    def _sync_color_mode(self, mode: ColorMode):
        self.radio_color_disable.blockSignals(True)
        self.radio_color_red.blockSignals(True)
        self.radio_color_uv.blockSignals(True)
        if mode == ColorMode.DISABLE:
            self.radio_color_disable.setChecked(True)
        elif mode == ColorMode.RED:
            self.radio_color_red.setChecked(True)
        elif mode == ColorMode.UV:
            self.radio_color_uv.setChecked(True)
        self.radio_color_disable.blockSignals(False)
        self.radio_color_red.blockSignals(False)
        self.radio_color_uv.blockSignals(False)

    def _sync_image_source(self, src: ProjectorImageSource):
        self.radio_src_active.blockSignals(True)
        self.radio_src_custom.blockSignals(True)
        if src == ProjectorImageSource.CUSTOM_FILE:
            self.radio_src_custom.setChecked(True)
            self.txt_custom_file.setEnabled(True)
            self.btn_browse_custom.setEnabled(True)
        else:
            self.radio_src_active.setChecked(True)
            self.txt_custom_file.setEnabled(False)
            self.btn_browse_custom.setEnabled(False)
        self.radio_src_active.blockSignals(False)
        self.radio_src_custom.blockSignals(False)

    def _on_step_changed(self, checked: bool, val: float):
        if checked:
            self.current_jog_step = val

    def _jog(self, dx: float, dy: float, dz: float):
        op = JogOperation({"x": dx, "y": dy, "z": dz}, relative=True)
        self.bridge.start_operation(op)

    def _on_home_clicked(self):
        op = HomeOperation()
        self.bridge.start_operation(op)

    def _on_autofocus_clicked(self):
        op = AutofocusOperation(blue_only=(self.engine.projector.color_mode == ColorMode.UV))
        self.bridge.start_operation(op)

    def _on_align_clicked(self):
        op = AlignmentOperation()
        self.bridge.start_operation(op)

    def _on_pos_changed(self, coords: tuple):
        x, y, z = coords
        self.lbl_pos_x.setText(f"{x:.1f}")
        self.lbl_pos_y.setText(f"{y:.1f}")
        self.lbl_pos_z.setText(f"{z:.1f}")

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        for btn in [
            self.btn_y_pos,
            self.btn_y_neg,
            self.btn_x_pos,
            self.btn_x_neg,
            self.btn_z_pos,
            self.btn_z_neg,
            self.btn_home,
            self.btn_autofocus,
            self.btn_align,
            self.radio_color_disable,
            self.radio_color_red,
            self.radio_color_uv,
            self.radio_src_active,
            self.radio_src_custom,
        ]:
            btn.setEnabled(not is_busy)

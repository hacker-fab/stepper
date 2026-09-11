from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from core.events import ShownImage
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
        self.current_jog_step = 100.0  # µm default

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Coordinates Readout Card
        coord_box = QGroupBox("Platform Coordinates")
        coord_layout = QHBoxLayout(coord_box)
        coord_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_x = QLabel("X: 0.000 mm")
        self.lbl_x.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 12px;")
        coord_layout.addWidget(self.lbl_x)

        self.lbl_y = QLabel("Y: 0.000 mm")
        self.lbl_y.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 12px;")
        coord_layout.addWidget(self.lbl_y)

        self.lbl_z = QLabel("Z: 0.000 mm")
        self.lbl_z.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 12px;")
        coord_layout.addWidget(self.lbl_z)
        layout.addWidget(coord_box)

        # 2. Stage Motion Controls
        motion_box = QGroupBox("Platform Motion")
        motion_layout = QVBoxLayout(motion_box)
        motion_layout.setContentsMargins(6, 6, 6, 6)
        motion_layout.setSpacing(4)

        # Step size selectors
        step_row = QHBoxLayout()
        self.step_btn_group = QButtonGroup(self)
        steps = [("10 µm", 10.0), ("100 µm", 100.0), ("1 mm", 1000.0), ("10 mm", 10000.0)]
        for text, val in steps:
            rb = QRadioButton(text)
            if val == 100.0:
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, v=val: self._on_step_changed(checked, v))
            self.step_btn_group.addButton(rb)
            step_row.addWidget(rb)
        motion_layout.addLayout(step_row)

        # D-pad and Z focus grid
        jog_row = QHBoxLayout()

        # XY Grid
        xy_grid = QGridLayout()
        xy_grid.setSpacing(4)

        self.btn_yp = QPushButton("▲ Y+")
        self.btn_yp.clicked.connect(lambda: self._jog(0, self.current_jog_step, 0))
        xy_grid.addWidget(self.btn_yp, 0, 1)

        self.btn_xm = QPushButton("◀ X-")
        self.btn_xm.clicked.connect(lambda: self._jog(-self.current_jog_step, 0, 0))
        xy_grid.addWidget(self.btn_xm, 1, 0)

        self.lbl_xy_center = QLabel("XY")
        self.lbl_xy_center.setAlignment(Qt.AlignCenter)
        self.lbl_xy_center.setStyleSheet("color: #71717a; font-weight: bold;")
        xy_grid.addWidget(self.lbl_xy_center, 1, 1)

        self.btn_xp = QPushButton("X+ ▶")
        self.btn_xp.clicked.connect(lambda: self._jog(self.current_jog_step, 0, 0))
        xy_grid.addWidget(self.btn_xp, 1, 2)

        self.btn_ym = QPushButton("▼ Y-")
        self.btn_ym.clicked.connect(lambda: self._jog(0, -self.current_jog_step, 0))
        xy_grid.addWidget(self.btn_ym, 2, 1)
        jog_row.addLayout(xy_grid)

        # Z Focus controls
        z_col = QVBoxLayout()
        self.btn_zp = QPushButton("▲ +Z (In)")
        self.btn_zp.clicked.connect(lambda: self._jog(0, 0, self.current_jog_step))
        z_col.addWidget(self.btn_zp)

        self.btn_zm = QPushButton("▼ -Z (Out)")
        self.btn_zm.clicked.connect(lambda: self._jog(0, 0, -self.current_jog_step))
        z_col.addWidget(self.btn_zm)
        jog_row.addLayout(z_col)

        motion_layout.addLayout(jog_row)

        # Homing & Query Config buttons
        home_row = QHBoxLayout()
        self.btn_home = QPushButton("Home Stage ($H)")
        self.btn_home.clicked.connect(self._on_home_clicked)
        home_row.addWidget(self.btn_home)

        self.btn_query = QPushButton("Query Config")
        self.btn_query.clicked.connect(lambda: self.engine.stage.get_position())
        home_row.addWidget(self.btn_query)
        motion_layout.addLayout(home_row)

        layout.addWidget(motion_box)

        # 3. Focus & Alignment Controls
        af_box = QGroupBox("Focus & Alignment")
        af_layout = QVBoxLayout(af_box)
        af_layout.setContentsMargins(6, 6, 6, 6)
        af_layout.setSpacing(4)

        af_btn_row = QHBoxLayout()
        self.btn_autofocus = QPushButton("Run Autofocus")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        af_btn_row.addWidget(self.btn_autofocus)

        self.btn_align = QPushButton("Align to Marks")
        self.btn_align.clicked.connect(self._on_align_clicked)
        af_btn_row.addWidget(self.btn_align)
        af_layout.addLayout(af_btn_row)


        layout.addWidget(af_box)

        # 4. Projector Illumination Modes
        proj_box = QGroupBox("Projector Mode")
        proj_layout = QVBoxLayout(proj_box)
        proj_layout.setContentsMargins(6, 6, 6, 6)
        proj_layout.setSpacing(4)

        mode_btn_row = QHBoxLayout()
        self.btn_mode_clear = QPushButton("Clear")
        self.btn_mode_clear.clicked.connect(lambda: self.engine.projector.set_mode(ShownImage.CLEAR))
        mode_btn_row.addWidget(self.btn_mode_clear)

        self.btn_mode_red = QPushButton("Red Only")
        self.btn_mode_red.setStyleSheet("background-color: #991b1b; color: white; font-weight: bold;")
        self.btn_mode_red.clicked.connect(self._on_mode_red_clicked)
        mode_btn_row.addWidget(self.btn_mode_red)

        self.btn_mode_uv = QPushButton("UV Mode")
        self.btn_mode_uv.setStyleSheet("background-color: #4338ca; color: white; font-weight: bold;")
        self.btn_mode_uv.clicked.connect(self._on_mode_uv_clicked)
        mode_btn_row.addWidget(self.btn_mode_uv)
        proj_layout.addLayout(mode_btn_row)

        # Red source sub-selector
        red_src_row = QHBoxLayout()
        red_src_row.addWidget(QLabel("Red Source:"))
        self.btn_red_src_pattern = QPushButton("Pattern")
        self.btn_red_src_pattern.clicked.connect(self._on_red_pattern_clicked)
        red_src_row.addWidget(self.btn_red_src_pattern)

        self.btn_red_src_solid = QPushButton("Solid")
        self.btn_red_src_solid.clicked.connect(self._on_red_solid_clicked)
        red_src_row.addWidget(self.btn_red_src_solid)

        self.btn_red_src_img = QPushButton("Crosshair")
        self.btn_red_src_img.clicked.connect(self._on_red_crosshair_clicked)
        red_src_row.addWidget(self.btn_red_src_img)
        proj_layout.addLayout(red_src_row)

        layout.addWidget(proj_box)
        layout.addStretch()

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_pos_changed)
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_aborted.connect(lambda *_: self._update_lock_state())
        self._update_lock_state()

    def _on_mode_red_clicked(self):
        layer = self.engine.project.active_layer
        img = layer.render_pattern(self.engine.project.settings, self.engine.projector.size(), color_channels=(True, False, False))
        self.engine.projector.set_mode(ShownImage.RED_FOCUS, img)

    def _on_mode_uv_clicked(self):
        layer = self.engine.project.active_layer
        img = layer.render_pattern(self.engine.project.settings, self.engine.projector.size(), color_channels=(False, False, True))
        self.engine.projector.set_mode(ShownImage.UV_FOCUS, img)

    def _on_red_pattern_clicked(self):
        self._on_mode_red_clicked()

    def _on_red_solid_clicked(self):
        from PIL import Image
        img = Image.new("RGB", self.engine.projector.size(), "red")
        self.engine.projector.set_mode(ShownImage.RED_FOCUS, img)

    def _on_red_crosshair_clicked(self):
        from PIL import Image, ImageDraw
        w, h = self.engine.projector.size()
        img = Image.new("RGB", (w, h), "black")
        draw = ImageDraw.Draw(img)
        draw.line([(w // 2, 0), (w // 2, h)], fill="red", width=2)
        draw.line([(0, h // 2), (w, h // 2)], fill="red", width=2)
        self.engine.projector.set_mode(ShownImage.RED_FOCUS, img)

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
        op = AutofocusOperation(blue_only=(self.engine.projector.mode in (ShownImage.UV_FOCUS, ShownImage.PATTERN)))
        self.bridge.start_operation(op)

    def _on_align_clicked(self):
        op = AlignmentOperation()
        self.bridge.start_operation(op)

    def _on_pos_changed(self, coords: tuple):
        x, y, z = coords
        self.lbl_x.setText(f"X: {x:.3f} mm")
        self.lbl_y.setText(f"Y: {y:.3f} mm")
        self.lbl_z.setText(f"Z: {z:.3f} mm")

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        for btn in [
            self.btn_yp,
            self.btn_ym,
            self.btn_xp,
            self.btn_xm,
            self.btn_zp,
            self.btn_zm,
            self.btn_home,
            self.btn_autofocus,
            self.btn_align,
            self.btn_mode_clear,
            self.btn_mode_red,
            self.btn_mode_uv,
        ]:
            btn.setEnabled(not is_busy)

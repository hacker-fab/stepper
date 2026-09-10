import os
from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.alignment import compute_standard_alignment_offset, detect_alignment_markers
from core.engine import StepperEngine
from core.events import MovementLock, RedFocusSource, ShownImage
from core.tiling import calculate_tile_position, generate_snake_sequence
from ui.bridge import QtEngineBridge


class ProcessControlPanelWidget(QTabWidget):
    """Grouped panel for all step/action/process controls (tabbed for clarity)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.current_jog_step = 100.0  # µm default

        # Tab 1: Stage Motion
        self.motion_tab = QWidget()
        self._setup_motion_tab()
        self.addTab(self.motion_tab, "Stage Motion")

        # Tab 2: Focus & Align
        self.focus_tab = QWidget()
        self._setup_focus_tab()
        self.addTab(self.focus_tab, "Focus & Align")

        # Tab 3: UV Exposure
        self.exposure_tab = QWidget()
        self._setup_exposure_tab()
        self.addTab(self.exposure_tab, "UV Exposure")

        # Tab 4: Tiling Run
        self.tiling_run_tab = QWidget()
        self._setup_tiling_run_tab()
        self.addTab(self.tiling_run_tab, "Tiling Run")

        # Connect signals
        self.bridge.movement_lock_changed.connect(self._on_lock_changed)

    # -------------------------------------------------------------------------
    # Tab 1: Stage Motion
    # -------------------------------------------------------------------------
    def _setup_motion_tab(self):
        layout = QVBoxLayout(self.motion_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Step size selectors
        step_group = QGroupBox("Step Size")
        step_layout = QHBoxLayout(step_group)
        self.step_btn_group = QButtonGroup(self)

        steps = [("10 µm", 10.0), ("100 µm", 100.0), ("1 mm", 1000.0), ("10 mm", 10000.0)]
        for text, val in steps:
            rb = QRadioButton(text)
            if val == 100.0:
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, v=val: self._on_step_changed(checked, v))
            self.step_btn_group.addButton(rb)
            step_layout.addWidget(rb)

        layout.addWidget(step_group)

        # Jog controls grid
        jog_row = QHBoxLayout()

        # XY Jog D-Pad
        xy_group = QGroupBox("XY Pan")
        xy_grid = QGridLayout(xy_group)
        xy_grid.setSpacing(4)

        self.btn_yp = QPushButton("▲ Y+")
        self.btn_yp.clicked.connect(lambda: self._jog(0, self.current_jog_step, 0))
        xy_grid.addWidget(self.btn_yp, 0, 1)

        self.btn_xm = QPushButton("◀ X-")
        self.btn_xm.clicked.connect(lambda: self._jog(-self.current_jog_step, 0, 0))
        xy_grid.addWidget(self.btn_xm, 1, 0)

        self.btn_center = QLabel("XY")
        self.btn_center.setAlignment(Qt.AlignCenter)
        xy_grid.addWidget(self.btn_center, 1, 1)

        self.btn_xp = QPushButton("X+ ▶")
        self.btn_xp.clicked.connect(lambda: self._jog(self.current_jog_step, 0, 0))
        xy_grid.addWidget(self.btn_xp, 1, 2)

        self.btn_ym = QPushButton("▼ Y-")
        self.btn_ym.clicked.connect(lambda: self._jog(0, -self.current_jog_step, 0))
        xy_grid.addWidget(self.btn_ym, 2, 1)

        jog_row.addWidget(xy_group)

        # Z Focus Jog
        z_group = QGroupBox("Z Focus")
        z_layout = QVBoxLayout(z_group)
        self.btn_zp = QPushButton("▲ +Z (In)")
        self.btn_zp.clicked.connect(lambda: self._jog(0, 0, self.current_jog_step))
        z_layout.addWidget(self.btn_zp)

        self.btn_zm = QPushButton("▼ -Z (Out)")
        self.btn_zm.clicked.connect(lambda: self._jog(0, 0, -self.current_jog_step))
        z_layout.addWidget(self.btn_zm)

        jog_row.addWidget(z_group)
        layout.addLayout(jog_row)

        # Homing & Zeroing row
        actions_row = QHBoxLayout()
        self.btn_home = QPushButton("Home Stage ($H)")
        self.btn_home.clicked.connect(self._on_home_clicked)
        actions_row.addWidget(self.btn_home)

        self.btn_query = QPushButton("Query Config")
        self.btn_query.clicked.connect(lambda: self.engine.query_config())
        actions_row.addWidget(self.btn_query)

        layout.addLayout(actions_row)
        layout.addStretch()

    def _on_step_changed(self, checked: bool, val: float):
        if checked:
            self.current_jog_step = val

    def _jog(self, dx: float, dy: float, dz: float):
        # Convert µm to mm if needed (GRBL stage move_relative receives µm in coords)
        self.engine.move_relative({"x": dx, "y": dy, "z": dz})

    def _on_home_clicked(self):
        self.bridge.status_message.emit("Homing Stage...", is_busy=True)
        self.bridge.run_async(
            self.engine.home_stage,
            on_finished=lambda _: self.bridge.status_message.emit("Homing Complete"),
            on_error=lambda err: self.bridge.warning_emitted.emit(f"Homing failed: {err}"),
        )

    # -------------------------------------------------------------------------
    # Tab 2: Focus & Align
    # -------------------------------------------------------------------------
    def _setup_focus_tab(self):
        layout = QVBoxLayout(self.focus_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Safe Red Mode
        red_group = QGroupBox("Safe Red Illumination")
        red_layout = QVBoxLayout(red_group)

        self.btn_red_mode = QPushButton("Enter Red Focus Mode")
        self.btn_red_mode.setStyleSheet("background-color: #7f1d1d; font-weight: bold;")
        self.btn_red_mode.clicked.connect(lambda: self.engine.enter_red_mode())
        red_layout.addWidget(self.btn_red_mode)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.btn_src_pattern = QPushButton("Pattern")
        self.btn_src_pattern.clicked.connect(
            lambda: self.engine.set_red_focus_source(RedFocusSource.PATTERN)
        )
        src_row.addWidget(self.btn_src_pattern)

        self.btn_src_solid = QPushButton("Solid Red")
        self.btn_src_solid.clicked.connect(
            lambda: self.engine.set_red_focus_source(RedFocusSource.SOLID)
        )
        src_row.addWidget(self.btn_src_solid)

        self.btn_src_img = QPushButton("Image")
        self.btn_src_img.clicked.connect(
            lambda: self.engine.set_red_focus_source(RedFocusSource.IMAGE)
        )
        src_row.addWidget(self.btn_src_img)

        red_layout.addLayout(src_row)
        layout.addWidget(red_group)

        # Autofocus
        af_group = QGroupBox("Autofocus")
        af_layout = QVBoxLayout(af_group)

        self.btn_autofocus = QPushButton("Run Autofocus")
        self.btn_autofocus.clicked.connect(self._on_autofocus_clicked)
        af_layout.addWidget(self.btn_autofocus)
        layout.addWidget(af_group)

        # Optical Alignment (YOLO)
        align_group = QGroupBox("Optical Alignment (YOLO)")
        align_layout = QVBoxLayout(align_group)

        self.cb_detection = QCheckBox("Real-time Marker Detection")
        self.cb_detection.setChecked(self.engine.realtime_detection)
        self.cb_detection.toggled.connect(self._on_realtime_detection_toggled)
        align_layout.addWidget(self.cb_detection)

        self.btn_align = QPushButton("Align to Markers")
        self.btn_align.clicked.connect(self._on_align_clicked)
        align_layout.addWidget(self.btn_align)

        layout.addWidget(align_group)
        layout.addStretch()

    def _on_autofocus_clicked(self):
        self.bridge.status_message.emit("Running Autofocus...", is_busy=True)
        self.bridge.run_async(
            self.engine.autofocus,
            blue_only=self.engine.in_uv(),
            on_finished=lambda _: self.bridge.status_message.emit("Autofocus Complete"),
            on_error=lambda err: self.bridge.warning_emitted.emit(f"Autofocus failed: {err}"),
        )

    def _on_realtime_detection_toggled(self, checked: bool):
        self.engine.realtime_detection = checked

    def _on_align_clicked(self):
        if self.engine.camera_image is None or self.engine.model is None or not self.engine.config:
            self.bridge.warning_emitted.emit("Camera feed or alignment model not available.")
            return

        h, w = self.engine.camera_image.shape[:2]
        markers, _ = detect_alignment_markers(self.engine.model, self.engine.camera_image)
        if not markers:
            self.bridge.status_message.emit("No alignment markers detected")
            return

        dx, dy = compute_standard_alignment_offset(markers, w, h, self.engine.config.alignment)
        self.engine.move_relative({"x": dx, "y": dy})
        self.bridge.status_message.emit(f"Aligned: dx={dx:.2f}µm, dy={dy:.2f}µm")

    # -------------------------------------------------------------------------
    # Tab 3: UV Exposure
    # -------------------------------------------------------------------------
    def _setup_exposure_tab(self):
        layout = QVBoxLayout(self.exposure_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        form = QFormLayout()
        self.spin_exp_time = QSpinBox()
        self.spin_exp_time.setRange(100, 600000)
        self.spin_exp_time.setValue(8000)
        self.spin_exp_time.setSingleStep(500)
        self.spin_exp_time.setSuffix(" ms")
        self.spin_exp_time.valueChanged.connect(
            lambda val: setattr(self.engine, "exposure_time", val)
        )
        form.addRow("Exposure Duration:", self.spin_exp_time)
        layout.addLayout(form)

        self.btn_uv_mode = QPushButton("Enter UV Mode (Clear Output)")
        self.btn_uv_mode.setStyleSheet("background-color: #312e81; font-weight: bold;")
        self.btn_uv_mode.clicked.connect(lambda: self.engine.enter_uv_mode())
        layout.addWidget(self.btn_uv_mode)

        self.btn_begin_patterning = QPushButton("Begin Patterning Exposure")
        self.btn_begin_patterning.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; padding: 10px; font-size: 13px;"
        )
        self.btn_begin_patterning.clicked.connect(self._on_begin_patterning_clicked)
        layout.addWidget(self.btn_begin_patterning)

        self.btn_abort_exp = QPushButton("Abort Exposure")
        self.btn_abort_exp.setStyleSheet(
            "background-color: #dc2626; color: white; font-weight: bold;"
        )
        self.btn_abort_exp.clicked.connect(lambda: self.engine.abort_patterning())
        layout.addWidget(self.btn_abort_exp)

        layout.addStretch()

    def _on_begin_patterning_clicked(self):
        self.bridge.status_message.emit("Starting UV exposure...", is_busy=True)
        self.bridge.run_async(
            self.engine.begin_patterning,
            on_finished=lambda _: self.bridge.status_message.emit("Exposure Finished"),
            on_error=lambda err: self.bridge.warning_emitted.emit(f"Exposure failed: {err}"),
        )

    # -------------------------------------------------------------------------
    # Tab 4: Tiling Run
    # -------------------------------------------------------------------------
    def _setup_tiling_run_tab(self):
        layout = QVBoxLayout(self.tiling_run_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        self.spin_tiling_x_count = QSpinBox()
        self.spin_tiling_x_count.setRange(1, 100)
        self.spin_tiling_x_count.setValue(3)
        form.addRow("X Tiles Count:", self.spin_tiling_x_count)

        self.spin_tiling_y_count = QSpinBox()
        self.spin_tiling_y_count.setRange(1, 100)
        self.spin_tiling_y_count.setValue(3)
        form.addRow("Y Tiles Count:", self.spin_tiling_y_count)

        self.spin_pitch_x = QDoubleSpinBox()
        self.spin_pitch_x.setRange(10.0, 50000.0)
        self.spin_pitch_x.setValue(983.0)  # 1037 - 54
        self.spin_pitch_x.setSuffix(" µm")
        form.addRow("X Pitch Offset:", self.spin_pitch_x)

        self.spin_pitch_y = QDoubleSpinBox()
        self.spin_pitch_y.setRange(10.0, 50000.0)
        self.spin_pitch_y.setValue(512.0)  # 539 - 27
        self.spin_pitch_y.setSuffix(" µm")
        form.addRow("Y Pitch Offset:", self.spin_pitch_y)

        layout.addLayout(form)

        self.btn_start_tiling = QPushButton("Begin Step-and-Repeat Tiling")
        self.btn_start_tiling.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_start_tiling.clicked.connect(self._on_start_tiling_clicked)
        layout.addWidget(self.btn_start_tiling)

        self.btn_abort_tiling = QPushButton("Abort Tiling")
        self.btn_abort_tiling.clicked.connect(lambda: self.engine.abort_patterning())
        layout.addWidget(self.btn_abort_tiling)

        layout.addStretch()

    def _on_start_tiling_clicked(self):
        x_count = self.spin_tiling_x_count.value()
        y_count = self.spin_tiling_y_count.value()
        x_pitch = self.spin_pitch_x.value()
        y_pitch = self.spin_pitch_y.value()

        def tiling_worker():
            seq = generate_snake_sequence(x_count, y_count)
            x_start, y_start = self.engine.stage_setpoint[0], self.engine.stage_setpoint[1]

            for i, (x_idx, y_idx) in enumerate(seq):
                if self.engine.should_abort:
                    break

                self.bridge.status_message.emit(
                    f"Tiling ({i + 1}/{len(seq)}): Tile ({x_idx}, {y_idx})", is_busy=True
                )

                # Move stage if not first tile
                if not (x_idx == 0 and y_idx == 0):
                    tx, ty = calculate_tile_position(
                        x_start, y_start, 1, 1, x_idx, y_idx, x_pitch, y_pitch
                    )
                    self.engine.move_absolute({"x": tx, "y": ty})

                # Red autofocus
                self.engine.autofocus(blue_only=False)

                # Load tile image if exists
                tile_path = f"tiles/tile_{y_idx},{x_idx}.png"
                if os.path.exists(tile_path):
                    tile_img = Image.open(tile_path)
                    self.engine.set_pattern_image(tile_img, tile_path)

                # Switch to UV mode and expose
                self.engine.enter_uv_mode(mode_switch_autofocus=False)
                self.engine.begin_patterning()
                self.engine.enter_red_mode(mode_switch_autofocus=False)

            self.bridge.status_message.emit("Tiling Run Complete")

        self.bridge.run_async(tiling_worker)

    def _on_lock_changed(self, lock: MovementLock):
        is_locked = lock == MovementLock.LOCKED
        self.motion_tab.setEnabled(not is_locked)
        self.btn_autofocus.setEnabled(not is_locked)
        self.btn_align.setEnabled(not is_locked)
        self.btn_begin_patterning.setEnabled(not is_locked)
        self.btn_start_tiling.setEnabled(not is_locked)


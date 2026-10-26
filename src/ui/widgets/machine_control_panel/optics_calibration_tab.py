from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations import OpticsCalibrationOperation
from ui.bridge import QtEngineBridge


class OpticsCalibrationTabWidget(QScrollArea):
    """Tab 2: Optics chromatic Z offset calibration procedure and controls."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._latest_cal_offset: Optional[float] = None
        self._on_apply_callback = None

        container = QWidget()
        optics_layout = QVBoxLayout(container)
        optics_layout.setContentsMargins(6, 6, 6, 6)
        optics_layout.setSpacing(8)

        info_box = QGroupBox("Calibration Procedure")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>Optics Chromatic Z Offset Calibration</b><br><br>"
            "1. Place a bare silicon chip on the stage.<br>"
            "2. Perform rough manual focus on the chip surface.<br>"
            "3. Click <b>'Start Optics Calibration'</b> below.<br>"
            "4. The system will project Red ArUco tags, verify detection, and find the optimal Red focal plane (Z_red).<br>"
            "5. It will then switch to UV illumination and find the UV focal plane (Z_uv).<br>"
            "6. The chromatic offset ΔZ = Z_uv - Z_red will be calculated and reported."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        optics_layout.addWidget(info_box)

        cal_ctrl_box = QGroupBox("Optics Calibration Controls")
        cal_ctrl_layout = QVBoxLayout(cal_ctrl_box)

        param_grid = QGridLayout()
        param_grid.addWidget(QLabel("Search Range:"), 0, 0)
        self.spin_cal_search_range = QDoubleSpinBox()
        self.spin_cal_search_range.setRange(1.0, 5000.0)
        self.spin_cal_search_range.setDecimals(1)
        self.spin_cal_search_range.setSingleStep(10.0)
        self.spin_cal_search_range.setValue(500.0)
        self.spin_cal_search_range.setSuffix(" µm")
        param_grid.addWidget(self.spin_cal_search_range, 0, 1)

        param_grid.addWidget(QLabel("Target Accuracy:"), 1, 0)
        self.spin_cal_target_accuracy = QDoubleSpinBox()
        self.spin_cal_target_accuracy.setRange(0.05, 50.0)
        self.spin_cal_target_accuracy.setDecimals(2)
        self.spin_cal_target_accuracy.setSingleStep(0.1)
        self.spin_cal_target_accuracy.setValue(0.5)
        self.spin_cal_target_accuracy.setSuffix(" µm")
        param_grid.addWidget(self.spin_cal_target_accuracy, 1, 1)

        cal_ctrl_layout.addLayout(param_grid)

        self.btn_start_optics_cal = QPushButton("Start Optics Calibration")
        self.btn_start_optics_cal.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_optics_cal.clicked.connect(self._on_start_optics_cal_clicked)
        cal_ctrl_layout.addWidget(self.btn_start_optics_cal)

        res_grid = QGridLayout()
        res_grid.addWidget(QLabel("<b>Red Focus (Z_red):</b>"), 0, 0)
        self.lbl_cal_red_z = QLabel("--")
        self.lbl_cal_red_z.setStyleSheet("font-family: monospace; font-size: 13px;")
        res_grid.addWidget(self.lbl_cal_red_z, 0, 1)

        res_grid.addWidget(QLabel("<b>UV Focus (Z_uv):</b>"), 1, 0)
        self.lbl_cal_uv_z = QLabel("--")
        self.lbl_cal_uv_z.setStyleSheet("font-family: monospace; font-size: 13px;")
        res_grid.addWidget(self.lbl_cal_uv_z, 1, 1)

        res_grid.addWidget(QLabel("<b>UV-Red Offset (ΔZ):</b>"), 2, 0)
        self.lbl_cal_offset_z = QLabel("--")
        self.lbl_cal_offset_z.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold; color: #2196F3;")
        res_grid.addWidget(self.lbl_cal_offset_z, 2, 1)

        cal_ctrl_layout.addLayout(res_grid)

        self.btn_apply_cal_offset = QPushButton("Apply Offset to Autofocus Config")
        self.btn_apply_cal_offset.setEnabled(False)
        self.btn_apply_cal_offset.clicked.connect(self._on_apply_cal_offset_clicked)
        cal_ctrl_layout.addWidget(self.btn_apply_cal_offset)

        optics_layout.addWidget(cal_ctrl_box)
        optics_layout.addStretch()

        self.setWidget(container)

    @property
    def latest_cal_offset(self) -> Optional[float]:
        return self._latest_cal_offset

    @latest_cal_offset.setter
    def latest_cal_offset(self, val: Optional[float]):
        self._latest_cal_offset = val

    def set_apply_callback(self, callback):
        self._on_apply_callback = callback

    def _on_start_optics_cal_clicked(self):
        op = OpticsCalibrationOperation(
            search_range=self.spin_cal_search_range.value(),
            target_accuracy=self.spin_cal_target_accuracy.value(),
        )
        self._active_op = op
        self.bridge.start_operation(op)

    def _on_apply_cal_offset_clicked(self):
        if self._latest_cal_offset is not None:
            if hasattr(self.engine, "autofocus_config"):
                if self.engine.autofocus_config is None:
                    from operations.autofocus import AutofocusConfig
                    self.engine.autofocus_config = AutofocusConfig()
                self.engine.autofocus_config.uv_z_offset = self._latest_cal_offset
            if self._on_apply_callback:
                self._on_apply_callback(self._latest_cal_offset)

    def on_operation_finished(self, op_or_name=None, err=None):
        from core.operation import Operation
        op = op_or_name if isinstance(op_or_name, Operation) else getattr(self, "_active_op", None)
        if not isinstance(op, Operation) and hasattr(self.engine, "operations"):
            current = self.engine.operations.current_operation
            if isinstance(current, OpticsCalibrationOperation):
                op = current
        if isinstance(op, OpticsCalibrationOperation) and err is None:
            if op.red_best_z is not None:
                self.lbl_cal_red_z.setText(f"{op.red_best_z:.2f} µm")
            if op.uv_best_z is not None:
                self.lbl_cal_uv_z.setText(f"{op.uv_best_z:.2f} µm")
            if op.uv_z_offset is not None:
                self.lbl_cal_offset_z.setText(f"{op.uv_z_offset:+.2f} µm")
                self._latest_cal_offset = op.uv_z_offset
                self.btn_apply_cal_offset.setEnabled(True)
                self._on_apply_cal_offset_clicked()

    def update_lock_state(self, is_busy: bool):
        self.btn_start_optics_cal.setEnabled(not is_busy)
        self.spin_cal_search_range.setEnabled(not is_busy)
        self.spin_cal_target_accuracy.setEnabled(not is_busy)
        if is_busy:
            self.btn_apply_cal_offset.setEnabled(False)
        elif self._latest_cal_offset is not None:
            self.btn_apply_cal_offset.setEnabled(True)

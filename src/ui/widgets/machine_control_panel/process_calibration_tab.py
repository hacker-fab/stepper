from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations.process_calibration import ProcessCalibrationOperation
from ui.bridge import QtEngineBridge


class ProcessCalibrationTabWidget(QScrollArea):
    """Tab 3: Process Calibration / FEM matrix controls."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        proc_layout = QVBoxLayout(container)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(8)

        # 1. Info / Instructions
        info_box = QGroupBox("Process Calibration Procedure")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>Focus-Exposure Matrix (FEM)</b><br><br>"
            "1. Focus at the initial substrate position.<br>"
            "2. Sweep exposure duration and Z offset from Min to Max over the configured steps.<br>"
            "3. At each step, autofocus is performed (using the UV-Red autofocus offset), the additional Z sweep offset is applied, and the calibration reticle pattern is exposed.<br>"
            "4. The stage automatically steps outward in a 2D spiral pattern from the center."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        proc_layout.addWidget(info_box)

        # 2. Controls Group
        ctrl_box = QGroupBox("Calibration Parameters")
        ctrl_layout = QVBoxLayout(ctrl_box)

        param_grid = QGridLayout()

        # Min Exposure
        param_grid.addWidget(QLabel("Min Exposure:"), 0, 0)
        self.spin_min_exposure = QDoubleSpinBox()
        self.spin_min_exposure.setRange(0.05, 600.0)
        self.spin_min_exposure.setDecimals(2)
        self.spin_min_exposure.setSingleStep(0.5)
        self.spin_min_exposure.setValue(1.0)
        self.spin_min_exposure.setSuffix(" s")
        param_grid.addWidget(self.spin_min_exposure, 0, 1)

        # Max Exposure
        param_grid.addWidget(QLabel("Max Exposure:"), 1, 0)
        self.spin_max_exposure = QDoubleSpinBox()
        self.spin_max_exposure.setRange(0.05, 600.0)
        self.spin_max_exposure.setDecimals(2)
        self.spin_max_exposure.setSingleStep(0.5)
        self.spin_max_exposure.setValue(5.0)
        self.spin_max_exposure.setSuffix(" s")
        param_grid.addWidget(self.spin_max_exposure, 1, 1)

        # Sweep Steps (Exposure steps)
        param_grid.addWidget(QLabel("Exposure Steps:"), 2, 0)
        self.spin_sweep_steps = QSpinBox()
        self.spin_sweep_steps.setRange(1, 100)
        self.spin_sweep_steps.setSingleStep(1)
        self.spin_sweep_steps.setValue(5)
        param_grid.addWidget(self.spin_sweep_steps, 2, 1)

        # Min Z Offset
        param_grid.addWidget(QLabel("Min Z Offset:"), 3, 0)
        self.spin_min_z_offset = QDoubleSpinBox()
        self.spin_min_z_offset.setRange(-1000.0, 1000.0)
        self.spin_min_z_offset.setDecimals(2)
        self.spin_min_z_offset.setSingleStep(1.0)
        self.spin_min_z_offset.setValue(0.0)
        self.spin_min_z_offset.setSuffix(" µm")
        param_grid.addWidget(self.spin_min_z_offset, 3, 1)

        # Max Z Offset
        param_grid.addWidget(QLabel("Max Z Offset:"), 4, 0)
        self.spin_max_z_offset = QDoubleSpinBox()
        self.spin_max_z_offset.setRange(-1000.0, 1000.0)
        self.spin_max_z_offset.setDecimals(2)
        self.spin_max_z_offset.setSingleStep(1.0)
        self.spin_max_z_offset.setValue(0.0)
        self.spin_max_z_offset.setSuffix(" µm")
        param_grid.addWidget(self.spin_max_z_offset, 4, 1)

        # Z Steps
        param_grid.addWidget(QLabel("Z Steps:"), 5, 0)
        self.spin_z_steps = QSpinBox()
        self.spin_z_steps.setRange(1, 100)
        self.spin_z_steps.setSingleStep(1)
        self.spin_z_steps.setValue(1)
        param_grid.addWidget(self.spin_z_steps, 5, 1)

        # Motion Distance (pitch / step distance)
        param_grid.addWidget(QLabel("Motion Distance:"), 6, 0)
        self.spin_motion_distance = QDoubleSpinBox()
        self.spin_motion_distance.setRange(10.0, 50000.0)
        self.spin_motion_distance.setDecimals(1)
        self.spin_motion_distance.setSingleStep(100.0)
        self.spin_motion_distance.setValue(1000.0)
        self.spin_motion_distance.setSuffix(" µm")
        param_grid.addWidget(self.spin_motion_distance, 6, 1)

        ctrl_layout.addLayout(param_grid)

        # Start button
        self.btn_start_cal = QPushButton("Start Process Calibration")
        self.btn_start_cal.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_cal.clicked.connect(self._on_start_cal_clicked)
        ctrl_layout.addWidget(self.btn_start_cal)

        # Placeholder alias for backward compatibility
        self.btn_fem_placeholder = self.btn_start_cal

        # Status Label
        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #666; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_status)

        proc_layout.addWidget(ctrl_box)
        proc_layout.addStretch()

        self.setWidget(container)

    def _on_start_cal_clicked(self):
        op = ProcessCalibrationOperation(
            min_exposure=self.spin_min_exposure.value(),
            max_exposure=self.spin_max_exposure.value(),
            sweep_steps=self.spin_sweep_steps.value(),
            motion_distance=self.spin_motion_distance.value(),
            min_z_offset=self.spin_min_z_offset.value(),
            max_z_offset=self.spin_max_z_offset.value(),
            z_steps=self.spin_z_steps.value(),
            autofocus_config=getattr(self.engine, "autofocus_config", None),
        )
        self._active_op = op
        self.lbl_status.setText("Status: Running calibration...")
        self.bridge.start_operation(op)

    def on_operation_finished(self, op_or_name=None, err=None):
        from core.operation import Operation
        op = op_or_name if isinstance(op_or_name, Operation) else getattr(self, "_active_op", None)
        if isinstance(op, ProcessCalibrationOperation) or op_or_name == "Process Calibration":
            if err is not None:
                self.lbl_status.setText(f"Status: Failed - {err}")
            else:
                self.lbl_status.setText("Status: Calibration completed successfully")

    def update_lock_state(self, is_busy: bool):
        self.btn_start_cal.setEnabled(not is_busy)
        self.spin_min_exposure.setEnabled(not is_busy)
        self.spin_max_exposure.setEnabled(not is_busy)
        self.spin_sweep_steps.setEnabled(not is_busy)
        self.spin_min_z_offset.setEnabled(not is_busy)
        self.spin_max_z_offset.setEnabled(not is_busy)
        self.spin_z_steps.setEnabled(not is_busy)
        self.spin_motion_distance.setEnabled(not is_busy)

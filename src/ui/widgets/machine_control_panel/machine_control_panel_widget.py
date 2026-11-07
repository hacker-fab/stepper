from typing import Optional

from PySide6.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge
from .manual_control_tab import ManualControlTabWidget
from .machine_config_tab import MachineConfigTabWidget
from .ml_data_collection_tab import MLDataCollectionTabWidget
from .optics_calibration_tab import OpticsCalibrationTabWidget
from .process_calibration_tab import ProcessCalibrationTabWidget
from .stage_calibration_tab import StageCalibrationTabWidget



class MachineControlPanelWidget(QWidget):
    """Right dock panel: Machine motion control, autofocus, and calibration tabs."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Tab Widget
        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs)

        # Tab 1: Manual Control
        self.manual_tab = ManualControlTabWidget(self.engine, self.bridge, self)
        self.tabs.addTab(self.manual_tab, "Manual Control")

        # Tab 2: Optics Calibration
        self.optics_tab = OpticsCalibrationTabWidget(self.engine, self.bridge, self)
        self.optics_tab.set_apply_callback(self._on_optics_offset_applied)
        self.tabs.addTab(self.optics_tab, "Optics Calibration")

        # Tab 3: Process Calibration
        self.process_tab = ProcessCalibrationTabWidget(self.engine, self.bridge, self)
        self.tabs.addTab(self.process_tab, "Process Calibration")

        # Tab 4: ML Data Collection
        self.ml_tab = MLDataCollectionTabWidget(self.engine, self.bridge, self)
        self.tabs.addTab(self.ml_tab, "ML Data Collection")

        # Tab 5: Stage Calibration
        self.stage_cal_tab = StageCalibrationTabWidget(self.engine, self.bridge, self)
        self.tabs.addTab(self.stage_cal_tab, "Stage Calibration")

        # Tab 6: Machine Config
        self.config_tab = MachineConfigTabWidget(self.engine, self.bridge, self)
        self.tabs.addTab(self.config_tab, "Machine Config")


        # Connect signals
        self.bridge.stage_position_changed.connect(self.manual_tab._on_pos_changed)
        self.bridge.projector_on_off_changed.connect(self.manual_tab._sync_projector_on_off)
        self.bridge.projector_color_mode_changed.connect(self.manual_tab._sync_color_mode)
        self.bridge.projector_image_source_changed.connect(self.manual_tab._sync_image_source)
        self.bridge.projector_brightness_changed.connect(self.manual_tab._sync_brightness)
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(self._on_operation_finished)
        self.bridge.operation_aborted.connect(self._on_operation_aborted)
        self.bridge.operation_failed.connect(self._on_operation_failed)

        self.manual_tab._sync_projector_on_off(self.engine.projector.is_on)
        self.manual_tab._sync_brightness(self.engine.projector.brightness)
        self._update_lock_state()

    def _on_optics_offset_applied(self, offset: float):
        self.config_tab.spin_uv_offset.setValue(offset)
        self.engine.machine_config.optics.uv_z_offset = offset

    def _on_operation_finished(self, op_or_name=None, err=None):
        self.optics_tab.on_operation_finished(op_or_name, err)
        self.process_tab.on_operation_finished(op_or_name, err)
        self.ml_tab.on_operation_finished(op_or_name, err)
        self.stage_cal_tab._on_operation_finished(op_or_name, err)
        self.config_tab._refresh_calibration_status()
        self._update_lock_state()

    def _on_operation_aborted(self, op_or_name=None):
        self.optics_tab.on_operation_finished(op_or_name, "Operation aborted")
        self.process_tab.on_operation_finished(op_or_name, "Operation aborted")
        self.ml_tab.on_operation_finished(op_or_name, "Operation aborted")
        self.stage_cal_tab._on_operation_finished(op_or_name, "Operation aborted")
        self.config_tab._refresh_calibration_status()
        self._update_lock_state()

    def _on_operation_failed(self, op_or_name=None, err=None):
        self.optics_tab.on_operation_finished(op_or_name, err)
        self.process_tab.on_operation_finished(op_or_name, err)
        self.ml_tab.on_operation_finished(op_or_name, err)
        self.stage_cal_tab._on_operation_finished(op_or_name, err)
        self.config_tab._refresh_calibration_status()
        self._update_lock_state()

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        self.manual_tab.update_lock_state(is_busy)
        self.optics_tab.update_lock_state(is_busy)
        self.process_tab.update_lock_state(is_busy)
        self.ml_tab.update_lock_state(is_busy)
        self.stage_cal_tab.update_lock_state(is_busy)
        self.config_tab.update_lock_state(is_busy)


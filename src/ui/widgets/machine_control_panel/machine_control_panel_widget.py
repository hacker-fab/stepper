from typing import Optional

from PySide6.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge
from .manual_control_tab import ManualControlTabWidget
from .ml_data_collection_tab import MLDataCollectionTabWidget
from .optics_calibration_tab import OpticsCalibrationTabWidget
from .process_calibration_tab import ProcessCalibrationTabWidget


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

        # Direct references for backwards compatibility with tests and callers
        self.lbl_pos_x = self.manual_tab.lbl_pos_x
        self.lbl_pos_y = self.manual_tab.lbl_pos_y
        self.lbl_pos_z = self.manual_tab.lbl_pos_z
        self.step_btn_group = self.manual_tab.step_btn_group
        self.btn_y_pos = self.manual_tab.btn_y_pos
        self.btn_x_neg = self.manual_tab.btn_x_neg
        self.btn_home = self.manual_tab.btn_home
        self.btn_x_pos = self.manual_tab.btn_x_pos
        self.btn_y_neg = self.manual_tab.btn_y_neg
        self.btn_z_pos = self.manual_tab.btn_z_pos
        self.btn_z_neg = self.manual_tab.btn_z_neg
        self.spin_sharp_z_range = self.manual_tab.spin_sharp_z_range
        self.btn_maximize_sharpness = self.manual_tab.btn_maximize_sharpness
        self.spin_uv_offset = self.manual_tab.spin_uv_offset
        self.btn_autofocus = self.manual_tab.btn_autofocus
        self.btn_align = self.manual_tab.btn_align
        self.btn_projector_power = self.manual_tab.btn_projector_power
        self.color_btn_group = self.manual_tab.color_btn_group
        self.radio_color_red = self.manual_tab.radio_color_red
        self.radio_color_uv = self.manual_tab.radio_color_uv
        self.src_btn_group = self.manual_tab.src_btn_group
        self.radio_src_active = self.manual_tab.radio_src_active
        self.radio_src_custom = self.manual_tab.radio_src_custom
        self.radio_src_solid = self.manual_tab.radio_src_solid
        self.txt_custom_file = self.manual_tab.txt_custom_file
        self.btn_browse_custom = self.manual_tab.btn_browse_custom
        self.custom_file_row = self.manual_tab.custom_file_row

        self.btn_start_optics_cal = self.optics_tab.btn_start_optics_cal
        self.spin_cal_search_range = self.optics_tab.spin_cal_search_range
        self.spin_cal_target_accuracy = self.optics_tab.spin_cal_target_accuracy
        self.lbl_cal_red_z = self.optics_tab.lbl_cal_red_z
        self.lbl_cal_uv_z = self.optics_tab.lbl_cal_uv_z
        self.lbl_cal_offset_z = self.optics_tab.lbl_cal_offset_z
        self.btn_apply_cal_offset = self.optics_tab.btn_apply_cal_offset

        self.btn_fem_placeholder = self.process_tab.btn_fem_placeholder
        self.btn_start_process_cal = self.process_tab.btn_start_cal
        self.spin_cal_min_exp = self.process_tab.spin_min_exposure
        self.spin_cal_max_exp = self.process_tab.spin_max_exposure
        self.spin_cal_sweep_steps = self.process_tab.spin_sweep_steps
        self.spin_cal_min_z = self.process_tab.spin_min_z_offset
        self.spin_cal_max_z = self.process_tab.spin_max_z_offset
        self.spin_cal_z_steps = self.process_tab.spin_z_steps
        self.spin_cal_motion_dist = self.process_tab.spin_motion_distance

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

    @property
    def current_jog_step(self) -> float:
        return self.manual_tab.current_jog_step

    @current_jog_step.setter
    def current_jog_step(self, val: float):
        self.manual_tab.current_jog_step = val

    @property
    def _latest_cal_offset(self) -> Optional[float]:
        return self.optics_tab.latest_cal_offset

    @_latest_cal_offset.setter
    def _latest_cal_offset(self, val: Optional[float]):
        self.optics_tab.latest_cal_offset = val

    def _on_optics_offset_applied(self, offset: float):
        self.spin_uv_offset.setValue(offset)

    def _on_operation_finished(self, op_or_name=None, err=None):
        self.optics_tab.on_operation_finished(op_or_name, err)
        self.process_tab.on_operation_finished(op_or_name, err)
        self.ml_tab.on_operation_finished(op_or_name, err)
        self._update_lock_state()

    def _on_operation_aborted(self, op_or_name=None):
        self.optics_tab.on_operation_finished(op_or_name, "Operation aborted")
        self.process_tab.on_operation_finished(op_or_name, "Operation aborted")
        self.ml_tab.on_operation_finished(op_or_name, "Operation aborted")
        self._update_lock_state()

    def _on_operation_failed(self, op_or_name=None, err=None):
        self.optics_tab.on_operation_finished(op_or_name, err)
        self.process_tab.on_operation_finished(op_or_name, err)
        self.ml_tab.on_operation_finished(op_or_name, err)
        self._update_lock_state()

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        self.manual_tab.update_lock_state(is_busy)
        self.optics_tab.update_lock_state(is_busy)
        self.process_tab.update_lock_state(is_busy)
        self.ml_tab.update_lock_state(is_busy)

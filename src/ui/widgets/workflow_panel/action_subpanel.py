from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations import ExposureOperation, ExposureOperationConfig, TiledExposureOperation
from ui.bridge import QtEngineBridge


class ActionSubpanelWidget(QWidget):
    """Right-most portion: Layer execution dashboard and expose trigger."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Status header card
        card = QGroupBox("Active Layer Execution")
        card_layout = QFormLayout(card)
        card_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_active_name = QLabel("Layer 1")
        self.lbl_active_name.setStyleSheet("font-weight: bold; font-size: 13px;")
        card_layout.addRow("Active Target:", self.lbl_active_name)

        self.lbl_active_exp = QLabel("8000 ms")
        self.lbl_active_exp.setStyleSheet("font-weight: bold;")
        card_layout.addRow("Exposure:", self.lbl_active_exp)

        self.lbl_active_mode = QLabel("Single Field Exposure")
        self.lbl_active_mode.setStyleSheet("font-weight: bold;")
        card_layout.addRow("Execution Mode:", self.lbl_active_mode)

        layout.addWidget(card)

        # Big trigger action button
        self.btn_expose = QPushButton("Expose Layer")
        self.btn_expose.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        self.btn_expose.clicked.connect(self._on_expose_clicked)
        layout.addWidget(self.btn_expose)

        # Connect signals
        self.bridge.project_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.active_layer_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_dashboard())
        self.bridge.exposure_history_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.projector_image_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_aborted.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_failed.connect(lambda *_: self._update_lock_state())
        self._refresh_dashboard()
        self._update_lock_state()

    def _refresh_dashboard(self):
        layer = self.engine.project.active_layer
        effective = self.engine.project.settings.with_overrides(layer.overrides)

        self.lbl_active_name.setText(layer.name)
        self.lbl_active_exp.setText(f"{int(effective.exposure_time)} ms")

        if effective.tiling_enabled:
            self.lbl_active_mode.setText("Step-and-Repeat Tiling")
            self.btn_expose.setText("Run Tiled Exposure")
        else:
            self.lbl_active_mode.setText("Single Field Exposure")
            self.btn_expose.setText("Expose Layer")

    def _on_expose_clicked(self):
        layer_idx = self.engine.project.active_layer_index
        layer = self.engine.project.active_layer
        effective = self.engine.project.settings.with_overrides(layer.overrides)

        if effective.tiling_enabled:
            op = TiledExposureOperation(
                layer_index=layer_idx,
                settings=effective,
                autofocus_config=self.engine.autofocus_config,
            )
        else:
            op = ExposureOperation(
                layer_index=layer_idx,
                config=ExposureOperationConfig(exposure_time=effective.exposure_time),
            )

        self.bridge.start_operation(op)

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        self.btn_expose.setEnabled(not is_busy)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations import ExposureOperation, LayerExposureOperation
from ui.bridge import QtEngineBridge


class ActionSubpanelWidget(QWidget):
    """Right-most portion: Layer execution dashboard and expose trigger."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Status header card
        card = QGroupBox("Active Layer Execution")
        card_layout = QFormLayout(card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        self.lbl_active_name = QLabel("Layer 1")
        self.lbl_active_name.setStyleSheet("font-weight: bold; font-size: 13px;")
        card_layout.addRow("Active Target:", self.lbl_active_name)

        self.lbl_active_exp = QLabel("8000 ms")
        self.lbl_active_exp.setStyleSheet("font-weight: bold;")
        card_layout.addRow("Exposure:", self.lbl_active_exp)

        self.lbl_active_mode = QLabel("Fullscreen Projector")
        self.lbl_active_mode.setStyleSheet("font-weight: bold;")
        card_layout.addRow("Execution Mode:", self.lbl_active_mode)

        self.lbl_marker_status = QLabel("Identity (0/2 Markers)")
        self.lbl_marker_status.setStyleSheet("font-size: 11px;")
        card_layout.addRow("Chip Space:", self.lbl_marker_status)

        self.combo_origin = QComboBox()
        card_layout.addRow("Target Origin:", self.combo_origin)

        layout.addWidget(card)

        # Big trigger action button (unified LayerExposureOperation with AF)
        self.btn_expose = QPushButton("Expose Layer")
        self.btn_expose.setStyleSheet("font-size: 13px; font-weight: bold; padding: 8px;")
        self.btn_expose.clicked.connect(self._on_expose_clicked)
        layout.addWidget(self.btn_expose)

        # Quick single-shot expose button (no AF, no marker check, exposes active tile right here)
        self.btn_quick_expose = QPushButton("Quick Expose Here (No AF)")
        self.btn_quick_expose.setToolTip(
            "Immediately exposes the active tile at the current stage position without autofocus or marker requirements"
        )
        self.btn_quick_expose.clicked.connect(self._on_quick_expose_clicked)
        layout.addWidget(self.btn_quick_expose)

        # Connect signals
        self.bridge.project_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.active_layer_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_dashboard())
        self.bridge.exposure_history_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.chip_progress_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.projector_image_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_aborted.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_failed.connect(lambda *_: self._update_lock_state())
        self._refresh_dashboard()
        self._update_lock_state()

    def _refresh_dashboard(self):
        project = self.engine.project
        layer_idx = project.active_layer_index
        layer = project.active_layer
        effective = project.settings.with_overrides(layer.overrides)
        cp = project.chip_progress

        self.lbl_active_name.setText(layer.name)
        self.lbl_active_exp.setText(f"{int(effective.exposure_time)} ms")

        if layer.is_fullscreen_mode:
            self.lbl_active_mode.setText("Fullscreen (1 Tile)")
        else:
            tile_cnt = max(1, len(layer._tile_cache))
            self.lbl_active_mode.setText(f"Physical Tiling ({tile_cnt} Tile(s))")

        # Update marker status
        if cp.has_reference_markers():
            if cp.marker1_corr != cp.marker1_chip or cp.marker2_corr != cp.marker2_chip:
                self.lbl_marker_status.setText("Aligned (2/2 Re-located)")
                self.lbl_marker_status.setStyleSheet("color: #55ff55; font-weight: bold; font-size: 11px;")
            else:
                self.lbl_marker_status.setText("Reference Set (2/2 Markers)")
                self.lbl_marker_status.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        elif cp.marker1_chip is not None or cp.marker2_chip is not None:
            self.lbl_marker_status.setText("Partial (1/2 Markers)")
            self.lbl_marker_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_marker_status.setText("Identity (0/2 Markers)")
            self.lbl_marker_status.setStyleSheet("color: #888888; font-size: 11px;")

        # Populate origin selector
        prev_data = self.combo_origin.currentData()
        self.combo_origin.blockSignals(True)
        self.combo_origin.clear()
        origins = cp.pattern_origins_chip

        self.combo_origin.addItem("Create new (current pos)", ("current", None))
        if origins:
            self.combo_origin.addItem(f"All previous ({len(origins)})", ("all", None))
            for idx, (ox, oy) in enumerate(origins):
                self.combo_origin.addItem(f"Idx {idx + 1}", ("index", idx))
                self.combo_origin.setItemData(
                    self.combo_origin.count() - 1,
                    f"Origin {idx + 1} at ({ox:.1f}, {oy:.1f}) µm",
                    Qt.ToolTipRole,
                )

        if prev_data is not None:
            for i in range(self.combo_origin.count()):
                if self.combo_origin.itemData(i) == prev_data:
                    self.combo_origin.setCurrentIndex(i)
                    break
        elif layer_idx > 0 and origins:
            self.combo_origin.setCurrentIndex(1)
        self.combo_origin.blockSignals(False)

    def _on_expose_clicked(self):
        layer_idx = self.engine.project.active_layer_index
        layer = self.engine.project.active_layer
        effective = self.engine.project.settings.with_overrides(layer.overrides)

        data = self.combo_origin.currentData()
        origin_mode = "auto"
        origin_index = None
        if isinstance(data, tuple) and len(data) == 2:
            origin_mode, origin_index = data

        op = LayerExposureOperation(
            layer_index=layer_idx,
            settings=effective,
            origin_mode=origin_mode,
            origin_index=origin_index,
            autofocus=True,
        )
        self.bridge.start_operation(op)

    def _on_quick_expose_clicked(self):
        data = self.combo_origin.currentData()
        origin_mode = data[0] if isinstance(data, tuple) and len(data) == 2 else None
        if origin_mode != "current":
            QMessageBox.warning(
                self,
                "Quick Expose Unavailable",
                "Quick expose is not available for overlapped exposure. Target origin mode must be set to 'Create new (current pos)'.",
            )
            return

        layer_idx = self.engine.project.active_layer_index
        layer = self.engine.project.active_layer
        effective = self.engine.project.settings.with_overrides(layer.overrides)
        tile_idx = 0 if layer.is_fullscreen_mode else self.engine.project.active_tile_index

        op = ExposureOperation(
            layer_index=layer_idx,
            exposure_time=effective.exposure_time,
            tile_index=tile_idx,
            record_origin=True,
            single_fullscreen_tile=layer.is_fullscreen_mode,
        )
        self.bridge.start_operation(op)

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        self.btn_expose.setEnabled(not is_busy)
        self.btn_quick_expose.setEnabled(not is_busy)
        self.combo_origin.setEnabled(not is_busy)


import os
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from core.events import Event
from ui.bridge import QtEngineBridge


class LayerSubpanelWidget(QTabWidget):
    """Middle portion: Pattern configuration, tiling preview, and setting overrides."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        # Tab 1: Pattern & Tiling Preview
        self.tab_pattern = QWidget()
        self._setup_pattern_tab()
        self.addTab(self.tab_pattern, "Pattern & Tiling")

        # Tab 2: Setting Overrides
        self.tab_overrides = QWidget()
        self._setup_overrides_tab()
        self.addTab(self.tab_overrides, "Setting Overrides")

        # Signals
        self.bridge.project_changed.connect(lambda _: self._on_project_or_layer_changed())
        self.bridge.active_layer_changed.connect(lambda _: self._on_project_or_layer_changed())
        self.bridge.active_tile_changed.connect(lambda _: self._refresh_tile_preview())
        self.bridge.layer_cache_recomputed.connect(lambda _: self._refresh_tile_preview())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_layer_view())
        self.bridge.projector_image_changed.connect(lambda _: self._refresh_layer_view())
        self._on_project_or_layer_changed()

    def _on_project_or_layer_changed(self):
        self._refresh_layer_view()
        self._refresh_tile_preview()

    def _setup_pattern_tab(self):
        layout = QVBoxLayout(self.tab_pattern)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # File picker row
        file_row = QHBoxLayout()
        self.btn_load_pattern = QPushButton("Select Pattern File...")
        self.btn_load_pattern.clicked.connect(self._on_load_pattern_clicked)
        file_row.addWidget(self.btn_load_pattern)

        self.lbl_pattern_file = QLabel("No pattern selected")
        self.lbl_pattern_file.setStyleSheet("font-style: italic;")
        file_row.addWidget(self.lbl_pattern_file, stretch=1)
        layout.addLayout(file_row)

        # Thumbnail preview & options
        mid_row = QHBoxLayout()
        self.lbl_thumb = QLabel("Thumbnail")
        self.lbl_thumb.setFixedSize(110, 80)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        mid_row.addWidget(self.lbl_thumb)

        opts_layout = QVBoxLayout()
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(-1, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setToolTip("Pattern binarization threshold (-1 to disable, 0-255)")
        self.spin_threshold.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.spin_threshold)

        self.lbl_threshold_hint = QLabel("(-1 to disable)")
        self.lbl_threshold_hint.setStyleSheet("color: gray;")
        thresh_row.addWidget(self.lbl_threshold_hint)
        thresh_row.addStretch()

        opts_layout.addLayout(thresh_row)
        opts_layout.addStretch()
        mid_row.addLayout(opts_layout)
        mid_row.addStretch()
        layout.addLayout(mid_row)

        # Pattern Scaling section
        scale_box = QGroupBox("Pattern Scaling")
        scale_layout = QVBoxLayout(scale_box)
        scale_layout.setContentsMargins(6, 6, 6, 6)
        scale_layout.setSpacing(6)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("W:"))
        self.spin_scale_w = QSpinBox()
        self.spin_scale_w.setRange(-1, 100000)
        self.spin_scale_w.setValue(-1)
        self.spin_scale_w.setToolTip("Target pattern width in pixels (-1 to disable scaling)")
        self.spin_scale_w.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(self.spin_scale_w)

        scale_row.addWidget(QLabel("H:"))
        self.spin_scale_h = QSpinBox()
        self.spin_scale_h.setRange(-1, 100000)
        self.spin_scale_h.setValue(-1)
        self.spin_scale_h.setToolTip("Target pattern height in pixels (-1 to disable scaling)")
        self.spin_scale_h.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(self.spin_scale_h)

        self.btn_match_projector = QPushButton("Set to Projector Resolution")
        self.btn_match_projector.clicked.connect(self._on_match_projector_clicked)
        scale_row.addWidget(self.btn_match_projector)
        scale_row.addStretch()

        scale_layout.addLayout(scale_row)
        layout.addWidget(scale_box)

        # Tiling preview & navigation section
        tiling_box = QGroupBox("Tiling & Tile Navigation")
        tiling_layout = QVBoxLayout(tiling_box)
        tiling_layout.setContentsMargins(6, 6, 6, 6)
        tiling_layout.setSpacing(6)

        self.lbl_tiling_status = QLabel("Tiling: Disabled (Single Pattern Mode)")
        self.lbl_tiling_status.setStyleSheet("font-weight: bold;")
        tiling_layout.addWidget(self.lbl_tiling_status)

        self.lbl_tiling_details = QLabel("Standard single-shot exposure mode.")
        self.lbl_tiling_details.setStyleSheet("font-size: 11px;")
        tiling_layout.addWidget(self.lbl_tiling_details)

        # Tile Navigation Row
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Active Tile:"))

        self.btn_prev_tile = QPushButton("◀ Prev")
        self.btn_prev_tile.clicked.connect(self._on_prev_tile_clicked)
        nav_row.addWidget(self.btn_prev_tile)

        self.spin_tile_index = QSpinBox()
        self.spin_tile_index.setRange(0, 0)
        self.spin_tile_index.valueChanged.connect(self._on_tile_index_changed)
        nav_row.addWidget(self.spin_tile_index)

        self.lbl_tile_count = QLabel("of 1")
        nav_row.addWidget(self.lbl_tile_count)

        self.btn_next_tile = QPushButton("Next ▶")
        self.btn_next_tile.clicked.connect(self._on_next_tile_clicked)
        nav_row.addWidget(self.btn_next_tile)

        tiling_layout.addLayout(nav_row)

        # Tile Preview
        preview_row = QHBoxLayout()
        self.lbl_tile_preview = QLabel("No Tile")
        self.lbl_tile_preview.setFixedSize(160, 90)
        self.lbl_tile_preview.setAlignment(Qt.AlignCenter)
        preview_row.addWidget(self.lbl_tile_preview)

        preview_actions = QVBoxLayout()
        self.btn_regenerate_tiles = QPushButton("Generate Tiles")
        self.btn_regenerate_tiles.clicked.connect(self._on_regenerate_tiles_clicked)
        preview_actions.addWidget(self.btn_regenerate_tiles)
        preview_actions.addStretch()
        preview_row.addLayout(preview_actions)

        preview_row.addStretch()
        tiling_layout.addLayout(preview_row)

        layout.addWidget(tiling_box)
        layout.addStretch()

    def _setup_overrides_tab(self):
        layout = QVBoxLayout(self.tab_overrides)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Exposure override
        exp_box = QGroupBox("Exposure Override")
        exp_form = QFormLayout(exp_box)

        self.chk_override_exp = QCheckBox("Override Project Exposure Duration")
        self.chk_override_exp.toggled.connect(self._on_override_exp_toggled)
        exp_form.addRow(self.chk_override_exp)

        self.spin_override_exp = QSpinBox()
        self.spin_override_exp.setRange(100, 600000)
        self.spin_override_exp.setValue(8000)
        self.spin_override_exp.setSingleStep(500)
        self.spin_override_exp.setSuffix(" ms")
        self.spin_override_exp.setEnabled(False)
        self.spin_override_exp.valueChanged.connect(self._on_override_exp_value_changed)
        exp_form.addRow("Duration:", self.spin_override_exp)
        layout.addWidget(exp_box)

        # Tiling override
        tile_box = QGroupBox("Tiling Override")
        tile_form = QFormLayout(tile_box)

        self.chk_override_tiling = QCheckBox("Override Project Tiling Settings")
        self.chk_override_tiling.toggled.connect(self._on_override_tiling_toggled)
        tile_form.addRow(self.chk_override_tiling)

        self.chk_override_tiling_enable = QCheckBox("Enable Tiling for this Layer")
        self.chk_override_tiling_enable.setEnabled(False)
        self.chk_override_tiling_enable.toggled.connect(self._on_override_tiling_enable_toggled)
        tile_form.addRow(self.chk_override_tiling_enable)

        layout.addWidget(tile_box)

        # Effective summary box
        summary_box = QGroupBox("Effective Resolved Settings")
        summary_form = QFormLayout(summary_box)
        self.lbl_eff_exp = QLabel("8000 ms")
        self.lbl_eff_exp.setStyleSheet("font-weight: bold;")
        summary_form.addRow("Effective Exposure:", self.lbl_eff_exp)

        self.lbl_eff_tiling = QLabel("Disabled")
        self.lbl_eff_tiling.setStyleSheet("font-weight: bold;")
        summary_form.addRow("Effective Tiling:", self.lbl_eff_tiling)
        layout.addWidget(summary_box)

        layout.addStretch()

    def _refresh_layer_view(self):
        layer = self.engine.project.active_layer
        effective = self.engine.project.settings.with_overrides(layer.overrides)

        # File & thumbnail
        if layer.pattern_path:
            self.lbl_pattern_file.setText(Path(layer.pattern_path).name)
            self._load_thumbnail(layer.pattern_path)
        else:
            self.lbl_pattern_file.setText("No pattern selected")
            self.lbl_thumb.clear()
            self.lbl_thumb.setText("Thumbnail")

        # Tiling preview
        if effective.tiling_enabled:
            self.lbl_tiling_status.setText("Tiling: Enabled (Step-and-Repeat Active)")
            self.lbl_tiling_details.setText(
                f"Tile: {effective.tile_width}x{effective.tile_height} px | "
                f"Pitch: dx={effective.pitch_x}µm, dy={effective.pitch_y}µm"
            )
            self.btn_regenerate_tiles.setEnabled(bool(layer.pattern_path))
        else:
            self.lbl_tiling_status.setText("Tiling: Disabled (Single Pattern Mode)")
            self.lbl_tiling_details.setText("Standard single-shot exposure will be used.")
            self.btn_regenerate_tiles.setEnabled(bool(layer.pattern_path))

        # Pattern threshold UI
        self.spin_threshold.blockSignals(True)
        self.spin_threshold.setValue(layer.threshold)
        self.spin_threshold.blockSignals(False)

        # Pattern scaling UI
        self.spin_scale_w.blockSignals(True)
        self.spin_scale_h.blockSignals(True)
        self.spin_scale_w.setValue(layer.scale_w)
        self.spin_scale_h.setValue(layer.scale_h)
        self.spin_scale_w.blockSignals(False)
        self.spin_scale_h.blockSignals(False)

        # Overrides UI
        self.chk_override_exp.blockSignals(True)
        has_exp_override = layer.overrides.exposure_time is not None
        self.chk_override_exp.setChecked(has_exp_override)
        self.spin_override_exp.setEnabled(has_exp_override)
        if has_exp_override:
            self.spin_override_exp.setValue(int(layer.overrides.exposure_time))
        self.chk_override_exp.blockSignals(False)

        self.chk_override_tiling.blockSignals(True)
        has_tile_override = layer.overrides.tiling_enabled is not None
        self.chk_override_tiling.setChecked(has_tile_override)
        self.chk_override_tiling_enable.setEnabled(has_tile_override)
        if has_tile_override:
            self.chk_override_tiling_enable.setChecked(bool(layer.overrides.tiling_enabled))
        self.chk_override_tiling.blockSignals(False)

        # Summaries
        self.lbl_eff_exp.setText(
            f"{int(effective.exposure_time)} ms "
            f"({'Overridden' if has_exp_override else 'Default'})"
        )
        self.lbl_eff_tiling.setText(
            f"{'Enabled' if effective.tiling_enabled else 'Disabled'} "
            f"({'Overridden' if has_tile_override else 'Default'})"
        )

    def _refresh_tile_preview(self):
        layer = self.engine.project.active_layer
        tile_count = len(layer._tile_cache)
        active_idx = self.engine.project.active_tile_index

        self.spin_tile_index.blockSignals(True)
        self.spin_tile_index.setMaximum(max(0, tile_count - 1))
        self.spin_tile_index.setValue(min(active_idx, max(0, tile_count - 1)))
        self.spin_tile_index.blockSignals(False)

        self.lbl_tile_count.setText(f"of {max(1, tile_count)}")
        self.btn_prev_tile.setEnabled(active_idx > 0)
        self.btn_next_tile.setEnabled(active_idx < max(0, tile_count - 1))

        # Render preview for the active tile
        layer = self.engine.project.active_layer
        tile_data = layer.get_tile(active_idx)
        tile = tile_data[0] if tile_data is not None else None
        if tile is not None and isinstance(tile, np.ndarray):
            th, tw = tile.shape[:2]
            scale = min(160 / max(1, tw), 90 / max(1, th))
            nw = max(1, int(tw * scale))
            nh = max(1, int(th * scale))
            thumb = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_LINEAR)
            contig = np.ascontiguousarray(thumb)
            if contig.ndim == 2:
                qimg = QImage(contig.data, nw, nh, nw, QImage.Format_Grayscale8)
            elif contig.shape[2] == 3:
                qimg = QImage(contig.data, nw, nh, 3 * nw, QImage.Format_BGR888)
            elif contig.shape[2] == 4:
                qimg = QImage(contig.data, nw, nh, 4 * nw, QImage.Format_RGBA8888)
            else:
                qimg = None
            if qimg is not None:
                self.lbl_tile_preview.setPixmap(QPixmap.fromImage(qimg))
            else:
                self.lbl_tile_preview.clear()
        else:
            self.lbl_tile_preview.clear()
            self.lbl_tile_preview.setText("No Tile")

    def _on_tile_index_changed(self, val: int):
        self.engine.project.select_tile(val)

    def _on_prev_tile_clicked(self):
        cur = self.engine.project.active_tile_index
        if cur > 0:
            self.engine.project.select_tile(cur - 1)

    def _on_next_tile_clicked(self):
        cur = self.engine.project.active_tile_index
        max_idx = max(0, self.engine.project.get_active_layer_tile_count() - 1)
        if cur < max_idx:
            self.engine.project.select_tile(cur + 1)

    def _on_threshold_changed(self):
        layer = self.engine.project.active_layer
        layer.set_threshold(self.spin_threshold.value())
        if layer.pattern_path:
            self._load_thumbnail(layer.pattern_path)

    def _on_scale_changed(self):
        layer = self.engine.project.active_layer
        layer.scale_w = self.spin_scale_w.value()
        layer.scale_h = self.spin_scale_h.value()
        layer._pattern_cache = None
        layer._tile_cache = []
        layer._tile_coords = []

    def _on_match_projector_clicked(self):
        if hasattr(self.engine, "projector") and self.engine.projector is not None:
            pw, ph = self.engine.projector.projector_size()
            self.spin_scale_w.setValue(pw)
            self.spin_scale_h.setValue(ph)
            self.bridge.status_message.emit(f"Set scale size to projector resolution: {pw}x{ph}")

    def _on_regenerate_tiles_clicked(self):
        layer = self.engine.project.active_layer
        layer.generate_tiles(force=True)
        self.bridge.status_message.emit("Generated tiles for active layer.")

    def _load_thumbnail(self, path: str):
        if os.path.exists(path):
            try:
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    if raw.ndim == 2:
                        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    elif raw.shape[2] == 4:
                        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

                    layer = self.engine.project.active_layer
                    if layer.threshold != -1:
                        mask = np.any(raw[..., :3] > layer.threshold, axis=-1)
                        bw = np.zeros_like(raw)
                        bw[mask] = 255
                        raw = bw

                    ih, iw = raw.shape[:2]
                    scale = min(110 / max(1, iw), 80 / max(1, ih))
                    nw = max(1, int(iw * scale))
                    nh = max(1, int(ih * scale))
                    thumb = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_NEAREST if layer.threshold != -1 else cv2.INTER_LINEAR)
                    contig = np.ascontiguousarray(thumb)
                    qimg = QImage(contig.data, nw, nh, 3 * nw, QImage.Format_BGR888)
                    self.lbl_thumb.setPixmap(QPixmap.fromImage(qimg))
                else:
                    self.lbl_thumb.setText("Preview Error")
            except Exception:
                self.lbl_thumb.setText("Preview Error")

    def _on_load_pattern_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Pattern", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            layer = self.engine.project.active_layer
            layer.set_pattern_path(filename)
            self.engine.event_bus.emit(Event.PROJECT_CHANGED, self.engine.project)
            self.bridge.status_message.emit(f"Loaded mask: {Path(filename).name}")

    def _on_override_exp_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        self.spin_override_exp.setEnabled(checked)
        val = float(self.spin_override_exp.value()) if checked else None
        layer.set_exposure_override(val)

    def _on_override_exp_value_changed(self, val: int):
        layer = self.engine.project.active_layer
        if self.chk_override_exp.isChecked():
            layer.set_exposure_override(float(val))

    def _on_override_tiling_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        self.chk_override_tiling_enable.setEnabled(checked)
        val = self.chk_override_tiling_enable.isChecked() if checked else None
        layer.set_tiling_override(val)

    def _on_override_tiling_enable_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        if self.chk_override_tiling.isChecked():
            layer.set_tiling_override(checked)

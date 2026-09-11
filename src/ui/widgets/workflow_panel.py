import os
from pathlib import Path
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.chip_project import ChipLayer, PatterningSettings
from core.engine import StepperEngine
from core.events import Event
from operations import ExposureOperation, TiledExposureOperation
from operations.tiling import split_image_with_overlap
from ui.bridge import QtEngineBridge


class WorkflowPanelWidget(QWidget):
    """Main workflow dock container using a horizontal QSplitter: (project | layer | action)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        # Horizontal Splitter: [project | layer | action]
        self.splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(self.splitter)

        # 1. Project Portion
        self.project_subpanel = ProjectSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.project_subpanel)

        # 2. Layer Portion
        self.layer_subpanel = LayerSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.layer_subpanel)

        # 3. Action Portion
        self.action_subpanel = ActionSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.action_subpanel)

        # Set default stretch / proportional widths
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 3)
        self.splitter.setSizes([320, 420, 320])


# =============================================================================
# Portion 1: Chip Project
# =============================================================================
class ProjectSubpanelWidget(QTabWidget):
    """Left-most portion: Project level layer selection & global settings."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        # Tab 1: Chip Layers
        self.tab_layers = QWidget()
        self._setup_layers_tab()
        self.addTab(self.tab_layers, "Chip Layers")

        # Tab 2: Global Settings
        self.tab_settings = QWidget()
        self._setup_settings_tab()
        self.addTab(self.tab_settings, "Project Settings")

        # Signals
        self.bridge.project_changed.connect(lambda _: self._refresh_ui())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_ui())
        self.bridge.active_layer_changed.connect(lambda _: self._refresh_layer_selection())
        self._refresh_ui()

    def _setup_layers_tab(self):
        layout = QVBoxLayout(self.tab_layers)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Project file buttons row
        file_row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new_project)
        file_row.addWidget(self.btn_new)

        self.btn_load = QPushButton("Load...")
        self.btn_load.clicked.connect(self._on_load_project)
        file_row.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save...")
        self.btn_save.clicked.connect(self._on_save_project)
        file_row.addWidget(self.btn_save)
        layout.addLayout(file_row)

        # Layer management row
        layer_btn_row = QHBoxLayout()
        self.btn_add_layer = QPushButton("+ Add Layer")
        self.btn_add_layer.clicked.connect(self._on_add_layer)
        layer_btn_row.addWidget(self.btn_add_layer)

        self.btn_remove_layer = QPushButton("- Remove Layer")
        self.btn_remove_layer.clicked.connect(self._on_remove_layer)
        layer_btn_row.addWidget(self.btn_remove_layer)
        layout.addLayout(layer_btn_row)

        # Layers table
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(3)
        self.layers_table.setHorizontalHeaderLabels(["Layer Name", "Mask", "Runs"])
        self.layers_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.layers_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.layers_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.layers_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.layers_table.setSelectionMode(QTableWidget.SingleSelection)
        self.layers_table.itemSelectionChanged.connect(self._on_layer_row_selected)
        layout.addWidget(self.layers_table)

    def _setup_settings_tab(self):
        layout = QVBoxLayout(self.tab_settings)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Exposure defaults
        exp_group = QGroupBox("Exposure Defaults")
        exp_form = QFormLayout(exp_group)
        self.spin_default_exp = QSpinBox()
        self.spin_default_exp.setRange(100, 600000)
        self.spin_default_exp.setValue(int(self.engine.project.settings.exposure_time))
        self.spin_default_exp.setSingleStep(500)
        self.spin_default_exp.setSuffix(" ms")
        self.spin_default_exp.valueChanged.connect(self._on_default_exp_changed)
        exp_form.addRow("Default Exposure:", self.spin_default_exp)
        layout.addWidget(exp_group)

        # Tiling defaults
        tile_group = QGroupBox("Tiling Defaults")
        tile_form = QFormLayout(tile_group)

        self.chk_default_tiling = QCheckBox("Enable Tiling by Default")
        self.chk_default_tiling.setChecked(self.engine.project.settings.tiling_enabled)
        self.chk_default_tiling.toggled.connect(self._on_default_tiling_toggled)
        tile_form.addRow(self.chk_default_tiling)

        self.spin_tile_w = QSpinBox()
        self.spin_tile_w.setRange(100, 10000)
        self.spin_tile_w.setValue(self.engine.project.settings.tile_width)
        self.spin_tile_w.valueChanged.connect(
            lambda val: (
                setattr(self.engine.project.settings, "tile_width", val),
                self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED),
            )
        )
        tile_form.addRow("Tile Width (px):", self.spin_tile_w)

        self.spin_tile_h = QSpinBox()
        self.spin_tile_h.setRange(100, 10000)
        self.spin_tile_h.setValue(self.engine.project.settings.tile_height)
        self.spin_tile_h.valueChanged.connect(
            lambda val: (
                setattr(self.engine.project.settings, "tile_height", val),
                self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED),
            )
        )
        tile_form.addRow("Tile Height (px):", self.spin_tile_h)

        self.spin_pitch_x = QDoubleSpinBox()
        self.spin_pitch_x.setRange(10.0, 50000.0)
        self.spin_pitch_x.setValue(self.engine.project.settings.pitch_x)
        self.spin_pitch_x.setSuffix(" µm")
        self.spin_pitch_x.valueChanged.connect(
            lambda val: (
                setattr(self.engine.project.settings, "pitch_x", val),
                self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED),
            )
        )
        tile_form.addRow("Pitch X Offset:", self.spin_pitch_x)

        self.spin_pitch_y = QDoubleSpinBox()
        self.spin_pitch_y.setRange(10.0, 50000.0)
        self.spin_pitch_y.setValue(self.engine.project.settings.pitch_y)
        self.spin_pitch_y.setSuffix(" µm")
        self.spin_pitch_y.valueChanged.connect(
            lambda val: (
                setattr(self.engine.project.settings, "pitch_y", val),
                self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED),
            )
        )
        tile_form.addRow("Pitch Y Offset:", self.spin_pitch_y)

        layout.addWidget(tile_group)
        layout.addStretch()

    def _refresh_ui(self):
        self.layers_table.blockSignals(True)
        layers = self.engine.project.layers
        self.layers_table.setRowCount(len(layers))

        for i, layer in enumerate(layers):
            item_name = QTableWidgetItem(layer.name)
            item_mask = QTableWidgetItem("✓ Loaded" if layer.pattern_path else "None")
            item_runs = QTableWidgetItem(str(len(layer.exposures)))
            self.layers_table.setItem(i, 0, item_name)
            self.layers_table.setItem(i, 1, item_mask)
            self.layers_table.setItem(i, 2, item_runs)

        active_idx = self.engine.project.active_layer_index
        if 0 <= active_idx < len(layers):
            self.layers_table.selectRow(active_idx)

        self.btn_remove_layer.setEnabled(len(layers) > 1)
        self.layers_table.blockSignals(False)

        # Update settings inputs
        s = self.engine.project.settings
        self.spin_default_exp.blockSignals(True)
        self.spin_default_exp.setValue(int(s.exposure_time))
        self.spin_default_exp.blockSignals(False)

    def _refresh_layer_selection(self):
        active_idx = self.engine.project.active_layer_index
        if 0 <= active_idx < self.layers_table.rowCount():
            self.layers_table.blockSignals(True)
            self.layers_table.selectRow(active_idx)
            self.layers_table.blockSignals(False)

    def _on_layer_row_selected(self):
        selected_rows = self.layers_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            self.engine.project.select_layer(row)

    def _on_add_layer(self):
        self.engine.project.add_layer()

    def _on_remove_layer(self):
        idx = self.engine.project.active_layer_index
        if len(self.engine.project.layers) <= 1:
            self.bridge.warning_emitted.emit("Cannot delete the last remaining layer.")
            return
        self.engine.project.remove_layer(idx)

    def _on_new_project(self):
        self.engine.new_project()
        self.bridge.status_message.emit("Created new chip project")

    def _on_load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Chip Project", "", "JSON (*.json)")
        if path:
            self.engine.load_project(path)
            self.bridge.status_message.emit(f"Loaded project: {Path(path).name}")

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Chip Project", "", "JSON (*.json)")
        if path:
            self.engine.save_project(path)
            self.bridge.status_message.emit(f"Saved project: {Path(path).name}")

    def _on_default_exp_changed(self, val: int):
        self.engine.project.settings.exposure_time = float(val)
        self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def _on_default_tiling_toggled(self, checked: bool):
        self.engine.project.settings.tiling_enabled = checked
        self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)


# =============================================================================
# Portion 2: Layer Panel
# =============================================================================
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
        self.bridge.project_changed.connect(lambda _: self._refresh_layer_view())
        self.bridge.active_layer_changed.connect(lambda _: self._refresh_layer_view())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_layer_view())
        self.bridge.projector_image_changed.connect(lambda _: self._refresh_layer_view())
        self._refresh_layer_view()

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
        self.lbl_pattern_file.setStyleSheet("color: #9ca3af; font-style: italic;")
        file_row.addWidget(self.lbl_pattern_file, stretch=1)
        layout.addLayout(file_row)

        # Thumbnail and offset adjustments
        mid_row = QHBoxLayout()
        self.lbl_thumb = QLabel("Thumbnail")
        self.lbl_thumb.setFixedSize(110, 80)
        self.lbl_thumb.setStyleSheet("background-color: #09090b; border: 1px solid #3f3f46;")
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        mid_row.addWidget(self.lbl_thumb)

        adj_group = QGroupBox("Fine Align Offsets")
        adj_form = QFormLayout(adj_group)
        adj_form.setContentsMargins(4, 4, 4, 4)
        adj_form.setSpacing(4)

        self.spin_shift_x = QDoubleSpinBox()
        self.spin_shift_x.setRange(-2000.0, 2000.0)
        self.spin_shift_x.setValue(0.0)
        self.spin_shift_x.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Shift X (px):", self.spin_shift_x)

        self.spin_shift_y = QDoubleSpinBox()
        self.spin_shift_y.setRange(-2000.0, 2000.0)
        self.spin_shift_y.setValue(0.0)
        self.spin_shift_y.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Shift Y (px):", self.spin_shift_y)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(-180.0, 180.0)
        self.spin_theta.setValue(0.0)
        self.spin_theta.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Rotation θ (°):", self.spin_theta)

        mid_row.addWidget(adj_group)
        layout.addLayout(mid_row)

        # Tiling preview & slicing section
        tiling_box = QGroupBox("Tiling Preview & Slicing")
        tiling_layout = QVBoxLayout(tiling_box)
        tiling_layout.setContentsMargins(6, 6, 6, 6)
        tiling_layout.setSpacing(4)

        self.lbl_tiling_status = QLabel("Tiling: Disabled (Single Pattern Mode)")
        self.lbl_tiling_status.setStyleSheet("font-weight: bold; color: #38bdf8;")
        tiling_layout.addWidget(self.lbl_tiling_status)

        self.lbl_tiling_details = QLabel("Project default tiling is inactive.")
        self.lbl_tiling_details.setStyleSheet("color: #9ca3af; font-size: 11px;")
        tiling_layout.addWidget(self.lbl_tiling_details)

        self.btn_segment = QPushButton("Segment Large Image into Tiles")
        self.btn_segment.clicked.connect(self._on_segment_clicked)
        tiling_layout.addWidget(self.btn_segment)

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
        self.lbl_eff_exp.setStyleSheet("font-weight: bold; color: #10b981;")
        summary_form.addRow("Effective Exposure:", self.lbl_eff_exp)

        self.lbl_eff_tiling = QLabel("Disabled")
        self.lbl_eff_tiling.setStyleSheet("font-weight: bold; color: #10b981;")
        summary_form.addRow("Effective Tiling:", self.lbl_eff_tiling)
        layout.addWidget(summary_box)

        layout.addStretch()

    def _refresh_layer_view(self):
        layer = self.engine.project.active_layer
        effective = layer.get_effective_settings(self.engine.project.settings)

        # File & thumbnail
        if layer.pattern_path:
            self.lbl_pattern_file.setText(Path(layer.pattern_path).name)
            self._load_thumbnail(layer.pattern_path)
        else:
            self.lbl_pattern_file.setText("No pattern selected")
            self.lbl_thumb.clear()
            self.lbl_thumb.setText("Thumbnail")

        # Offsets
        self.spin_shift_x.blockSignals(True)
        self.spin_shift_y.blockSignals(True)
        self.spin_theta.blockSignals(True)
        self.spin_shift_x.setValue(layer.image_adjust[0])
        self.spin_shift_y.setValue(layer.image_adjust[1])
        self.spin_theta.setValue(layer.image_adjust[2])
        self.spin_shift_x.blockSignals(False)
        self.spin_shift_y.blockSignals(False)
        self.spin_theta.blockSignals(False)

        # Tiling preview
        if effective.tiling_enabled:
            self.lbl_tiling_status.setText("Tiling: Enabled (Step-and-Repeat Active)")
            self.lbl_tiling_details.setText(
                f"Tile: {effective.tile_width}x{effective.tile_height} px | "
                f"Pitch: dx={effective.pitch_x}µm, dy={effective.pitch_y}µm"
            )
            self.btn_segment.setEnabled(bool(layer.pattern_path))
        else:
            self.lbl_tiling_status.setText("Tiling: Disabled (Single Pattern Mode)")
            self.lbl_tiling_details.setText("Standard single-shot exposure will be used.")
            self.btn_segment.setEnabled(False)

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

    def _load_thumbnail(self, path: str):
        if os.path.exists(path):
            try:
                img = Image.open(path)
                thumb = img.resize((110, 80))
                if thumb.mode != "RGBA":
                    thumb = thumb.convert("RGBA")
                data = thumb.tobytes("raw", "RGBA")
                qimg = QImage(data, 110, 80, QImage.Format_RGBA8888)
                self.lbl_thumb.setPixmap(QPixmap.fromImage(qimg))
            except Exception:
                self.lbl_thumb.setText("Preview Error")

    def _on_load_pattern_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Pattern", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            layer = self.engine.project.active_layer
            layer.pattern_path = filename
            layer.invalidate_render_cache()
            self.bridge.status_message.emit(f"Loaded mask: {Path(filename).name}")

    def _on_adjust_changed(self):
        layer = self.engine.project.active_layer
        layer.image_adjust = (
            self.spin_shift_x.value(),
            self.spin_shift_y.value(),
            self.spin_theta.value(),
        )
        layer.invalidate_render_cache()

    def _on_segment_clicked(self):
        layer = self.engine.project.active_layer
        effective = layer.get_effective_settings(self.engine.project.settings)
        if not layer.pattern_path or not os.path.exists(layer.pattern_path):
            self.bridge.warning_emitted.emit("Please select a valid pattern image first.")
            return

        try:
            nx, ny, total, (w, h) = split_image_with_overlap(
                image_path=layer.pattern_path,
                tile_width=effective.tile_width,
                tile_height=effective.tile_height,
                overlap_x=effective.overlap_x,
                overlap_y=effective.overlap_y,
                output_dir="tiles",
            )
            self.bridge.status_message.emit(f"Generated {total} tiles in 'tiles/' ({nx}x{ny})")
        except Exception as e:
            self.bridge.warning_emitted.emit(f"Segmentation failed: {e}")

    def _on_override_exp_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        self.spin_override_exp.setEnabled(checked)
        if checked:
            layer.overrides.exposure_time = float(self.spin_override_exp.value())
        else:
            layer.overrides.exposure_time = None
        self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def _on_override_exp_value_changed(self, val: int):
        layer = self.engine.project.active_layer
        if self.chk_override_exp.isChecked():
            layer.overrides.exposure_time = float(val)
            self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def _on_override_tiling_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        self.chk_override_tiling_enable.setEnabled(checked)
        if checked:
            layer.overrides.tiling_enabled = self.chk_override_tiling_enable.isChecked()
        else:
            layer.overrides.tiling_enabled = None
        self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)

    def _on_override_tiling_enable_toggled(self, checked: bool):
        layer = self.engine.project.active_layer
        if self.chk_override_tiling.isChecked():
            layer.overrides.tiling_enabled = checked
            self.engine.events.emit(Event.EXPOSURE_CONFIG_CHANGED)


# =============================================================================
# Portion 3: Action Execution Dashboard
# =============================================================================
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
        self.lbl_active_name.setStyleSheet("font-weight: bold; font-size: 13px; color: #f4f4f5;")
        card_layout.addRow("Active Target:", self.lbl_active_name)

        self.lbl_active_exp = QLabel("8000 ms")
        self.lbl_active_exp.setStyleSheet("font-weight: bold; color: #38bdf8;")
        card_layout.addRow("Exposure:", self.lbl_active_exp)

        self.lbl_active_mode = QLabel("Single Field Exposure")
        self.lbl_active_mode.setStyleSheet("font-weight: bold; color: #a1a1aa;")
        card_layout.addRow("Execution Mode:", self.lbl_active_mode)

        layout.addWidget(card)

        # Big trigger action button
        self.btn_expose = QPushButton("Expose Layer")
        self.btn_expose.setStyleSheet(
            """
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:disabled {
                background-color: #3f3f46;
                color: #71717a;
            }
            """
        )
        self.btn_expose.clicked.connect(self._on_expose_clicked)
        layout.addWidget(self.btn_expose)

        # History table
        hist_box = QGroupBox("Layer Exposure History")
        hist_layout = QVBoxLayout(hist_box)
        hist_layout.setContentsMargins(4, 4, 4, 4)

        self.table_history = QTableWidget()
        self.table_history.setColumnCount(3)
        self.table_history.setHorizontalHeaderLabels(["Timestamp", "Duration", "Status"])
        self.table_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_history.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hist_layout.addWidget(self.table_history)

        layout.addWidget(hist_box)

        # Connect signals
        self.bridge.project_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.active_layer_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.exposure_config_changed.connect(lambda: self._refresh_dashboard())
        self.bridge.projector_image_changed.connect(lambda _: self._refresh_dashboard())
        self.bridge.operation_started.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_finished.connect(lambda *_: self._update_lock_state())
        self.bridge.operation_aborted.connect(lambda *_: self._update_lock_state())
        self._refresh_dashboard()
        self._update_lock_state()

    def _refresh_dashboard(self):
        layer = self.engine.project.active_layer
        effective = layer.get_effective_settings(self.engine.project.settings)

        self.lbl_active_name.setText(layer.name)
        self.lbl_active_exp.setText(f"{int(effective.exposure_time)} ms")

        if effective.tiling_enabled:
            self.lbl_active_mode.setText("Step-and-Repeat Tiling")
            self.btn_expose.setText("Run Tiled Exposure")
        else:
            self.lbl_active_mode.setText("Single Field Exposure")
            self.btn_expose.setText("Expose Layer")

        # History table
        self.table_history.setRowCount(len(layer.exposures))
        for i, exp in enumerate(reversed(layer.exposures)):
            time_str = exp.time.strftime("%H:%M:%S")
            dur_str = f"{int(exp.duration)}ms"
            status_str = "Aborted" if exp.aborted else "Success"
            self.table_history.setItem(i, 0, QTableWidgetItem(time_str))
            self.table_history.setItem(i, 1, QTableWidgetItem(dur_str))
            item_status = QTableWidgetItem(status_str)
            if exp.aborted:
                item_status.setForeground(Qt.red)
            else:
                item_status.setForeground(Qt.green)
            self.table_history.setItem(i, 2, item_status)

    def _on_expose_clicked(self):
        layer_idx = self.engine.project.active_layer_index
        layer = self.engine.project.active_layer
        effective = layer.get_effective_settings(self.engine.project.settings)

        if effective.tiling_enabled:
            op = TiledExposureOperation(layer_index=layer_idx, settings=effective)
        else:
            op = ExposureOperation(layer_index=layer_idx, settings=effective)

        self.bridge.start_operation(op)

    def _update_lock_state(self, *args):
        is_busy = self.engine.operations.current_operation is not None
        self.btn_expose.setEnabled(not is_busy)

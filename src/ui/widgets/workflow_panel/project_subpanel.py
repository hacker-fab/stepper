from pathlib import Path
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


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
        self.bridge.exposure_history_changed.connect(lambda _: self._refresh_ui())
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
        self.spin_tile_w.valueChanged.connect(self._on_tile_w_changed)
        tile_form.addRow("Tile Width (px):", self.spin_tile_w)

        self.spin_tile_h = QSpinBox()
        self.spin_tile_h.setRange(100, 10000)
        self.spin_tile_h.setValue(self.engine.project.settings.tile_height)
        self.spin_tile_h.valueChanged.connect(self._on_tile_h_changed)
        tile_form.addRow("Tile Height (px):", self.spin_tile_h)

        self.spin_pitch_x = QDoubleSpinBox()
        self.spin_pitch_x.setRange(10.0, 50000.0)
        self.spin_pitch_x.setValue(self.engine.project.settings.pitch_x)
        self.spin_pitch_x.setSuffix(" µm")
        self.spin_pitch_x.valueChanged.connect(self._on_pitch_x_changed)
        tile_form.addRow("Pitch X Offset:", self.spin_pitch_x)

        self.spin_pitch_y = QDoubleSpinBox()
        self.spin_pitch_y.setRange(10.0, 50000.0)
        self.spin_pitch_y.setValue(self.engine.project.settings.pitch_y)
        self.spin_pitch_y.setSuffix(" µm")
        self.spin_pitch_y.valueChanged.connect(self._on_pitch_y_changed)
        tile_form.addRow("Pitch Y Offset:", self.spin_pitch_y)

        layout.addWidget(tile_group)
        layout.addStretch()

    def _refresh_ui(self):
        self.layers_table.blockSignals(True)
        layers = self.engine.project.layers
        self.layers_table.setRowCount(len(layers))
        exp_history = getattr(self.engine.project, "exposure_history", [])

        for i, layer in enumerate(layers):
            item_name = QTableWidgetItem(layer.name)
            item_mask = QTableWidgetItem("✓ Loaded" if layer.pattern_path else "None")
            runs = sum(1 for e in exp_history if getattr(e, "layer_index", None) == i)
            item_runs = QTableWidgetItem(str(runs))
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

        self.chk_default_tiling.blockSignals(True)
        self.chk_default_tiling.setChecked(s.tiling_enabled)
        self.chk_default_tiling.blockSignals(False)

        self.spin_tile_w.blockSignals(True)
        self.spin_tile_w.setValue(s.tile_width)
        self.spin_tile_w.blockSignals(False)

        self.spin_tile_h.blockSignals(True)
        self.spin_tile_h.setValue(s.tile_height)
        self.spin_tile_h.blockSignals(False)

        self.spin_pitch_x.blockSignals(True)
        self.spin_pitch_x.setValue(s.pitch_x)
        self.spin_pitch_x.blockSignals(False)

        self.spin_pitch_y.blockSignals(True)
        self.spin_pitch_y.setValue(s.pitch_y)
        self.spin_pitch_y.blockSignals(False)

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
        self.engine.project.update_settings(exposure_time=float(val))

    def _on_default_tiling_toggled(self, checked: bool):
        self.engine.project.update_settings(tiling_enabled=checked)

    def _on_tile_w_changed(self, val: int):
        self.engine.project.update_settings(tile_width=val)

    def _on_tile_h_changed(self, val: int):
        self.engine.project.update_settings(tile_height=val)

    def _on_pitch_x_changed(self, val: float):
        self.engine.project.update_settings(pitch_x=val)

    def _on_pitch_y_changed(self, val: float):
        self.engine.project.update_settings(pitch_y=val)

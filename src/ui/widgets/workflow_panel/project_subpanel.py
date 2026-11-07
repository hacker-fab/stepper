from pathlib import Path
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
        self.bridge.chip_progress_changed.connect(lambda _: self._refresh_ui())
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

        # Chip progress file buttons row
        prog_row = QHBoxLayout()
        self.btn_reset_progress = QPushButton("Reset Progress")
        self.btn_reset_progress.setToolTip("Reset chip markers, origins, and exposure history")
        self.btn_reset_progress.clicked.connect(self._on_reset_progress)
        prog_row.addWidget(self.btn_reset_progress)

        self.btn_load_progress = QPushButton("Load Progress...")
        self.btn_load_progress.clicked.connect(self._on_load_progress)
        prog_row.addWidget(self.btn_load_progress)

        self.btn_save_progress = QPushButton("Save Progress...")
        self.btn_save_progress.clicked.connect(self._on_save_progress)
        prog_row.addWidget(self.btn_save_progress)
        layout.addLayout(prog_row)

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

        # Tiling Overlap defaults
        tile_group = QGroupBox("Tiling Overlap (Chip Space)")
        tile_form = QFormLayout(tile_group)

        self.spin_overlap_x = QDoubleSpinBox()
        self.spin_overlap_x.setRange(0.0, 10000.0)
        self.spin_overlap_x.setDecimals(2)
        self.spin_overlap_x.setSingleStep(1.0)
        self.spin_overlap_x.setValue(self.engine.project.settings.overlap_x)
        self.spin_overlap_x.setSuffix(" µm")
        self.spin_overlap_x.valueChanged.connect(self._on_overlap_x_changed)
        tile_form.addRow("Overlap X:", self.spin_overlap_x)

        self.spin_overlap_y = QDoubleSpinBox()
        self.spin_overlap_y.setRange(0.0, 10000.0)
        self.spin_overlap_y.setDecimals(2)
        self.spin_overlap_y.setSingleStep(1.0)
        self.spin_overlap_y.setValue(self.engine.project.settings.overlap_y)
        self.spin_overlap_y.setSuffix(" µm")
        self.spin_overlap_y.valueChanged.connect(self._on_overlap_y_changed)
        tile_form.addRow("Overlap Y:", self.spin_overlap_y)

        layout.addWidget(tile_group)

        # Pattern Physical Size (Project-wide setting for all layers)
        size_group = QGroupBox("Pattern Physical Size (µm)")
        size_layout = QVBoxLayout(size_group)
        size_layout.setContentsMargins(6, 6, 6, 6)
        size_layout.setSpacing(6)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width:"))
        self.spin_width_um = QDoubleSpinBox()
        self.spin_width_um.setRange(-1.0, 100000.0)
        self.spin_width_um.setDecimals(1)
        self.spin_width_um.setSingleStep(10.0)
        self.spin_width_um.setValue(self.engine.project.settings.width_um)
        self.spin_width_um.setSuffix(" µm")
        self.spin_width_um.setToolTip("Target pattern width in µm (-1 for Fullscreen / Projector FOV)")
        self.spin_width_um.valueChanged.connect(self._on_pattern_size_changed)
        size_row.addWidget(self.spin_width_um)

        size_row.addWidget(QLabel("Height:"))
        self.spin_height_um = QDoubleSpinBox()
        self.spin_height_um.setRange(-1.0, 100000.0)
        self.spin_height_um.setDecimals(1)
        self.spin_height_um.setSingleStep(10.0)
        self.spin_height_um.setValue(self.engine.project.settings.height_um)
        self.spin_height_um.setSuffix(" µm")
        self.spin_height_um.setToolTip("Target pattern height in µm (-1 for Fullscreen / Projector FOV)")
        self.spin_height_um.valueChanged.connect(self._on_pattern_size_changed)
        size_row.addWidget(self.spin_height_um)
        size_layout.addLayout(size_row)

        btn_row = QHBoxLayout()
        self.btn_fullscreen = QPushButton("Fullscreen (-1)")
        self.btn_fullscreen.setToolTip("Set both dimensions to -1 for quick single-tile fullscreen projection")
        self.btn_fullscreen.clicked.connect(self._on_fullscreen_clicked)
        btn_row.addWidget(self.btn_fullscreen)

        self.btn_match_projector = QPushButton("Set to Projector FOV")
        self.btn_match_projector.setToolTip("Set physical dimensions to 1 projector frame in µm")
        self.btn_match_projector.clicked.connect(self._on_match_projector_clicked)
        btn_row.addWidget(self.btn_match_projector)
        btn_row.addStretch()
        size_layout.addLayout(btn_row)

        layout.addWidget(size_group)
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

        self.spin_overlap_x.blockSignals(True)
        self.spin_overlap_x.setValue(s.overlap_x)
        self.spin_overlap_x.blockSignals(False)

        self.spin_overlap_y.blockSignals(True)
        self.spin_overlap_y.setValue(s.overlap_y)
        self.spin_overlap_y.blockSignals(False)

        self.spin_width_um.blockSignals(True)
        self.spin_width_um.setValue(s.width_um)
        self.spin_width_um.blockSignals(False)

        self.spin_height_um.blockSignals(True)
        self.spin_height_um.setValue(s.height_um)
        self.spin_height_um.blockSignals(False)

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

    def _on_reset_progress(self):
        self.engine.reset_chip_progress()
        self.bridge.status_message.emit("Reset chip progress (markers, origins, and exposure history)")

    def _on_load_progress(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Chip Progress", "", "JSON (*.json)")
        if path:
            self.engine.load_chip_progress(path)
            self.bridge.status_message.emit(f"Loaded chip progress: {Path(path).name}")

    def _on_save_progress(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Chip Progress", "", "JSON (*.json)")
        if path:
            self.engine.save_chip_progress(path)
            self.bridge.status_message.emit(f"Saved chip progress: {Path(path).name}")

    def _on_default_exp_changed(self, val: int):
        self.engine.project.update_settings(exposure_time=float(val))

    def _on_overlap_x_changed(self, val: float):
        self.engine.project.update_settings(overlap_x=float(val))

    def _on_overlap_y_changed(self, val: float):
        self.engine.project.update_settings(overlap_y=float(val))

    def _on_pattern_size_changed(self):
        w = self.spin_width_um.value()
        h = self.spin_height_um.value()
        self.engine.project.set_pattern_size(w, h)

    def _on_fullscreen_clicked(self):
        self.spin_width_um.blockSignals(True)
        self.spin_height_um.blockSignals(True)
        self.spin_width_um.setValue(-1.0)
        self.spin_height_um.setValue(-1.0)
        self.spin_width_um.blockSignals(False)
        self.spin_height_um.blockSignals(False)
        self.engine.project.set_pattern_size(-1.0, -1.0)
        self.bridge.status_message.emit("Set project pattern size to Fullscreen mode (-1 × -1)")

    def _on_match_projector_clicked(self):
        pw, ph = self.engine.project.projector_size
        if hasattr(self.engine, "projector") and self.engine.projector is not None:
            pw, ph = self.engine.projector.projector_size()
        proj_px = 1.0
        if self.engine.machine_config is not None:
            proj_px = float(self.engine.machine_config.optics.projector_pixel_size)
        w_um = pw * proj_px
        h_um = ph * proj_px
        self.spin_width_um.blockSignals(True)
        self.spin_height_um.blockSignals(True)
        self.spin_width_um.setValue(w_um)
        self.spin_height_um.setValue(h_um)
        self.spin_width_um.blockSignals(False)
        self.spin_height_um.blockSignals(False)
        self.engine.project.set_pattern_size(w_um, h_um)
        self.bridge.status_message.emit(f"Set project pattern size to Projector FOV: {w_um:.1f} × {h_um:.1f} µm")


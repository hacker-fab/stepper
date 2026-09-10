from pathlib import Path
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
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
from core.tiling import split_image_with_overlap
from ui.bridge import QtEngineBridge


class DataPanelWidget(QTabWidget):
    """Grouped panel for all file loading and data management (tabbed for clarity)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        # Tab 1: Pattern Mask
        self.pattern_tab = QWidget()
        self._setup_pattern_tab()
        self.addTab(self.pattern_tab, "Pattern Mask")

        # Tab 2: Chip Project
        self.chip_tab = QWidget()
        self._setup_chip_tab()
        self.addTab(self.chip_tab, "Chip Project")

        # Tab 3: Tiling Slicing
        self.tiling_tab = QWidget()
        self._setup_tiling_tab()
        self.addTab(self.tiling_tab, "Tiling Slicing")

        # Signal connections
        self.bridge.chip_changed.connect(lambda _: self._refresh_chip_table())

    # -------------------------------------------------------------------------
    # Tab 1: Pattern Mask Setup
    # -------------------------------------------------------------------------
    def _setup_pattern_tab(self):
        layout = QVBoxLayout(self.pattern_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # File selection row
        file_row = QHBoxLayout()
        self.load_pattern_btn = QPushButton("Select Pattern File...")
        self.load_pattern_btn.clicked.connect(self._on_load_pattern_clicked)
        file_row.addWidget(self.load_pattern_btn)

        self.pattern_path_label = QLabel("No pattern loaded")
        self.pattern_path_label.setStyleSheet("color: #888888; font-style: italic;")
        file_row.addWidget(self.pattern_path_label, stretch=1)
        layout.addLayout(file_row)

        # Thumbnail and adjustments container
        middle_row = QHBoxLayout()

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(120, 80)
        self.thumb_label.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setText("Thumbnail")
        middle_row.addWidget(self.thumb_label)

        # Fine adjustments form
        adj_group = QGroupBox("Adjustments")
        adj_form = QFormLayout(adj_group)
        adj_form.setContentsMargins(6, 6, 6, 6)
        adj_form.setSpacing(4)

        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-2000.0, 2000.0)
        self.spin_x.setSingleStep(1.0)
        self.spin_x.setValue(0.0)
        self.spin_x.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Shift X (px):", self.spin_x)

        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-2000.0, 2000.0)
        self.spin_y.setSingleStep(1.0)
        self.spin_y.setValue(0.0)
        self.spin_y.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Shift Y (px):", self.spin_y)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(-180.0, 180.0)
        self.spin_theta.setSingleStep(0.1)
        self.spin_theta.setValue(0.0)
        self.spin_theta.valueChanged.connect(self._on_adjust_changed)
        adj_form.addRow("Rotation θ (°):", self.spin_theta)

        middle_row.addWidget(adj_group)
        layout.addLayout(middle_row)

        # Preprocessing group
        pre_group = QGroupBox("Preprocessing")
        pre_form = QFormLayout(pre_group)
        pre_form.setContentsMargins(6, 6, 6, 6)

        self.spin_border = QDoubleSpinBox()
        self.spin_border.setRange(0.0, 500.0)
        self.spin_border.setValue(0.0)
        self.spin_border.valueChanged.connect(lambda val: self.engine.set_border_size(val))
        pre_form.addRow("Border (px):", self.spin_border)

        self.spin_posterize = QSpinBox()
        self.spin_posterize.setRange(0, 256)
        self.spin_posterize.setValue(0)
        self.spin_posterize.setSpecialValueText("Disabled")
        self.spin_posterize.valueChanged.connect(
            lambda val: self.engine.set_posterize_strength(val if val > 0 else None)
        )
        pre_form.addRow("Posterize:", self.spin_posterize)

        layout.addWidget(pre_group)
        layout.addStretch()

    def _on_load_pattern_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Pattern", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            try:
                img = Image.open(filename)
                self.engine.set_pattern_image(img, filename)
                self.pattern_path_label.setText(Path(filename).name)

                # Update thumbnail
                thumb = img.resize((120, 80))
                if thumb.mode != "RGBA":
                    thumb = thumb.convert("RGBA")
                data = thumb.tobytes("raw", "RGBA")
                qimg = QImage(data, 120, 80, QImage.Format_RGBA8888)
                self.thumb_label.setPixmap(QPixmap.fromImage(qimg))
                self.bridge.status_message.emit(f"Pattern loaded: {Path(filename).name}")
            except Exception as e:
                self.bridge.warning_emitted.emit(f"Failed to load image: {e}")

    def _on_adjust_changed(self):
        self.engine.set_image_position(
            self.spin_x.value(), self.spin_y.value(), self.spin_theta.value()
        )

    # -------------------------------------------------------------------------
    # Tab 2: Chip Project Setup
    # -------------------------------------------------------------------------
    def _setup_chip_tab(self):
        layout = QVBoxLayout(self.chip_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        self.btn_load_chip = QPushButton("Load Chip...")
        self.btn_load_chip.clicked.connect(self._on_load_chip)
        btn_row.addWidget(self.btn_load_chip)

        self.btn_save_chip = QPushButton("Save Chip...")
        self.btn_save_chip.clicked.connect(self._on_save_chip)
        btn_row.addWidget(self.btn_save_chip)

        self.btn_new_chip = QPushButton("New Chip")
        self.btn_new_chip.clicked.connect(lambda: self.engine.new_chip())
        btn_row.addWidget(self.btn_new_chip)

        self.btn_add_layer = QPushButton("+ Layer")
        self.btn_add_layer.clicked.connect(lambda: self.engine.add_chip_layer())
        btn_row.addWidget(self.btn_add_layer)

        layout.addLayout(btn_row)

        # Chip layers table
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(2)
        self.layers_table.setHorizontalHeaderLabels(["Layer #", "Exposures Count"])
        self.layers_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.layers_table)

    def _refresh_chip_table(self):
        chip = self.engine.chip
        self.layers_table.setRowCount(len(chip.layers))
        for i, layer in enumerate(chip.layers):
            self.layers_table.setItem(i, 0, QTableWidgetItem(f"Layer {i + 1}"))
            self.layers_table.setItem(i, 1, QTableWidgetItem(str(len(layer.exposures))))

    def _on_load_chip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Chip Project", "", "JSON (*.json)")
        if path:
            self.engine.load_chip(path)
            self.bridge.status_message.emit(f"Chip loaded: {Path(path).name}")

    def _on_save_chip(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Chip Project", "", "JSON (*.json)")
        if path:
            self.engine.save_chip(path)
            self.bridge.status_message.emit(f"Chip saved: {Path(path).name}")

    # -------------------------------------------------------------------------
    # Tab 3: Tiling Slicing Setup
    # -------------------------------------------------------------------------
    def _setup_tiling_tab(self):
        layout = QVBoxLayout(self.tiling_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        self.spin_tile_w = QSpinBox()
        self.spin_tile_w.setRange(100, 10000)
        self.spin_tile_w.setValue(3840)
        form.addRow("Tile Width (px):", self.spin_tile_w)

        self.spin_tile_h = QSpinBox()
        self.spin_tile_h.setRange(100, 10000)
        self.spin_tile_h.setValue(2160)
        form.addRow("Tile Height (px):", self.spin_tile_h)

        self.spin_overlap_x = QSpinBox()
        self.spin_overlap_x.setRange(0, 1000)
        self.spin_overlap_x.setValue(200)
        form.addRow("Overlap X (px):", self.spin_overlap_x)

        self.spin_overlap_y = QSpinBox()
        self.spin_overlap_y.setRange(0, 1000)
        self.spin_overlap_y.setValue(200)
        form.addRow("Overlap Y (px):", self.spin_overlap_y)

        layout.addLayout(form)

        self.btn_segment = QPushButton("Segment Large Image into Tiles")
        self.btn_segment.clicked.connect(self._on_segment_clicked)
        layout.addWidget(self.btn_segment)

        self.tiling_info_label = QLabel("No segmentation performed yet.")
        self.tiling_info_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.tiling_info_label)
        layout.addStretch()

    def _on_segment_clicked(self):
        if not self.engine.pattern_image_path:
            self.bridge.warning_emitted.emit("Please select a pattern image first.")
            return

        try:
            nx, ny, total, (w, h) = split_image_with_overlap(
                image_path=self.engine.pattern_image_path,
                tile_width=self.spin_tile_w.value(),
                tile_height=self.spin_tile_h.value(),
                overlap_x=self.spin_overlap_x.value(),
                overlap_y=self.spin_overlap_y.value(),
                output_dir="tiles",
            )
            self.tiling_info_label.setText(
                f"Source: {w}x{h} px | Layout: {nx} x {ny} tiles ({total} total)"
            )
            self.bridge.status_message.emit(f"Generated {total} tiles in 'tiles/' folder")
        except Exception as e:
            self.bridge.warning_emitted.emit(f"Segmentation failed: {e}")


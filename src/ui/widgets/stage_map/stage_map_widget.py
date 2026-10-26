from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge
from .stage_map_canvas import StageMapCanvas


class StageMapWidget(QWidget):
    """2D visualization of stage travel boundaries, tool position, and exposed spots."""

    def __init__(
        self,
        engine: StepperEngine,
        bridge: QtEngineBridge,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header row: coordinate readout + zoom controls + clear history button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.coord_label = QLabel("Position: X 0.000, Y 0.000, Z 0.000 um")
        self.coord_label.setStyleSheet("font-size: 11px;")
        header_layout.addWidget(self.coord_label, stretch=1)

        # Zoom controls
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(26)
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        header_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("Fit")
        self.btn_zoom_reset.setFixedWidth(44)
        self.btn_zoom_reset.setToolTip("Reset zoom/pan (fit to view)")
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        header_layout.addWidget(self.btn_zoom_reset)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(26)
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        header_layout.addWidget(self.btn_zoom_in)

        header_layout.addSpacing(6)

        self.btn_clear_history = QPushButton("Clear History")
        self.btn_clear_history.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_clear_history.setToolTip("Clear all exposure history records")
        self.btn_clear_history.clicked.connect(self._on_clear_history)
        header_layout.addWidget(self.btn_clear_history)

        layout.addLayout(header_layout)

        self.canvas = StageMapCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_stage_moved)
        self.bridge.project_changed.connect(lambda _: self.canvas.update())
        self.bridge.exposure_history_changed.connect(lambda _: self.canvas.update())

    def _on_stage_moved(self, coords: tuple):
        x, y, z = coords
        self.coord_label.setText(f"Position: X {x:.3f}, Y {y:.3f}, Z {z:.3f} um")
        self.canvas.update()

    def _on_clear_history(self):
        self.engine.project.clear_exposure_history()

    # ------------------------------------------------------------------
    # Zoom helpers (delegate to canvas, then update button label)
    # ------------------------------------------------------------------

    def _zoom_in(self):
        self.canvas.zoom(1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        self.canvas.zoom(1.0 / 1.25)
        self._update_zoom_label()

    def _zoom_reset(self):
        self.canvas.reset_zoom()
        self._update_zoom_label()

    def _update_zoom_label(self):
        pct = int(round(self.canvas.zoom_level * 100))
        self.btn_zoom_reset.setText(f"{pct}%")

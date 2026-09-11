from typing import Optional
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


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

        # Header readout
        self.coord_label = QLabel("Position: X 0.000, Y 0.000, Z 0.000 mm")
        self.coord_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self.coord_label)

        self.canvas = StageMapCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_stage_moved)
        self.bridge.project_changed.connect(lambda _: self.canvas.update())

    def _on_stage_moved(self, coords: tuple):
        x, y, z = coords
        self.coord_label.setText(f"Position: X {x:.3f}, Y {y:.3f}, Z {z:.3f} mm")
        self.canvas.update()


class StageMapCanvas(QFrame):
    """Drawing area for the stage map."""

    def __init__(self, parent_widget: StageMapWidget):
        super().__init__()
        self.parent_widget = parent_widget
        self.setStyleSheet("background-color: #121217; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Margins
        margin = 25
        dw = w - 2 * margin
        dh = h - 2 * margin

        if dw <= 0 or dh <= 0:
            return

        # Stage bounds in mm
        bounds = self.parent_widget.engine.stage.get_bounds()
        if bounds:
            min_x, max_x = bounds["x"]
            min_y, max_y = bounds["y"]
        else:
            min_x, max_x = -15.0, 0.0
            min_y, max_y = 0.0, 15.0

        span_x = max(1e-5, max_x - min_x)
        span_y = max(1e-5, max_y - min_y)

        def to_screen(x: float, y: float):
            sx = margin + (x - min_x) / span_x * dw
            sy = margin + (max_y - y) / span_y * dh
            return sx, sy

        # Draw stage travel border
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRect(margin, margin, dw, dh)

        # Draw grid lines
        painter.setPen(QPen(QColor("#1e293b").lighter(130), 1, Qt.DotLine))
        for gx in range(int(min_x), int(max_x) + 1, 5):
            sx, _ = to_screen(gx, min_y)
            painter.drawLine(int(sx), margin, int(sx), margin + dh)
        for gy in range(int(min_y), int(max_y) + 1, 5):
            _, sy = to_screen(min_x, gy)
            painter.drawLine(margin, int(sy), margin + dw, int(sy))

        # Draw previous exposure footprints
        chip_project = self.parent_widget.engine.project
        painter.setPen(QPen(QColor("#f59e0b"), 1))
        painter.setBrush(QBrush(QColor(245, 158, 11, 80)))
        for layer in chip_project.layers:
            for exp in layer.exposures:
                ex_x, ex_y, _ = exp.coords
                sx, sy = to_screen(ex_x, ex_y)
                # Draw exposure tile footprint (~1mm x 0.5mm approx)
                tile_w = max(4.0, (1.0 / span_x) * dw)
                tile_h = max(3.0, (0.5 / span_y) * dh)
                painter.drawRect(QRectF(sx - tile_w / 2, sy - tile_h / 2, tile_w, tile_h))

        # Draw current stage position indicator
        cx, cy, _ = self.parent_widget.engine.stage.get_position()
        cur_sx, cur_sy = to_screen(cx, cy)

        # Target cross
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawLine(int(cur_sx - 8), int(cur_sy), int(cur_sx + 8), int(cur_sy))
        painter.drawLine(int(cur_sx), int(cur_sy - 8), int(cur_sx), int(cur_sy + 8))

        # Target point
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(QPointF(cur_sx, cur_sy), 4, 4)


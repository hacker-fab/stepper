import math
from typing import Dict, List, Optional, Set, Tuple
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from stage_control.calibration import StageCalibration


class DistortionView(QWidget):
    """2D visualization widget showing nominal vs reconstructed optical calibration grid and distortion vectors."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #15151a; border-radius: 4px;")

        self._calibration: Optional[StageCalibration] = None
        self._detected_markers: Optional[Set[int]] = None
        self.vector_magnification: float = 20.0

        # View transform
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self._is_panning: bool = False
        self._last_mouse: Optional[QPointF] = None

    def set_calibration(
        self,
        cal: Optional[StageCalibration],
        detected_markers: Optional[Set[int]] = None,
    ):
        self._calibration = cal
        if detected_markers is not None:
            self._detected_markers = set(detected_markers)
        elif cal is not None and getattr(cal, "detected_marker_ids", None):
            self._detected_markers = set(cal.detected_marker_ids)
        else:
            self._detected_markers = None
        self.fit_view()
        self.update()

    def set_detected_markers(self, detected_markers: Optional[Set[int]]):
        self._detected_markers = set(detected_markers) if detected_markers is not None else None
        self.update()

    def set_magnification(self, mag: float):
        self.vector_magnification = max(1.0, float(mag))
        self.update()

    def fit_view(self):
        """Fits all points in the widget coordinate space."""
        if self._calibration is None or not self._calibration.points:
            return
        xs = [p.raw_xy[0] for p in self._calibration.points]
        ys = [p.raw_xy[1] for p in self._calibration.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(100.0, max_x - min_x)
        span_y = max(100.0, max_y - min_y)

        vw = max(50, self.width() - 80)
        vh = max(50, self.height() - 80)

        scale = min(vw / span_x, vh / span_y)
        self.zoom = scale * 0.85
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        self.pan_x = self.width() / 2.0 - center_x * self.zoom
        self.pan_y = self.height() / 2.0 - center_y * self.zoom
        self.update()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        pos = event.position()
        old_zoom = self.zoom
        new_zoom = max(0.001, min(100.0, old_zoom * factor))
        self.pan_x = pos.x() - (pos.x() - self.pan_x) * (new_zoom / old_zoom)
        self.pan_y = pos.y() - (pos.y() - self.pan_y) * (new_zoom / old_zoom)
        self.zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = True
            self._last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self._is_panning and self._last_mouse is not None:
            delta = event.position() - self._last_mouse
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = False

    def mouseDoubleClickEvent(self, event):
        self.fit_view()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#15151a"))

        if self._calibration is None or not self._calibration.points:
            painter.setPen(QColor("#777788"))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No calibration data available.\nReprocess images to view reconstruction.")
            return

        points = self._calibration.points
        mag = self.vector_magnification

        # Draw grid connections between adjacent nominal points
        grid_map = {(p.grid_index[0], p.grid_index[1]): p for p in points}
        # painter.setPen(QPen(QColor(60, 60, 80), 1, Qt.DashLine))
        # for (r, c), p in grid_map.items():
        #     x0 = p.raw_xy[0] * self.zoom + self.pan_x
        #     y0 = p.raw_xy[1] * self.zoom + self.pan_y
        #     if (r, c + 1) in grid_map:
        #         p_r = grid_map[(r, c + 1)]
        #         painter.drawLine(QPointF(x0, y0), QPointF(p_r.raw_xy[0] * self.zoom + self.pan_x, p_r.raw_xy[1] * self.zoom + self.pan_y))
        #     if (r + 1, c) in grid_map:
        #         p_d = grid_map[(r + 1, c)]
        #         painter.drawLine(QPointF(x0, y0), QPointF(p_d.raw_xy[0] * self.zoom + self.pan_x, p_d.raw_xy[1] * self.zoom + self.pan_y))

        # Draw nominal points (Blue), reconstructed points (Green), and distortion vector (Red)
        for p in points:
            x_nom = p.raw_xy[0] * self.zoom + self.pan_x
            y_nom = p.raw_xy[1] * self.zoom + self.pan_y

            dx = (p.corr_xy[0] - p.raw_xy[0]) * mag
            dy = (p.corr_xy[1] - p.raw_xy[1]) * mag
            x_recon = (p.raw_xy[0] + dx) * self.zoom + self.pan_x
            y_recon = (p.raw_xy[1] + dy) * self.zoom + self.pan_y

            # Vector arrow / line
            # if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            #     painter.setPen(QPen(QColor(255, 75, 75, 200), 1.5))
            #     painter.drawLine(QPointF(x_nom, y_nom), QPointF(x_recon, y_recon))

            # Nominal point
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(70, 140, 255)))
            painter.drawEllipse(QPointF(x_nom, y_nom), 3.5, 3.5)

            # Reconstructed point: green for detected anchor points (points with constraints), gray otherwise
            is_detected = (
                self._detected_markers is not None and p.marker_id in self._detected_markers
            ) or (
                getattr(self._calibration, "detected_marker_ids", None)
                and p.marker_id in self._calibration.detected_marker_ids
            )
            recon_color = QColor(50, 220, 100) if is_detected else QColor(130, 130, 140)
            painter.setBrush(QBrush(recon_color))
            painter.drawEllipse(QPointF(x_recon, y_recon), 3.0, 3.0)

            # Label tag ID
            # painter.setPen(QColor(180, 180, 200))
            # font = QFont()
            # font.setPointSize(8)
            # painter.setFont(font)
            # painter.drawText(QPointF(x_nom + 5, y_nom - 5), str(p.marker_id))

        # Legend overlay in top-right corner
        self._draw_legend(painter)

    def _draw_legend(self, painter: QPainter):
        legend_rect = QRectF(self.width() - 250, 12, 238, 106)
        painter.setPen(QPen(QColor(60, 60, 75), 1))
        painter.setBrush(QBrush(QColor(25, 25, 32, 220)))
        painter.drawRoundedRect(legend_rect, 6, 6)

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # Nominal
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(70, 140, 255)))
        painter.drawEllipse(QPointF(self.width() - 235, 28), 4, 4)
        painter.setPen(QColor(220, 220, 230))
        painter.drawText(QPointF(self.width() - 220, 32), "Nominal Stage Coordinate")

        # Detected Reconstructed (Green)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(50, 220, 100)))
        painter.drawEllipse(QPointF(self.width() - 235, 48), 4, 4)
        painter.setPen(QColor(220, 220, 230))
        painter.drawText(QPointF(self.width() - 220, 52), "Detected Anchor (Green)")

        # Undetected Reconstructed (Gray)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(130, 130, 140)))
        painter.drawEllipse(QPointF(self.width() - 235, 68), 4, 4)
        painter.setPen(QColor(220, 220, 230))
        painter.drawText(QPointF(self.width() - 220, 72), "Undetected Anchor (Gray)")

        # Vector
        painter.setPen(QPen(QColor(255, 75, 75), 2))
        painter.drawLine(QPointF(self.width() - 240, 88), QPointF(self.width() - 228, 88))
        painter.setPen(QColor(220, 220, 230))
        painter.drawText(QPointF(self.width() - 220, 92), f"Distortion Vector ({self.vector_magnification:.0f}x Mag)")


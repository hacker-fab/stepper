from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QFrame

from operations.movement import JogOperation

if TYPE_CHECKING:
    from .stage_map_widget import StageMapWidget


class StageMapCanvas(QFrame):
    """Drawing area for the stage map with zoom, pan, and click-to-move."""

    ZOOM_MIN = 0.2
    ZOOM_MAX = 40.0

    def __init__(self, parent_widget: "StageMapWidget"):
        super().__init__()
        self.parent_widget = parent_widget
        self.setStyleSheet("background-color: #121217; border-radius: 4px;")

        # Zoom & pan state
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Mouse drag tracking
        self._is_panning: bool = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._drag_distance: float = 0.0

        # Crosshair cursor signals interactive click-to-move
        self.setCursor(QCursor(Qt.CrossCursor))

    # ------------------------------------------------------------------
    # Zoom / pan API
    # ------------------------------------------------------------------

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def zoom(self, factor: float, center_point: Optional[QPointF] = None):
        old_zoom = self.zoom_level
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        if center_point is not None:
            t = self._get_transform()
            if t is not None:
                ox, oy, scale, min_x, max_x, min_y, max_y = t
                span_x = max_x - min_x
                span_y = max_y - min_y
                old_box_w = span_x * scale
                old_box_h = span_y * scale

                # Normalised position of mouse within the zoomed stage box
                u = (center_point.x() - ox) / old_box_w
                v = (center_point.y() - oy) / old_box_h

                self.zoom_level = new_zoom
                base_scale = scale / old_zoom
                new_scale = base_scale * new_zoom
                new_box_w = span_x * new_scale
                new_box_h = span_y * new_scale

                margin = 25
                avail_w = self.width() - 2 * margin
                avail_h = self.height() - 2 * margin
                new_ox_base = margin + (avail_w - new_box_w) / 2.0
                new_oy_base = margin + (avail_h - new_box_h) / 2.0

                self.pan_x = center_point.x() - u * new_box_w - new_ox_base
                self.pan_y = center_point.y() - v * new_box_h - new_oy_base
            else:
                self.zoom_level = new_zoom
        else:
            self.zoom_level = new_zoom

        self.update()

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _get_transform(self):
        """Return (ox, oy, scale, min_x, max_x, min_y, max_y) for current view state."""
        w = self.width()
        h = self.height()
        margin = 25
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin
        if avail_w <= 0 or avail_h <= 0:
            return None

        bounds = self.parent_widget.engine.stage.get_bounds()
        if bounds:
            min_x, max_x = bounds["x"]
            min_y, max_y = bounds["y"]
        else:
            min_x, max_x = -15000.0, 0.0
            min_y, max_y = 0.0, 15000.0

        span_x = max(1e-5, max_x - min_x)
        span_y = max(1e-5, max_y - min_y)

        base_scale = min(avail_w / span_x, avail_h / span_y)
        scale = base_scale * self.zoom_level
        box_w = span_x * scale
        box_h = span_y * scale

        ox = margin + (avail_w - box_w) / 2.0 + self.pan_x
        oy = margin + (avail_h - box_h) / 2.0 + self.pan_y

        return ox, oy, scale, min_x, max_x, min_y, max_y

    def _to_screen(self, x: float, y: float, t) -> tuple:
        ox, oy, scale, min_x, max_x, min_y, max_y = t
        return ox + (x - min_x) * scale, oy + (max_y - y) * scale

    def _from_screen(self, sx: float, sy: float, t) -> tuple:
        """Screen pixel → stage µm."""
        ox, oy, scale, min_x, max_x, min_y, max_y = t
        return (sx - ox) / scale + min_x, max_y - (sy - oy) / scale

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        t = self._get_transform()
        if t is None:
            return

        ox, oy, scale, min_x, max_x, min_y, max_y = t
        span_x = max(1e-5, max_x - min_x)
        span_y = max(1e-5, max_y - min_y)
        box_w = span_x * scale
        box_h = span_y * scale

        # Draw stage travel border
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRect(QRectF(ox, oy, box_w, box_h))

        # Draw previous exposure footprints
        chip_project = self.parent_widget.engine.project
        if chip_project:
            painter.setPen(QPen(QColor("#f59e0b"), 1))
            painter.setBrush(QBrush(QColor(245, 158, 11, 80)))
            pitch_x = chip_project.settings.pitch_x
            pitch_y = chip_project.settings.pitch_y
            tile_w = max(4.0, (pitch_x / span_x) * box_w)
            tile_h = max(3.0, (pitch_y / span_y) * box_h)

            for exp in chip_project.exposure_history:
                ex_x, ex_y = exp.coords[0], exp.coords[1]
                sx, sy = self._to_screen(ex_x, ex_y, t)
                painter.drawRect(QRectF(sx - tile_w / 2, sy - tile_h / 2, tile_w, tile_h))

        # Draw current stage position indicator
        cx, cy, _ = self.parent_widget.engine.stage.get_position()
        cur_sx, cur_sy = self._to_screen(cx, cy, t)

        # Target cross
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawLine(QPointF(cur_sx - 8, cur_sy), QPointF(cur_sx + 8, cur_sy))
        painter.drawLine(QPointF(cur_sx, cur_sy - 8), QPointF(cur_sx, cur_sy + 8))

        # Target dot
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(QPointF(cur_sx, cur_sy), 4, 4)

        # Zoom level badge (bottom-right)
        pct_text = f"{int(round(self.zoom_level * 100))}%"
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(pct_text)
        th = fm.height()
        painter.setPen(QColor("#64748b"))
        painter.drawText(
            QRectF(self.width() - tw - 8, self.height() - th - 6, tw + 2, th),
            Qt.AlignRight | Qt.AlignVCenter,
            pct_text,
        )

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            self.zoom(factor, event.position())
            self.parent_widget._update_zoom_label()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self._drag_distance = 0.0
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_panning and self._last_mouse_pos is not None:
            delta = event.position() - self._last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._drag_distance += (delta.x() ** 2 + delta.y() ** 2) ** 0.5
            self._last_mouse_pos = event.position()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_panning:
            drag_dist = self._drag_distance
            self._is_panning = False
            self._last_mouse_pos = None
            # Short drag = click → move stage
            if drag_dist < 5.0:
                self._move_stage_to(event.position())
            event.accept()
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            self._last_mouse_pos = None
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Double-click resets zoom and pan."""
        if event.button() == Qt.LeftButton:
            self.reset_zoom()
            self.parent_widget._update_zoom_label()
        event.accept()

    # ------------------------------------------------------------------
    # Stage movement
    # ------------------------------------------------------------------

    def _move_stage_to(self, screen_pos: QPointF):
        """Convert a click position to stage coordinates and dispatch an absolute move."""
        t = self._get_transform()
        if t is None:
            return

        ox, oy, scale, min_x, max_x, min_y, max_y = t
        span_x = max_x - min_x
        span_y = max_y - min_y
        stage_rect = QRectF(ox, oy, span_x * scale, span_y * scale)

        if not stage_rect.contains(screen_pos):
            return  # Ignore clicks outside the stage travel boundary

        x_mm, y_mm = self._from_screen(screen_pos.x(), screen_pos.y(), t)
        # Clamp to stage bounds
        x_mm = max(min_x, min(max_x, x_mm))
        y_mm = max(min_y, min(max_y, y_mm))

        op = JogOperation({"x": x_mm, "y": y_mm}, relative=False)
        self.parent_widget.bridge.start_operation(op)

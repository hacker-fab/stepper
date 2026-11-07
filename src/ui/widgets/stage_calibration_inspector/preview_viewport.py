from typing import List, Optional
import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from lib.stage_calibration_reconstruction.detector import draw_detected_markers


class PreviewViewport(QWidget):
    """Interactive viewport for inspected camera images with ArUco detection overlays, zoom, and pan."""

    zoom_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #121216; border-radius: 4px;")

        self._raw_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._display_qimage: Optional[QImage] = None

        self.corners: List[np.ndarray] = []
        self.ids: Optional[np.ndarray] = None

        # Zoom & Pan
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        self._is_panning: bool = False
        self._last_mouse_pos: Optional[QPointF] = None

        self.setMouseTracking(True)

    def display_frame(
        self,
        frame: Optional[np.ndarray],
        corners: Optional[List[np.ndarray]] = None,
        ids: Optional[np.ndarray] = None,
        overlay_markers: bool = True,
        fit_view: bool = False,
    ):
        """Displays a given frame (e.g. input or intermediate debug image) with optional marker overlay."""
        self._raw_frame = frame
        self.corners = corners if corners is not None else []
        self.ids = ids

        if frame is None or getattr(frame, "size", 0) == 0:
            self._annotated_frame = None
            self._display_qimage = None
            self.update()
            return

        if overlay_markers and self.ids is not None and len(self.corners) > 0:
            annotated = draw_detected_markers(frame, self.corners, self.ids)
        else:
            annotated = frame.copy()

        self._annotated_frame = annotated

        h, w = annotated.shape[:2]
        bytes_per_line = annotated.strides[0]
        if annotated.ndim == 2:
            self._display_qimage = QImage(annotated.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        else:
            self._display_qimage = QImage(annotated.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()

        if fit_view:
            self.fit_to_window()

        self.update()

    def set_image(
        self,
        frame: Optional[np.ndarray],
        corners: Optional[List[np.ndarray]] = None,
        ids: Optional[np.ndarray] = None,
        fit_view: bool = False,
    ):
        """Sets the current frame and detected marker annotations."""
        self.display_frame(frame, corners=corners, ids=ids, overlay_markers=True, fit_view=fit_view)

    def update_markers(self, corners: List[np.ndarray], ids: Optional[np.ndarray], overlay_markers: bool = True):
        """Updates marker annotations without resetting zoom or reloading raw frame."""
        self.display_frame(self._raw_frame, corners=corners, ids=ids, overlay_markers=overlay_markers, fit_view=False)

    def fit_to_window(self):
        """Fits the current image inside the viewport widget."""
        if self._display_qimage is None:
            return
        iw = self._display_qimage.width()
        ih = self._display_qimage.height()
        vw = max(10, self.width())
        vh = max(10, self.height())

        scale_x = vw / iw
        scale_y = vh / ih
        self.zoom_level = min(scale_x, scale_y) * 0.95
        self.pan_x = (vw - iw * self.zoom_level) / 2.0
        self.pan_y = (vh - ih * self.zoom_level) / 2.0
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def reset_zoom(self):
        """Sets zoom level to 1.0."""
        if self._display_qimage is None:
            return
        vw = max(10, self.width())
        vh = max(10, self.height())
        iw = self._display_qimage.width()
        ih = self._display_qimage.height()
        self.zoom_level = 1.0
        self.pan_x = (vw - iw) / 2.0
        self.pan_y = (vh - ih) / 2.0
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom_in(self):
        self._zoom_at_center(1.25)

    def zoom_out(self):
        self._zoom_at_center(0.8)

    def _zoom_at_center(self, factor: float):
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        old_zoom = self.zoom_level
        new_zoom = max(0.05, min(20.0, old_zoom * factor))
        self.pan_x = cx - (cx - self.pan_x) * (new_zoom / old_zoom)
        self.pan_y = cy - (cy - self.pan_y) * (new_zoom / old_zoom)
        self.zoom_level = new_zoom
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def wheelEvent(self, event):
        if self._display_qimage is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        pos = event.position()
        old_zoom = self.zoom_level
        new_zoom = max(0.05, min(20.0, old_zoom * factor))

        self.pan_x = pos.x() - (pos.x() - self.pan_x) * (new_zoom / old_zoom)
        self.pan_y = pos.y() - (pos.y() - self.pan_y) * (new_zoom / old_zoom)
        self.zoom_level = new_zoom
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if self._is_panning and self._last_mouse_pos is not None:
            delta = event.position() - self._last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_level == 1.0 or self._display_qimage is None:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#121216"))

        if self._display_qimage is None:
            painter.setPen(QColor("#666677"))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image selected")
            return

        iw = self._display_qimage.width()
        ih = self._display_qimage.height()
        target_rect = QRectF(
            self.pan_x,
            self.pan_y,
            iw * self.zoom_level,
            ih * self.zoom_level,
        )
        painter.drawImage(target_rect, self._display_qimage)


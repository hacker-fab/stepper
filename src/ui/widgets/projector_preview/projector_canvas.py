from typing import Optional, TYPE_CHECKING
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .projector_preview_widget import ProjectorPreviewWidget


class ProjectorCanvas(QWidget):
    def __init__(self, parent_view: "ProjectorPreviewWidget"):
        super().__init__(parent_view)
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #141416; border-radius: 4px;")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pixmap: Optional[QPixmap] = None
        self._qimage: Optional[QImage] = None

    def refresh_image(self):
        img = (
            self.parent_view.engine.projector._displayed_image_cache
            if hasattr(self.parent_view.engine, "projector")
            and self.parent_view.engine.projector is not None
            and self.parent_view.engine.projector.is_on
            else None
        )
        if img is not None and isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            contig = np.ascontiguousarray(img)
            if img.ndim == 2:
                qimg = QImage(contig.data, w, h, w, QImage.Format_Grayscale8)
            elif img.shape[2] == 3:
                qimg = QImage(contig.data, w, h, 3 * w, QImage.Format_BGR888)
            elif img.shape[2] == 4:
                qimg = QImage(contig.data, w, h, 4 * w, QImage.Format_RGBA8888)
            else:
                qimg = None

            if qimg is not None:
                self._qimage = qimg
                self._pixmap = QPixmap.fromImage(qimg)
            else:
                self._qimage = None
                self._pixmap = None
        else:
            self._qimage = None
            self._pixmap = None

        self.update()
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#141416"))

        # Determine projector display dimensions
        pw, ph = 1920, 1080
        if hasattr(self.parent_view.engine, "projector") and self.parent_view.engine.projector is not None:
            size = self.parent_view.engine.projector.projector_size()
            if size and size[0] > 0 and size[1] > 0:
                pw, ph = size

        margin = 8
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        scale = min(avail_w / pw, avail_h / ph)
        rect_w = max(1, int(pw * scale))
        rect_h = max(1, int(ph * scale))
        rect_x = (self.width() - rect_w) // 2
        rect_y = (self.height() - rect_h) // 2
        proj_rect = QRect(rect_x, rect_y, rect_w, rect_h)

        # Fill projector screen area with black background
        painter.fillRect(proj_rect, QColor("#000000"))

        is_on = (
            self.parent_view.engine.projector.is_on
            if hasattr(self.parent_view.engine, "projector")
            and self.parent_view.engine.projector is not None
            else False
        )

        if is_on and self._pixmap is not None and not self._pixmap.isNull():
            draw_w = max(1, int(self._pixmap.width() * scale))
            draw_h = max(1, int(self._pixmap.height() * scale))
            painter.save()
            painter.setClipRect(proj_rect)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(QRect(rect_x, rect_y, draw_w, draw_h), self._pixmap)
            painter.restore()
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(proj_rect, Qt.AlignCenter, "No Output (Screen Off)")

        # Draw rectangular frame representing the projector area
        frame_color = QColor("#38bdf8") if is_on else QColor("#52525b")
        painter.setPen(QPen(frame_color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(proj_rect)

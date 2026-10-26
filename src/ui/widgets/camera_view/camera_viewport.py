from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class CameraViewport(QWidget):
    """Display widget that paints the camera image with crosshair overlay, zooming, panning, and latent mask blending."""

    zoom_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #0d0d11; border-radius: 4px;")

        # Display images and frames
        self.current_frame: Optional[np.ndarray] = None
        self.current_qimage: Optional[QImage] = None
        self.latent_mask: Optional[np.ndarray] = None
        self.latent_mask_image: Optional[QImage] = None

        # Crosshair settings
        self.show_crosshairs: bool = True
        self.is_moving_crosshair: bool = False

        # Zoom and Pan parameters (UI level)
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Normalized crosshair coordinates relative to the image [0.0 - 1.0]
        self.crosshair_u: float = 0.5
        self.crosshair_v: float = 0.5

        # Secondary crosshair (set by right-click in move-crosshair mode)
        self.secondary_crosshair_u: float = 0.0
        self.secondary_crosshair_v: float = 0.0
        self.has_secondary_crosshair: bool = False

        # Mouse interaction state
        self._is_panning: bool = False
        self._last_mouse_pos: Optional[QPointF] = None

        # Cached composite image for get_display_image
        self._cached_composite_qimage: Optional[QImage] = None
        self._cached_composite_frame_qimg: Optional[QImage] = None
        self._cached_composite_mask_qimg: Optional[QImage] = None

    def set_frame(self, frame: Optional[np.ndarray]):
        """Sets the current camera frame and updates the underlying QImage."""
        self.current_frame = frame
        if frame is None:
            self.current_qimage = None
        else:
            h, w = frame.shape[:2]
            bytes_per_line = frame.strides[0]
            if frame.ndim == 2:
                self.current_qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
            else:
                self.current_qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()
        self.update()

    @staticmethod
    def _create_mask_qimage(mask: np.ndarray) -> QImage:
        h, w = mask.shape[:2]
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        alpha = np.clip(mask_2d * 255.0, 0.0, 255.0).astype(np.uint8)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[:, :, 1] = alpha  # Green channel premultiplied
        overlay[:, :, 3] = alpha  # Alpha channel
        return QImage(overlay.data, w, h, overlay.strides[0], QImage.Format_RGBA8888_Premultiplied).copy()

    def set_latent_mask(self, mask: Optional[object]):
        """Sets the latent vision mask.

        Enables blending when the mask is set, and shows the original image when mask is None.
        Accepts either a pre-rendered QImage or a numpy array mask.
        """
        if mask is None:
            if self.latent_mask is None and self.latent_mask_image is None:
                return
            self.latent_mask = None
            self.latent_mask_image = None
        elif isinstance(mask, QImage):
            if self.latent_mask_image is mask:
                return
            self.latent_mask = None
            self.latent_mask_image = mask
        elif isinstance(mask, np.ndarray):
            if self.latent_mask is mask:
                return
            self.latent_mask = mask
            self.latent_mask_image = self._create_mask_qimage(mask)
        else:
            self.latent_mask = None
            self.latent_mask_image = None
        self.update()

    def set_show_crosshairs(self, show: bool):
        """Toggles crosshair visibility."""
        self.show_crosshairs = show
        self.update()

    def set_moving_crosshair(self, moving: bool):
        """Sets whether cursor interaction moves crosshairs."""
        self.is_moving_crosshair = moving
        if moving:
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
        self.update()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom(self, factor: float, center_point: Optional[QPointF] = None):
        old_zoom = self.zoom_level
        new_zoom = max(0.2, min(25.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        qimg = self.current_qimage
        if qimg is not None and not qimg.isNull() and center_point is not None:
            # Mouse-centered zoom
            vw, vh = self.width(), self.height()
            iw, ih = qimg.width(), qimg.height()
            base_scale = min(vw / iw, vh / ih)

            old_dw = iw * base_scale * old_zoom
            old_dh = ih * base_scale * old_zoom
            old_dx = (vw - old_dw) / 2.0 + self.pan_x
            old_dy = (vh - old_dh) / 2.0 + self.pan_y

            # Normalized coordinate under the mouse
            u = (center_point.x() - old_dx) / old_dw
            v = (center_point.y() - old_dy) / old_dh

            self.zoom_level = new_zoom
            new_dw = iw * base_scale * new_zoom
            new_dh = ih * base_scale * new_zoom

            new_dx = center_point.x() - u * new_dw
            new_dy = center_point.y() - v * new_dh

            self.pan_x = new_dx - (vw - new_dw) / 2.0
            self.pan_y = new_dy - (vh - new_dh) / 2.0
        else:
            self.zoom_level = new_zoom

        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def _get_target_rect(self) -> QRectF:
        qimg = self.current_qimage
        vw = self.width()
        vh = self.height()
        if qimg is None or qimg.isNull():
            return QRectF(0, 0, vw, vh)

        iw = qimg.width()
        ih = qimg.height()
        base_scale = min(vw / iw, vh / ih)
        dw = iw * base_scale * self.zoom_level
        dh = ih * base_scale * self.zoom_level
        dx = (vw - dw) / 2.0 + self.pan_x
        dy = (vh - dh) / 2.0 + self.pan_y
        return QRectF(dx, dy, dw, dh)

    def get_display_image(self) -> Optional[QImage]:
        """Returns the QImage to render.

        Enables blending when the mask is set, and shows the original image when mask is None.
        """
        if self.latent_mask_image is None or self.current_qimage is None:
            return self.current_qimage

        if (
            self._cached_composite_qimage is not None
            and self._cached_composite_frame_qimg is self.current_qimage
            and self._cached_composite_mask_qimg is self.latent_mask_image
        ):
            return self._cached_composite_qimage

        target = QImage(self.current_qimage.size(), QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(target)
        painter.drawImage(0, 0, self.current_qimage)
        painter.drawImage(0, 0, self.latent_mask_image)
        painter.end()

        self._cached_composite_frame_qimg = self.current_qimage
        self._cached_composite_mask_qimg = self.latent_mask_image
        self._cached_composite_qimage = target
        return target

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Draw background
        painter.fillRect(self.rect(), QColor("#0d0d11"))

        target_rect = self._get_target_rect()

        if self.current_qimage is not None and not self.current_qimage.isNull():
            # Fast direct render into target_rect without per-frame CPU reallocation
            painter.drawImage(target_rect, self.current_qimage)
            if self.latent_mask_image is not None and not self.latent_mask_image.isNull():
                painter.drawImage(target_rect, self.latent_mask_image)
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Camera Feed Available")

        # Draw Crosshair
        if self.show_crosshairs:
            cx = target_rect.x() + self.crosshair_u * target_rect.width()
            cy = target_rect.y() + self.crosshair_v * target_rect.height()

            pen = QPen(QColor(0, 255, 128, 180), 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
            painter.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))

            # Center target circle
            pen_solid = QPen(QColor(0, 255, 128, 220), 1.5)
            painter.setPen(pen_solid)
            painter.drawEllipse(QPointF(cx, cy), 15, 15)
            painter.drawEllipse(QPointF(cx, cy), 35, 35)

            # Draw secondary crosshair (orange) and distance label
            if self.has_secondary_crosshair:
                scx = target_rect.x() + self.secondary_crosshair_u * target_rect.width()
                scy = target_rect.y() + self.secondary_crosshair_v * target_rect.height()

                pen_sec = QPen(QColor(255, 165, 0, 180), 1.5, Qt.DashLine)
                painter.setPen(pen_sec)
                painter.drawLine(QPointF(0, scy), QPointF(self.width(), scy))
                painter.drawLine(QPointF(scx, 0), QPointF(scx, self.height()))

                pen_sec_solid = QPen(QColor(255, 165, 0, 220), 1.5)
                painter.setPen(pen_sec_solid)
                painter.drawEllipse(QPointF(scx, scy), 15, 15)
                painter.drawEllipse(QPointF(scx, scy), 35, 35)

                # Compute pixel distance between the two crosshairs
                qimg = self.current_qimage
                if qimg is not None and not qimg.isNull() and target_rect.width() > 0:
                    iw = qimg.width()
                    ih = qimg.height()
                    du = (self.secondary_crosshair_u - self.crosshair_u) * iw
                    dv = (self.secondary_crosshair_v - self.crosshair_v) * ih
                    dist_px = (du ** 2 + dv ** 2) ** 0.5

                    # Draw a line connecting the two crosshairs
                    pen_line = QPen(QColor(255, 255, 255, 120), 1.0, Qt.DotLine)
                    painter.setPen(pen_line)
                    painter.drawLine(QPointF(cx, cy), QPointF(scx, scy))

                    # Distance label at the midpoint
                    mid_x = (cx + scx) / 2
                    mid_y = (cy + scy) / 2
                    label = f"{dist_px:.1f} px"
                    font = painter.font()
                    font.setPointSize(9)
                    font.setBold(True)
                    painter.setFont(font)
                    fm = painter.fontMetrics()
                    lw = fm.horizontalAdvance(label)
                    lh = fm.height()
                    pad = 4
                    bg_rect = QRectF(mid_x - lw / 2 - pad, mid_y - lh / 2 - pad, lw + pad * 2, lh + pad * 2)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 160))
                    painter.drawRoundedRect(bg_rect, 3, 3)
                    painter.setPen(QColor(255, 220, 80))
                    painter.drawText(QRectF(mid_x - lw / 2, mid_y - lh / 2, lw, lh), Qt.AlignCenter, label)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            self.zoom(factor, event.position())
        event.accept()

    def mousePressEvent(self, event):
        if self.is_moving_crosshair and event.button() == Qt.LeftButton:
            self._update_crosshair_from_pos(event.position())
            event.accept()
            return

        if self.is_moving_crosshair and event.button() == Qt.RightButton:
            self._update_secondary_crosshair_from_pos(event.position())
            event.accept()
            return

        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_moving_crosshair and (event.buttons() & Qt.LeftButton):
            self._update_crosshair_from_pos(event.position())
            event.accept()
            return

        if self._is_panning and self._last_mouse_pos is not None:
            delta = event.position() - self._last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_mouse_pos = event.position()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            self._last_mouse_pos = None
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.is_moving_crosshair:
            self.crosshair_u = 0.5
            self.crosshair_v = 0.5
            self.update()
        else:
            self.reset_zoom()
        event.accept()

    def _update_crosshair_from_pos(self, pos: QPointF):
        target_rect = self._get_target_rect()
        if target_rect.width() > 0 and target_rect.height() > 0:
            u = (pos.x() - target_rect.x()) / target_rect.width()
            v = (pos.y() - target_rect.y()) / target_rect.height()
            self.crosshair_u = max(0.0, min(1.0, u))
            self.crosshair_v = max(0.0, min(1.0, v))
            self.update()

    def _update_secondary_crosshair_from_pos(self, pos: QPointF):
        target_rect = self._get_target_rect()
        if target_rect.width() > 0 and target_rect.height() > 0:
            u = (pos.x() - target_rect.x()) / target_rect.width()
            v = (pos.y() - target_rect.y()) / target_rect.height()
            self.secondary_crosshair_u = max(0.0, min(1.0, u))
            self.secondary_crosshair_v = max(0.0, min(1.0, v))
            self.has_secondary_crosshair = True
            self.update()


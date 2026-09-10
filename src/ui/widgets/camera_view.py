import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera import CameraModule
from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


class CameraViewWidget(QWidget):
    """Live camera view with crosshairs, FPS tracking, and snapshot capture."""

    def __init__(
        self,
        engine: StepperEngine,
        bridge: QtEngineBridge,
        camera: Optional[CameraModule] = None,
        camera_scale: float = 0.5,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self.camera = camera
        self.camera_scale = camera_scale

        self.current_frame: Optional[np.ndarray] = None
        self.current_qimage: Optional[QImage] = None
        self.show_crosshairs = True

        # FPS metrics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        if self.camera and not self.camera.is_open():
            if not self.camera.open():
                print("Warning: Camera failed to open")

        self._init_ui()

        # Camera polling timer (approx 30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fetch_frame)
        self.timer.start(33)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header bar with controls
        header = QHBoxLayout()
        header.setSpacing(8)

        self.crosshair_cb = QCheckBox("Crosshair")
        self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(self._on_crosshair_toggled)
        header.addWidget(self.crosshair_cb)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        header.addWidget(self.snapshot_btn)

        header.addStretch()

        self.res_label = QLabel("No Camera")
        self.res_label.setStyleSheet("color: #888888; font-size: 11px;")
        header.addWidget(self.res_label)

        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet("color: #888888; font-size: 11px;")
        header.addWidget(self.fps_label)

        layout.addLayout(header)

        # Main viewport canvas
        self.viewport = CameraViewport(self)
        layout.addWidget(self.viewport, stretch=1)

    def _on_crosshair_toggled(self, checked: bool):
        self.show_crosshairs = checked
        self.viewport.update()

    def _take_snapshot(self):
        if self.current_frame is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        capture_dir = Path("stepper_captures")
        capture_dir.mkdir(exist_ok=True)
        filename = capture_dir / f"manual_snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), self.current_frame)
        print(f"Saved snapshot to {filename}")
        self.bridge.status_message.emit(f"Snapshot saved: {filename.name}")

    def cleanup(self):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        if self.camera and self.camera.is_open():
            try:
                self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def _fetch_frame(self):
        if not self.camera:
            return

        try:
            frame = self.camera.get_latest_frame()
        except Exception as e:
            print(f"Error fetching camera frame: {e}")
            return

        if frame is None:
            return

        self.current_frame = frame
        self.engine.set_latest_image(frame)

        # Convert to QImage
        h, w = frame.shape[:2]
        if frame.ndim == 2:
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # OpenCV provides BGR, convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

        self.current_qimage = qimg
        self.res_label.setText(f"{w}x{h}")

        # FPS calculate
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            self.current_fps = self.frame_count / dt
            self.fps_label.setText(f"{self.current_fps:.1f} FPS")
            self.frame_count = 0
            self.last_fps_time = now

        self.viewport.update()


class CameraViewport(QWidget):
    """Subwidget that paints the image with crosshair overlay."""

    def __init__(self, parent_view: CameraViewWidget):
        super().__init__()
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #0d0d11; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor("#0d0d11"))

        qimg = self.parent_view.current_qimage
        if qimg is not None and not qimg.isNull():
            # Scale maintaining aspect ratio
            scaled = qimg.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Camera Feed Available")

        # Draw Crosshair
        if self.parent_view.show_crosshairs:
            cx = self.width() / 2.0
            cy = self.height() / 2.0
            pen = QPen(QColor(0, 255, 128, 180), 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
            painter.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))

            # Center target circle
            pen_solid = QPen(QColor(0, 255, 128, 220), 1.5)
            painter.setPen(pen_solid)
            painter.drawEllipse(QPointF(cx, cy), 15, 15)
            painter.drawEllipse(QPointF(cx, cy), 35, 35)


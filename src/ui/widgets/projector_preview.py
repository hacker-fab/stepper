from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.engine import StepperEngine
from core.events import ColorMode
from ui.bridge import QtEngineBridge


class ProjectorPreviewWidget(QWidget):
    """Monitors what the projector is currently displaying in real time."""

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

        # Status mode label
        self.mode_label = QLabel("Output: Disabled (Off)")
        self.mode_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.mode_label)

        self.canvas = ProjectorCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.projector_image_changed.connect(self._on_image_changed)
        self.bridge.projector_color_mode_changed.connect(lambda *_: self._on_image_changed())

    def _on_image_changed(self, *args):
        color_mode = self.engine.projector.color_mode
        labels = {
            ColorMode.DISABLE: "Output: Disabled (Off)",
            ColorMode.RED: "Output: Red Illumination Active",
            ColorMode.UV: "Output: UV Illumination Active",
        }
        self.mode_label.setText(labels.get(color_mode, str(color_mode)))
        self.canvas.update()


class ProjectorCanvas(QWidget):
    def __init__(self, parent_view: ProjectorPreviewWidget):
        super().__init__()
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #000000; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        img = self.parent_view.engine.projector.current_image
        if img is not None:
            # Convert PIL Image to QPixmap
            if img.mode != "RGBA":
                img_rgba = img.convert("RGBA")
            else:
                img_rgba = img
            data = img_rgba.tobytes("raw", "RGBA")
            qimg = QImage(data, img_rgba.width, img_rgba.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)

            scaled = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#444444"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Output (Screen Off)")


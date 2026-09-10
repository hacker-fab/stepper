from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.engine import StepperEngine
from core.events import ShownImage
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
        self.mode_label = QLabel("Output: Clear (Off)")
        self.mode_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.mode_label)

        self.canvas = ProjectorCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.shown_image_changed.connect(self._on_image_changed)
        self.bridge.pattern_image_changed.connect(lambda: self.canvas.update())
        self.bridge.image_adjust_changed.connect(lambda: self.canvas.update())

    def _on_image_changed(self, shown_image: ShownImage):
        labels = {
            ShownImage.CLEAR: "Output: Clear (No UV/Red)",
            ShownImage.PATTERN: "Output: Pattern (UV Active)",
            ShownImage.RED_FOCUS: "Output: Red Focus Mode",
            ShownImage.UV_FOCUS: "Output: UV Focus Pattern",
            ShownImage.FLATFIELD: "Output: Flatfield Calibration",
        }
        self.mode_label.setText(labels.get(shown_image, str(shown_image)))
        self.canvas.update()


class ProjectorCanvas(QWidget):
    def __init__(self, parent_view: ProjectorPreviewWidget):
        super().__init__()
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #000000; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        img = self.parent_view.engine.current_image
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


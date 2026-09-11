from typing import Optional, Tuple
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow

from projector import ProjectorController


class QtProjectorMeta(type(QMainWindow), type(ProjectorController)):
    pass


class QtProjector(QMainWindow, ProjectorController, metaclass=QtProjectorMeta):
    """Full-screen projector window implemented in PySide6.

    Displays projected mask patterns on the secondary monitor (or falls back to primary).
    """

    _sig_update = Signal(object)

    def __init__(self, title: str = "Projector", background_color: str = "#000000"):
        super().__init__()
        ProjectorController.__init__(self)
        self.setWindowTitle(title)
        self.setStyleSheet(f"background-color: {background_color};")

        # Label to display the projected pattern
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self._sig_update.connect(self._handle_update)

        # Place on secondary screen if multiple screens exist
        screens = QGuiApplication.screens()
        if len(screens) > 1:
            screen = screens[1]
            self.setScreen(screen)
            self.move(screen.geometry().topLeft())
            self.showFullScreen()
        else:
            # Single screen setup: show normal/borderless or full screen
            self.resize(1920, 1080)
            self.showFullScreen()

        self.clear()

    def size(self) -> Tuple[int, int]:
        return (self.width(), self.height())

    def _pil_to_pixmap(self, image: Image.Image) -> QPixmap:
        """Converts a PIL Image to a Qt QPixmap."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)

    def _handle_update(self, image: Optional[Image.Image]):
        if image is None:
            w, h = self.size()
            pixmap = QPixmap(w, h)
            pixmap.fill(QColor("black"))
            self.label.setPixmap(pixmap)
        else:
            pixmap = self._pil_to_pixmap(image)
            self.label.setPixmap(pixmap)

    def show(self, image: Image.Image):
        self._sig_update.emit(image)

    def clear(self):
        self._sig_update.emit(None)


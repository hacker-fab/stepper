from typing import Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QMainWindow, QWidget

from core.events import Event
from projector import ProjectorController


class ProjectorCanvasWidget(QWidget):
    """Low-latency direct-blitting canvas for optical mask projection.

    Bypasses QPixmap server-side conversions and QLabel layout calculations by
    directly painting QImages to the widget surface via QPainter.drawImage with
    guaranteed 1:1 pixel fidelity.
    """

    def __init__(self, parent: Optional[QWidget] = None, background_color: str = "#000000"):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setStyleSheet(f"background-color: {background_color};")
        self._bg_color = QColor(background_color)
        self._qimage: Optional[QImage] = None
        self._is_on: bool = False

    def set_qimage(self, qimage: Optional[QImage]):
        self._qimage = qimage
        self.repaint()

    def set_on_off(self, on: bool):
        self._is_on = on
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)
        if self._is_on and self._qimage is not None and not self._qimage.isNull():
            # Exact 1:1 direct pixel draw without any interpolation or softening
            painter.drawImage(0, 0, self._qimage)


class QtProjectorMeta(type(QMainWindow), type(ProjectorController)):
    pass


class QtProjector(QMainWindow, ProjectorController, metaclass=QtProjectorMeta):
    """Full-screen projector window implemented in PySide6 with custom direct-blitting canvas.

    Displays projected mask patterns on the secondary monitor (or falls back to primary).
    """

    _sig_image_cache_changed = Signal(object)
    _sig_update = Signal()

    def __init__(self, title: str = "Projector", background_color: str = "#000000"):
        super().__init__()
        ProjectorController.__init__(self)
        self.setWindowTitle(title)
        self.setStyleSheet(f"background-color: {background_color};")

        # Custom direct-blitting canvas
        self.canvas = ProjectorCanvasWidget(self, background_color=background_color)
        self.setCentralWidget(self.canvas)

        self._sig_update.connect(self._handle_update)
        self._sig_image_cache_changed.connect(self._handle_image_cache_changed)

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

        self._sig_update.emit()

    def projector_size(self) -> Tuple[int, int]:
        return (self.width(), self.height())

    def _on_display_image_cache_changed(self):
        super()._on_display_image_cache_changed()
        self._sig_image_cache_changed.emit(self._displayed_image_cache)

    def update_display(self):
        """Updates projector state and triggers GUI repaint via thread-safe Qt signal."""
        self.display_ready_event.clear()
        self._sig_update.emit()

    def _ndarray_to_qimage(self, image: np.ndarray) -> QImage:
        """Converts a contiguous NumPy array to a QImage without copying memory."""
        h, w = image.shape[:2]
        contig = np.ascontiguousarray(image)
        if image.ndim == 2:
            return QImage(contig.data, w, h, w, QImage.Format_Grayscale8)
        elif image.shape[2] == 3:
            return QImage(contig.data, w, h, 3 * w, QImage.Format_BGR888)
        elif image.shape[2] == 4:
            return QImage(contig.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return QImage()

    def _handle_image_cache_changed(self, image: Optional[np.ndarray]):
        try:
            if image is None:
                qimg = None
            else:
                qimg = self._ndarray_to_qimage(image)
            self.canvas.set_qimage(qimg)
        finally:
            self.display_ready_event.set()
        

    def _handle_update(self):
        try:
            self.canvas.set_on_off(self.is_on)
        finally:
            self.display_ready_event.set()

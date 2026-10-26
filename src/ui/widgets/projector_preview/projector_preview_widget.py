from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.engine import StepperEngine
from core.events import ColorMode
from ui.bridge import QtEngineBridge
from .projector_canvas import ProjectorCanvas


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
        self.mode_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.mode_label)

        self.canvas = ProjectorCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.projector_image_changed.connect(self._on_image_changed)
        self.bridge.projector_color_mode_changed.connect(lambda *_: self._on_image_changed())
        self.bridge.projector_on_off_changed.connect(lambda *_: self._on_image_changed())

        # Initial refresh
        self._on_image_changed()

    def _on_image_changed(self, *args):
        is_on = self.engine.projector.is_on
        color_mode = self.engine.projector.color_mode
        mode_str = "Red" if color_mode == ColorMode.RED else "UV"
        if is_on:
            self.mode_label.setText(f"Output: ON ({mode_str} Illumination Active)")
            self.mode_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #4CAF50;")
        else:
            self.mode_label.setText(f"Output: OFF ({mode_str} Illumination Ready)")
            self.mode_label.setStyleSheet("font-size: 11px; color: #888888;")
        self.canvas.refresh_image()

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


class ActivityRibbonWidget(QFrame):
    """Middle ribbon displaying overall machine activity, progress, and quick abort."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(36)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            ActivityRibbonWidget {
                background-color: #1e1e24;
                border-top: 1px solid #333333;
                border-bottom: 1px solid #333333;
                padding: 4px;
            }
            QLabel {
                color: #e0e0e0;
                font-weight: bold;
                font-size: 13px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2b2b36;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
            QPushButton#abortBtn {
                background-color: #dc2626;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton#abortBtn:hover {
                background-color: #ef4444;
            }
            QPushButton#abortBtn:disabled {
                background-color: #4b5563;
                color: #9ca3af;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(16)

        # Status icon / text
        self.status_icon = QLabel("●")
        self.status_icon.setStyleSheet("color: #10b981; font-size: 16px;")  # Green for idle/ready
        layout.addWidget(self.status_icon)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Unified adaptive progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar, stretch=1)

        # Quick abort button
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setObjectName("abortBtn")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._on_abort_clicked)
        layout.addWidget(self.abort_btn)

        # Connect signals
        self.bridge.status_message.connect(self.set_status)
        self.bridge.pattern_progress_changed.connect(self._on_progress_changed)
        self.bridge.patterning_busy_changed.connect(self._on_busy_changed)

    def set_status(self, text: str, is_busy: bool = False, is_error: bool = False):
        self.status_label.setText(text)
        if is_error:
            self.status_icon.setStyleSheet("color: #ef4444; font-size: 16px;")
        elif is_busy:
            self.status_icon.setStyleSheet("color: #f59e0b; font-size: 16px;")
        else:
            self.status_icon.setStyleSheet("color: #10b981; font-size: 16px;")

    def _on_progress_changed(self, pattern_progress: float, exposure_progress: float):
        # Overall progress
        pct = int(exposure_progress * 100)
        self.progress_bar.setValue(pct)
        if self.engine.patterning_busy:
            self.set_status(f"Exposing Pattern... ({pct}%)", is_busy=True)

    def _on_busy_changed(self, busy: bool):
        self.abort_btn.setEnabled(busy)
        if not busy:
            self.progress_bar.setValue(0)
            self.set_status("Ready", is_busy=False)

    def _on_abort_clicked(self):
        self.engine.abort_patterning()
        self.set_status("Aborting exposure...", is_busy=True)


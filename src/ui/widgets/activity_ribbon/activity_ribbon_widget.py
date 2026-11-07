from collections import deque
import time
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer
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

        self._progress_history: deque[Tuple[float, float]] = deque()
        self._current_progress: float = 0.0

        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._update_eta)

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
            QLabel#etaLabel {
                color: #9ca3af;
                font-weight: normal;
                font-size: 12px;
                min-width: 75px;
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

        # Remaining time estimation label
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("etaLabel")
        layout.addWidget(self.eta_label)

        # Quick abort button
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setObjectName("abortBtn")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._on_abort_clicked)
        layout.addWidget(self.abort_btn)

        # Connect signals
        self.bridge.status_message.connect(self.set_status)
        self.bridge.operation_started.connect(self._on_operation_started)
        self.bridge.operation_progress.connect(self._on_operation_progress)
        self.bridge.operation_finished.connect(self._on_operation_finished)
        self.bridge.operation_aborted.connect(self._on_operation_aborted)
        self.bridge.operation_failed.connect(self._on_operation_failed)

    @staticmethod
    def format_eta(eta_seconds: Optional[float]) -> str:
        """Formats remaining seconds into ETA string (e.g. 'ETA: 02:45' or 'ETA: 01:12:05')."""
        if eta_seconds is None or eta_seconds < 0:
            return "ETA: --:--"
        if eta_seconds < 3600:
            mins = int(eta_seconds // 60)
            secs = int(eta_seconds % 60)
            return f"ETA: {mins:02d}:{secs:02d}"
        else:
            hrs = int(eta_seconds // 3600)
            mins = int((eta_seconds % 3600) // 60)
            secs = int(eta_seconds % 60)
            return f"ETA: {hrs:02d}:{mins:02d}:{secs:02d}"

    def _update_eta(self):
        """Calculates and updates remaining time estimation based on progress in the past minute."""
        now = time.monotonic()
        self._progress_history.append((now, self._current_progress))

        # Prune samples older than 60 seconds
        cutoff = now - 60.0
        while len(self._progress_history) > 1 and self._progress_history[0][0] < cutoff:
            self._progress_history.popleft()

        t_old, p_old = self._progress_history[0]
        t_new, p_new = self._progress_history[-1]
        dt = t_new - t_old
        dp = p_new - p_old

        if p_new >= 1.0:
            self.eta_label.setText("ETA: 00:00")
            return

        if dt < 2.0 or dp <= 1e-5:
            self.eta_label.setText("ETA: --:--")
            return

        speed = dp / dt
        rem = max(0.0, 1.0 - p_new)
        eta_seconds = rem / speed
        self.eta_label.setText(self.format_eta(eta_seconds))

    def set_status(self, text: str, is_busy: bool = False, is_error: bool = False):
        self.status_label.setText(text)
        if is_error:
            self.status_icon.setStyleSheet("color: #ef4444; font-size: 16px;")
        elif is_busy:
            self.status_icon.setStyleSheet("color: #f59e0b; font-size: 16px;")
        else:
            self.status_icon.setStyleSheet("color: #10b981; font-size: 16px;")

    def _on_operation_started(self, name: str):
        self.abort_btn.setEnabled(self.engine.operations.current_operation is not None)
        self.progress_bar.setValue(0)
        self._current_progress = 0.0
        self._progress_history.clear()
        self._progress_history.append((time.monotonic(), 0.0))
        self.eta_label.setText("ETA: --:--")
        self._eta_timer.start(1000)
        self.set_status(f"Running: {name}...", is_busy=True)

    def _on_operation_progress(self, progress: float, message: str):
        pct = int(progress * 100)
        self.progress_bar.setValue(pct)
        self._current_progress = max(0.0, min(1.0, float(progress)))
        self.set_status(message if message else f"Progress: {pct}%", is_busy=True)

    def _on_operation_finished(self, name: str):
        self._eta_timer.stop()
        self._progress_history.clear()
        self.eta_label.setText("")
        self.abort_btn.setEnabled(self.engine.operations.current_operation is not None)
        self.progress_bar.setValue(0)
        self.set_status(f"Ready ({name} finished)", is_busy=False)

    def _on_operation_aborted(self, name: str):
        self._eta_timer.stop()
        self._progress_history.clear()
        self.eta_label.setText("")
        self.abort_btn.setEnabled(self.engine.operations.current_operation is not None)
        self.progress_bar.setValue(0)
        self.set_status(f"Aborted: {name}", is_busy=False, is_error=True)

    def _on_operation_failed(self, name: str, error: str):
        self._eta_timer.stop()
        self._progress_history.clear()
        self.eta_label.setText("")
        self.abort_btn.setEnabled(self.engine.operations.current_operation is not None)
        self.progress_bar.setValue(0)
        self.set_status(f"Failed ({name}): {error}", is_busy=False, is_error=True)

    def _on_abort_clicked(self):
        self.engine.abort_operation()
        self.set_status("Aborting operation...", is_busy=True)

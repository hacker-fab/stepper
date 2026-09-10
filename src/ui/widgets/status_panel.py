from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from core.events import MovementLock, ShownImage
from ui.bridge import QtEngineBridge


class StatusPanelWidget(QTabWidget):
    """Grouped panel for live status, exposure history, and system console logs."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        # Tab 1: Live Status
        self.status_tab = QWidget()
        self._setup_status_tab()
        self.addTab(self.status_tab, "Live Status")

        # Tab 2: Exposure History
        self.history_tab = QWidget()
        self._setup_history_tab()
        self.addTab(self.history_tab, "Exposure History")

        # Tab 3: System Logs
        self.logs_tab = QWidget()
        self._setup_logs_tab()
        self.addTab(self.logs_tab, "System Logs")

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_pos_changed)
        self.bridge.shown_image_changed.connect(self._on_mode_changed)
        self.bridge.movement_lock_changed.connect(self._on_lock_changed)
        self.bridge.chip_changed.connect(lambda _: self._refresh_history())
        self.bridge.warning_emitted.connect(self._log_warning)
        self.bridge.status_message.connect(self._log_message)

    # -------------------------------------------------------------------------
    # Tab 1: Live Status
    # -------------------------------------------------------------------------
    def _setup_status_tab(self):
        layout = QVBoxLayout(self.status_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Coordinate Group
        pos_group = QGroupBox("Coordinates")
        pos_form = QFormLayout(pos_group)
        pos_form.setSpacing(4)

        self.lbl_x = QLabel("0.000 mm")
        self.lbl_x.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 13px;")
        pos_form.addRow("Work X:", self.lbl_x)

        self.lbl_y = QLabel("0.000 mm")
        self.lbl_y.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 13px;")
        pos_form.addRow("Work Y:", self.lbl_y)

        self.lbl_z = QLabel("0.000 mm")
        self.lbl_z.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 13px;")
        pos_form.addRow("Work Z:", self.lbl_z)

        layout.addWidget(pos_group)

        # State Group
        state_group = QGroupBox("System State")
        state_form = QFormLayout(state_group)
        state_form.setSpacing(4)

        self.lbl_mode = QLabel("CLEAR")
        self.lbl_mode.setStyleSheet("font-weight: bold;")
        state_form.addRow("Mode:", self.lbl_mode)

        self.lbl_lock = QLabel("UNLOCKED")
        self.lbl_lock.setStyleSheet("font-weight: bold; color: #10b981;")
        state_form.addRow("Movement:", self.lbl_lock)

        self.lbl_homing = QLabel("Ready" if not self.engine.hardware.stage.has_homing() else "Sensors Enabled")
        state_form.addRow("Homing:", self.lbl_homing)

        layout.addWidget(state_group)
        layout.addStretch()

    def _on_pos_changed(self, coords: tuple):
        x, y, z = coords
        self.lbl_x.setText(f"{x:.3f} mm")
        self.lbl_y.setText(f"{y:.3f} mm")
        self.lbl_z.setText(f"{z:.3f} mm")

    def _on_mode_changed(self, shown_image: ShownImage):
        self.lbl_mode.setText(shown_image.name)

    def _on_lock_changed(self, lock: MovementLock):
        self.lbl_lock.setText(lock.name)
        if lock == MovementLock.LOCKED:
            self.lbl_lock.setStyleSheet("font-weight: bold; color: #ef4444;")
        else:
            self.lbl_lock.setStyleSheet("font-weight: bold; color: #10b981;")

    # -------------------------------------------------------------------------
    # Tab 2: Exposure History
    # -------------------------------------------------------------------------
    def _setup_history_tab(self):
        layout = QVBoxLayout(self.history_tab)
        layout.setContentsMargins(4, 4, 4, 4)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Time", "Pattern", "Coordinates", "Duration", "Status"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.history_table)

    def _refresh_history(self):
        logs = self.engine.exposure_history
        self.history_table.setRowCount(len(logs))
        for i, log in enumerate(reversed(logs)):
            row = len(logs) - 1 - i
            self.history_table.setItem(row, 0, QTableWidgetItem(log.time.strftime("%H:%M:%S")))
            self.history_table.setItem(row, 1, QTableWidgetItem(log.path.split("/")[-1]))
            x, y, z = log.coords
            self.history_table.setItem(row, 2, QTableWidgetItem(f"({x:.1f}, {y:.1f}, {z:.1f})"))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{log.duration / 1000.0:.1f} s"))
            status_item = QTableWidgetItem("Aborted" if log.aborted else "Success")
            if log.aborted:
                status_item.setForeground(Qt.red)
            else:
                status_item.setForeground(Qt.green)
            self.history_table.setItem(row, 4, status_item)

    # -------------------------------------------------------------------------
    # Tab 3: System Logs
    # -------------------------------------------------------------------------
    def _setup_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #0b0b0f; color: #d1d5db; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_text, stretch=1)

        btn_clear = QPushButton("Clear Console")
        btn_clear.clicked.connect(self.log_text.clear)
        layout.addWidget(btn_clear)

    def _log_message(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{t}] {msg}")

    def _log_warning(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{t}] [WARNING] {msg}")

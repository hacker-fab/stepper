from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from core.engine import StepperEngine
from core.lithography import LithographerConfig
from ui.bridge import QtEngineBridge
from ui.widgets import (
    ActivityRibbonWidget,
    CameraViewWidget,
    DataPanelWidget,
    ProcessControlPanelWidget,
    ProjectorPreviewWidget,
    StageMapWidget,
    StatusPanelWidget,
)


class MainWindow(QMainWindow):
    """Main Application Window for Hacker Fab Stepper V2 using PySide6 QDockWidgets."""

    def __init__(self, config: LithographerConfig, engine: StepperEngine, bridge: QtEngineBridge):
        super().__init__()
        self.config = config
        self.engine = engine
        self.bridge = bridge

        self.setWindowTitle("Hacker Fab - Stepper V2")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # Enable multi-dock nesting (side-by-side splitting in both dimensions)
        self.setDockNestingEnabled(True)

        # Apply dark theme styling
        self._apply_theme()

        # 1. Middle Persistent Activity Ribbon (Fixed height, non-expanding)
        self.activity_ribbon = ActivityRibbonWidget(self.engine, self.bridge, self)
        self.activity_ribbon.setFixedHeight(36)
        self.setCentralWidget(self.activity_ribbon)

        # 2. Create the 6 Docks (All Non-Closable)
        self._create_docks()

        # 3. Setup Menu Bar & Status Bar
        self._setup_menus()
        self._setup_status_bar()

        # 4. Lay out the docks
        self._reset_dock_layout()

        # 5. Connect bridge warnings to modal/status alerts
        self.bridge.warning_emitted.connect(self._show_warning_dialog)

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #18181b;
            }
            QDockWidget {
                color: #f4f4f5;
                font-weight: bold;
                font-size: 12px;
                titlebar-close-icon: none;
            }
            QDockWidget::title {
                background-color: #27272a;
                padding: 6px;
                border-bottom: 1px solid #3f3f46;
                border-radius: 2px;
            }
            QTabWidget::pane {
                border: 1px solid #3f3f46;
                background-color: #18181b;
            }
            QTabBar::tab {
                background-color: #27272a;
                color: #a1a1aa;
                padding: 6px 14px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3f3f46;
                color: #ffffff;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #3f3f46;
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
                color: #e4e4e7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #3f3f46;
                color: #f4f4f5;
                border-radius: 4px;
                padding: 5px 12px;
                border: 1px solid #52525b;
            }
            QPushButton:hover {
                background-color: #52525b;
            }
            QPushButton:pressed {
                background-color: #27272a;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #27272a;
                color: #f4f4f5;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 4px;
            }
            QTableWidget {
                background-color: #18181b;
                color: #f4f4f5;
                gridline-color: #27272a;
                border: 1px solid #3f3f46;
            }
            QHeaderView::section {
                background-color: #27272a;
                color: #d4d4d8;
                padding: 4px;
                border: 1px solid #3f3f46;
            }
            """
        )

    def _create_dock(self, title: str, widget: QWidget, obj_name: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(obj_name)
        dock.setWidget(widget)
        # Explicitly movable and floatable, NOT closable per user instruction
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return dock

    def _create_docks(self):
        # Top Row Widgets
        self.stage_map_widget = StageMapWidget(self.engine, self.bridge, self)
        self.dock_map = self._create_dock("Stage Map", self.stage_map_widget, "dock_stage_map")

        self.camera_widget = CameraViewWidget(
            self.engine, self.bridge, self.config.camera, self.config.camera_scale, self
        )
        self.dock_camera = self._create_dock("Camera View", self.camera_widget, "dock_camera")

        self.projector_preview_widget = ProjectorPreviewWidget(self.engine, self.bridge, self)
        self.dock_proj = self._create_dock(
            "Projector Preview", self.projector_preview_widget, "dock_projector_preview"
        )

        # Bottom Row Widgets
        self.data_panel_widget = DataPanelWidget(self.engine, self.bridge, self)
        self.dock_data = self._create_dock("Data & Files", self.data_panel_widget, "dock_data")

        self.process_panel_widget = ProcessControlPanelWidget(self.engine, self.bridge, self)
        self.dock_process = self._create_dock(
            "Process & Actions", self.process_panel_widget, "dock_process"
        )

        self.status_panel_widget = StatusPanelWidget(self.engine, self.bridge, self)
        self.dock_status = self._create_dock(
            "Status & Logs", self.status_panel_widget, "dock_status"
        )

    def _setup_menus(self):
        menubar = self.menuBar()

        # View / Layout Menu
        layout_menu = menubar.addMenu("&Layout")

        act_reset = QAction("Reset to Default Layout", self)
        act_reset.triggered.connect(self._reset_dock_layout)
        layout_menu.addAction(act_reset)

        act_save = QAction("Save Current Layout", self)
        act_save.triggered.connect(self._save_dock_layout)
        layout_menu.addAction(act_save)

    def _setup_status_bar(self):
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        stage_status = "Stage: Connected" if self.engine.hardware.stage else "Stage: None"
        cam_status = "Camera: Active" if self.config.camera else "Camera: None"
        dlpc_status = "DLPC: Ready" if self.config.dlpc else "DLPC: None"

        status_bar.showMessage(f"{stage_status}  |  {cam_status}  |  {dlpc_status}")

    def _reset_dock_layout(self):
        """Places the 6 docks in the default 3-top / 3-bottom configuration."""
        # Top Row
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_map)
        self.splitDockWidget(self.dock_map, self.dock_camera, Qt.Horizontal)
        self.splitDockWidget(self.dock_camera, self.dock_proj, Qt.Horizontal)

        # Bottom Row
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_data)
        self.splitDockWidget(self.dock_data, self.dock_process, Qt.Horizontal)
        self.splitDockWidget(self.dock_process, self.dock_status, Qt.Horizontal)

        # Set default relative dimensions (equal thirds horizontally, equal halves vertically)
        self.resizeDocks([self.dock_map, self.dock_camera, self.dock_proj], [350, 600, 350], Qt.Horizontal)
        self.resizeDocks([self.dock_data, self.dock_process, self.dock_status], [400, 500, 400], Qt.Horizontal)
        self.resizeDocks([self.dock_camera, self.dock_process], [450, 450], Qt.Vertical)

    def _save_dock_layout(self):
        settings = QSettings("HackerFab", "StepperV2")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        self.statusBar().showMessage("Layout preferences saved", 3000)

    def _show_warning_dialog(self, msg: str):
        QMessageBox.warning(self, "Warning", msg)

    def closeEvent(self, event):
        self._save_dock_layout()
        # Auto-close projector window if open
        if hasattr(self.engine, "hardware") and hasattr(self.engine.hardware, "projector"):
            proj = self.engine.hardware.projector
            if proj is not None and hasattr(proj, "close"):
                try:
                    proj.close()
                except Exception as e:
                    print(f"Error closing projector: {e}")
        event.accept()


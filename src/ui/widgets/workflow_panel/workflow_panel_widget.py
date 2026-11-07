from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge
from .project_subpanel import ProjectSubpanelWidget
from .layer_subpanel import LayerSubpanelWidget
from .action_subpanel import ActionSubpanelWidget


class WorkflowPanelWidget(QWidget):
    """Main workflow dock container using a horizontal QSplitter: (project | layer | action)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        # Horizontal Splitter: [project | layer | action]
        self.splitter = QSplitter(Qt.Horizontal, self)
        main_layout.addWidget(self.splitter)

        # 1. Project Portion
        self.project_subpanel = ProjectSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.project_subpanel)

        # 2. Layer Portion
        self.layer_subpanel = LayerSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.layer_subpanel)

        # 3. Action Portion
        self.action_subpanel = ActionSubpanelWidget(self.engine, self.bridge, self)
        self.splitter.addWidget(self.action_subpanel)

        # Set default stretch / proportional widths
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 3)
        self.splitter.setSizes([320, 420, 320])

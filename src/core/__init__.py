"""Core lithography process package for Hacker Fab Stepper.

Contains the StepperEngine, base EngineModule, ChipProject data models,
Event bus, and Operation execution framework.
"""

from .chip_project import (
    ChipLayer,
    ChipProject,
    LayerSettingsOverride,
    PatterningSettings,
)
from .engine import StepperEngine
from .engine_module import EngineModule
from .events import (
    ColorMode,
    Event,
    EventBus,
    ProjectorImageSource,
    StrAutoEnum,
)
from .operation import (
    ExecutionContext,
    Operation,
    OperationManager,
)

__all__ = [
    "StepperEngine",
    "EngineModule",
    "ChipProject",
    "ChipLayer",
    "PatterningSettings",
    "LayerSettingsOverride",
    "Event",
    "EventBus",
    "ColorMode",
    "ProjectorImageSource",
    "StrAutoEnum",
    "Operation",
    "ExecutionContext",
    "OperationManager",
]

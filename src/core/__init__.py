"""Core lithography process package for Hacker Fab Stepper.

Contains the StepperEngine, base EngineModule, ChipProject data models,
Event bus, and Operation execution framework.
"""

from .chip_project import (
    ChipLayer,
    ChipProgress,
    ChipProject,
    ExposureRecord,
    LayerSettingsOverride,
    PatterningSettings,
)
from .engine import StepperEngine
from .engine_module import EngineModule
from .events import (
    CameraFlipMode,
    ColorMode,
    Event,
    EventBus,
    ProjectorImageSource,
    StrAutoEnum,
)
from .machine_config import (
    AutofocusMachineConfig,
    AutofocusStageConfig,
    ExposureMachineConfig,
    MachineConfig,
    OpticsMachineConfig,
    StageCalibrationMachineConfig,
)
from .operation import (
    ExecutionContext,
    Operation,
    OperationManager,
    sub_progress,
)

__all__ = [
    "StepperEngine",
    "EngineModule",
    "ChipProject",
    "ChipProgress",
    "ExposureRecord",
    "ChipLayer",
    "PatterningSettings",
    "LayerSettingsOverride",
    "Event",
    "EventBus",
    "CameraFlipMode",
    "ColorMode",
    "ProjectorImageSource",
    "StrAutoEnum",
    "Operation",
    "ExecutionContext",
    "OperationManager",
    "sub_progress",
    "MachineConfig",
    "AutofocusMachineConfig",
    "AutofocusStageConfig",
    "OpticsMachineConfig",
    "ExposureMachineConfig",
    "StageCalibrationMachineConfig",
]

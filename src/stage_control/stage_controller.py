# Hacker Fab
# J. Kent Wirant
# Stage Controller Interface


from core.engine_module import EngineModule
from abc import ABC, abstractmethod
from typing import Optional


class UnsupportedCommand(Exception):
    pass


# This will be an abstract interface for stage positioning.
class StageController(EngineModule):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def has_homing(self) -> bool:
        pass

    @abstractmethod
    def home(self) -> bool:
        pass

    @abstractmethod
    def move_relative(self, microns: dict[str, float]) -> bool:
        pass

    @abstractmethod
    def move_absolute(self, microns: dict[str, float]) -> bool:
        pass

    @abstractmethod
    def get_position(self) -> tuple[float, float, float]:
        pass

    @abstractmethod
    def get_bounds(self) -> Optional[dict[str, tuple[float, float]]]:
        pass



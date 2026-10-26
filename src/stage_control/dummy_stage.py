from core.events import Event

import math
import time
from typing import Callable, Optional

from stage_control.stage_controller import StageController


class DummyStage(StageController):
    """Simulated stage controller that maintains internal position state and delays movements."""

    def __init__(
        self,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        delay: float = 0.01,
        speed: Optional[float] = None,
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        delay_func: Callable[[float], None] = time.sleep,
    ):
        super().__init__()
        self._position: list[float] = [
            float(initial_position[0]),
            float(initial_position[1]),
            float(initial_position[2]),
        ]
        self.delay: float = delay
        self.speed: Optional[float] = speed
        self.delay_func: Callable[[float], None] = delay_func

        if bounds is None:
            self._bounds = {
                "x": (-50000.0, 50000.0),
                "y": (-50000.0, 50000.0),
                "z": (-10000.0, 10000.0),
            }
        else:
            self._bounds = bounds

    def _simulate_move(self, target_x: float, target_y: float, target_z: float):
        dx = target_x - self._position[0]
        dy = target_y - self._position[1]
        dz = target_z - self._position[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist > 0:
            if self.speed is not None and self.speed > 0:
                duration = dist / self.speed
            else:
                duration = self.delay
            if duration > 0:
                self.delay_func(duration)
        self._position = [target_x, target_y, target_z]

        if self.event_bus is not None:
            self.event_bus.emit(Event.STAGE_POSITION_CHANGED, self.get_position())

    def has_homing(self) -> bool:
        return True

    def home(self) -> bool:
        self._simulate_move(0.0, 0.0, 0.0)
        return True

    def move_relative(self, microns: dict[str, float]) -> bool:
        dx = microns.get("x", microns.get("X", 0.0))
        dy = microns.get("y", microns.get("Y", 0.0))
        dz = microns.get("z", microns.get("Z", 0.0))
        self._simulate_move(
            self._position[0] + dx,
            self._position[1] + dy,
            self._position[2] + dz,
        )
        return True

    def move_absolute(self, microns: dict[str, float]) -> bool:
        target_x = microns.get("x", microns.get("X", self._position[0]))
        target_y = microns.get("y", microns.get("Y", self._position[1]))
        target_z = microns.get("z", microns.get("Z", self._position[2]))
        self._simulate_move(target_x, target_y, target_z)
        return True


    def get_position(self) -> tuple[float, float, float]:
        return (self._position[0], self._position[1], self._position[2])

    def get_bounds(self) -> Optional[dict[str, tuple[float, float]]]:
        return dict(self._bounds) if self._bounds is not None else None

# Hacker Fab
# Casey Honaker
# Open Micro Manipulator Stage Controller

from typing import Optional
from dataclasses import dataclass
import math

try:
    from open_micro_stage_api import OpenMicroStageInterface
except ImportError:
    OpenMicroStageInterface = None

from stage_control.stage_controller import StageController
from core.events import Event

DEFAULT_FEED_RATE = 10.0
BLOCKING_MOVE = True


@dataclass
class Position:
    x: float
    y: float
    z: float

    def update(self, x: float, y: float, z: float):
        if x is not None and not math.isnan(x):
            self.x = x
        if y is not None and not math.isnan(y):
            self.y = y
        if z is not None and not math.isnan(z):
            self.z = z


class OMMStage(StageController):
    """Open Micro Manipulator stage controller implementation."""

    def __init__(self, z_max: float = -1.0):
        super().__init__()
        self._current_position: Position = Position(0.0, 0.0, 0.0)
        self.omm = OpenMicroStageInterface(False, False) if OpenMicroStageInterface else None
        self._z_max = z_max
        self.port = ""
        self.baud_rate = 921600

    def connect(self, port: str, baud_rate: int = 921600):
        """Connect to the OMM stage."""
        self.port = port
        self.baud_rate = baud_rate
        if self.omm is not None:
            self.omm.connect(port, baud_rate)
            self._update_position()

    def disconnect(self):
        """Disconnect from the OMM stage."""
        if self.omm is not None:
            self.omm.disconnect()

    def _update_position(self):
        """Update the internal position cache and emit position changed event."""
        if self.omm is not None:
            x, y, z = self.omm.read_current_position()
            self._current_position.update(x, y, z)
        if self.event_bus is not None:
            self.event_bus.emit(Event.STAGE_POSITION_CHANGED, self.get_position())

    def has_homing(self) -> bool:
        """Check if the stage supports homing."""
        return True

    def home(self) -> bool:
        """Home all axes on the stage."""
        if self.omm is None:
            return False
        res = self.omm.home()
        if res == self.omm.serial.ReplyStatus.OK:
            self._update_position()
            return True
        return False

    def move_relative(self, microns: dict[str, float]) -> bool:
        self._update_position()

        x_um = microns.get("x", microns.get("X", 0.0))
        y_um = microns.get("y", microns.get("Y", 0.0))
        z_um = microns.get("z", microns.get("Z", 0.0))
        f = microns.get("F", DEFAULT_FEED_RATE)

        x_mm = self._current_position.x + (x_um / 1000.0)
        y_mm = self._current_position.y + (y_um / 1000.0)
        z_mm = self._current_position.z + (z_um / 1000.0)

        return self._move_to(Position(x_mm, y_mm, z_mm), f)

    def move_absolute(self, microns: dict[str, float]) -> bool:
        self._update_position()

        x_um = microns.get("x", microns.get("X", None))
        y_um = microns.get("y", microns.get("Y", None))
        z_um = microns.get("z", microns.get("Z", None))
        f = microns.get("F", DEFAULT_FEED_RATE)

        x_mm = self._current_position.x if x_um is None else (x_um / 1000.0)
        y_mm = self._current_position.y if y_um is None else (y_um / 1000.0)
        z_mm = self._current_position.z if z_um is None else (z_um / 1000.0)

        return self._move_to(Position(x_mm, y_mm, z_mm), f)

    def get_position(self) -> tuple[float, float, float]:
        if self.omm is not None:
            x, y, z = self.omm.read_current_position()
            self._current_position.update(x, y, z)
        return (
            self._current_position.x * 1000.0,
            self._current_position.y * 1000.0,
            self._current_position.z * 1000.0,
        )

    def _move_to(self, pos: Position, feed_rate: float) -> bool:
        """Internal method to move to a specific position."""
        if self.omm is None:
            return False
        res = self.omm.move_to(pos.x, pos.y, pos.z, feed_rate, blocking=BLOCKING_MOVE)
        self.omm.wait_for_stop()
        self._update_position()
        return bool(res)

    def get_bounds(self) -> Optional[dict[str, tuple[float, float]]]:
        return {
            "x": (-12 * 1000.0, 12 * 1000.0),
            "y": (-12 * 1000.0, 12 * 1000.0),
            "z": (-12 * 1000.0, self._z_max * 1000.0),
        }
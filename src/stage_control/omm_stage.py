# Hacker Fab
# Casey Honaker
# Open Micro Manipulator Stage Controller

from open_micro_stage_api import OpenMicroStageInterface
from stage_control.stage_controller import StageController
from dataclasses import dataclass
import math

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
    
    def __init__(self):
        """
        Initialize the OMM stage controller.
        """
        self._current_position: Position = Position(0.0, 0.0, 0.0)
        self.omm = OpenMicroStageInterface(False, False)
    
    def connect(self, port: str, baud_rate: int = 921600):
        """Connect to the OMM stage."""
        self.port = port
        self.baud_rate = baud_rate
        self.omm.connect(port, baud_rate)
        self._update_position()
    
    def disconnect(self):
        """Disconnect from the OMM stage."""
        self.omm.disconnect()
    
    def _update_position(self):
        """Update the internal position cache."""
        x, y, z = self.omm.read_current_position()
        self._current_position.update(x, y, z)
    
    def move_by(self, amounts: dict[str, float]):
        """
        Move the stage by relative amounts.
        
        :param amounts: Dictionary with keys like 'X', 'Y', 'Z' and float values
        """
        self._update_position()
        
        x = self._current_position.x + amounts.get("X", amounts.get("x", 0.0))
        y = self._current_position.y + amounts.get("Y", amounts.get("y", 0.0))
        z = self._current_position.z + amounts.get("Z", amounts.get("z", 0.0))
        f = amounts.get("F", DEFAULT_FEED_RATE)

        self._move_to(Position(x, y, z), f)
    
    def move_to(self, amounts: dict[str, float]):
        """
        Move the stage to an absolute position.
        
        :param amounts: Dictionary with keys like 'X', 'Y', 'Z' and float values
        """
        self._update_position()

        x = amounts.get("X", amounts.get("x", self._current_position.x))
        y = amounts.get("Y", amounts.get("y", self._current_position.y))
        z = amounts.get("Z", amounts.get("z", self._current_position.z))
        f = amounts.get("F", DEFAULT_FEED_RATE)

        self._move_to(Position(x, y, z), f)
        
    
    def has_homing(self):
        """Check if the stage supports homing."""
        return True
    
    def home(self):
        """Home all axes on the stage."""
        res = self.omm.home()
        if res == self.omm.serial.ReplyStatus.OK:
            self._update_position()
    
    
    def move_relative(self, microns: dict[str, float]):
        self._update_position()

        x_um = microns.get("x", microns.get("X", 0.0))
        y_um = microns.get("y", microns.get("Y", 0.0))
        z_um = microns.get("z", microns.get("Z", 0.0))
        f = microns.get("F", DEFAULT_FEED_RATE)

        x_mm = self._current_position.x + (x_um / 1000.0)
        y_mm = self._current_position.y + (y_um / 1000.0)
        z_mm = self._current_position.z + (z_um / 1000.0)

        self._move_to(Position(x_mm, y_mm, z_mm), f)

    def move_absolute(self, microns: dict[str, float]):
        self._update_position()

        x_um = microns.get("x", microns.get("X", None))
        y_um = microns.get("y", microns.get("Y", None))
        z_um = microns.get("z", microns.get("Z", None))
        f = microns.get("F", DEFAULT_FEED_RATE)

        x_mm = self._current_position.x if x_um is None else (x_um / 1000.0)
        y_mm = self._current_position.y if y_um is None else (y_um / 1000.0)
        z_mm = self._current_position.z if z_um is None else (z_um / 1000.0)

        self._move_to(Position(x_mm, y_mm, z_mm), f)

    def soft_reset(self):
        print(f"ignoring soft_reset in dummy_stage controller")
    
    def set_on_start_location(self):
        print(f"ignoring set_on_start_location in dummy_stage controller")

    def get_autofocus(self):
        return self.get_position()[2]

    def get_position(self):
        self._update_position()
        return (
            self._current_position.x * 1000.0,
            self._current_position.y * 1000.0,
            self._current_position.z * 1000.0,
        )
    
    def get_on_start_location(self):
        print(f"ignoring get_on_start_location in dummy_stage controller")

    def _move_to(self, pos: Position, feed_rate: float):
        """Internal method to move to a specific position."""
        res = self.omm.move_to(pos.x, pos.y, pos.z, feed_rate, blocking=BLOCKING_MOVE)
        self.omm.wait_for_stop()
        self._update_position()
        return res

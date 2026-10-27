# Hacker Fab
# Casey Honaker
# Open Micro Manipulator Stage Controller

import math
from dataclasses import dataclass

from open_micro_stage_api import OpenMicroStageInterface

from stage_control.stage_controller import StageController

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

    def __init__(
        self,
        autofocus_offset,
        z_max: float,
        flip_xy: bool = False,
    ):
        """
        Initialize the OMM stage controller.

        flip_xy optionally converts between the GUI's axes (X right, Y up in the camera view)
        and this stage's own axes; see _set_axis_map. It defaults to False, which leaves every
        coordinate untouched.
        """
        self._set_axis_map(flip_xy)
        self._current_position: Position = Position(0.0, 0.0, 0.0)
        self.omm = OpenMicroStageInterface(False, False)
        self._autofocus_offset = autofocus_offset
        self._z_max = z_max

    def _set_axis_map(self, flip_xy: bool):
        """
        Builds the GUI <-> stage axis tables. With flip_xy the stage is treated as swapped and
        reversed on both axes relative to the camera view (+X moves the view down, +Y moves it
        left): GUI X drives stage Y, GUI Y drives stage X, and both stage axes run backwards,
        so stage_x = -gui_y and stage_y = -gui_x. Without it every value passes through as is.
        """
        swap_xy = invert_x = invert_y = flip_xy
        self._axis = {"x": "y", "y": "x"} if swap_xy else {"x": "x", "y": "y"}
        self._sign = {"x": -1.0 if invert_x else 1.0, "y": -1.0 if invert_y else 1.0}
        if flip_xy:
            print("X/Y flip active: GUI axes are mapped to the stage's swapped, reversed axes")

    def _to_stage_axes(self, coords: dict[str, float]) -> dict[str, float]:
        """GUI coordinates (or deltas) -> stage coordinates. Missing axes stay missing."""
        out: dict[str, float] = {}
        for key, value in coords.items():
            gui_axis = key.lower()
            if gui_axis in self._axis:
                stage_axis = self._axis[gui_axis]
                out[stage_axis] = self._sign[stage_axis] * value
            else:
                out[key] = value
        return out

    def _to_gui_position(self, pos: tuple[float, float, float]) -> tuple[float, float, float]:
        """Stage (x, y, z) -> GUI (x, y, z)."""
        stage = {"x": pos[0], "y": pos[1]}
        gui_x = self._sign[self._axis["x"]] * stage[self._axis["x"]]
        gui_y = self._sign[self._axis["y"]] * stage[self._axis["y"]]
        return (gui_x, gui_y, pos[2])

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
        amounts = self._to_stage_axes(amounts)

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
        amounts = self._to_stage_axes(amounts)

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

        if self.omm.serial is None:
            print("Serial interface is not initialized. Cannot home the stage.")
            return

        if res == self.omm.serial.ReplyStatus.OK:
            self._update_position()

    def move_relative(self, microns: dict[str, float]):
        self._update_position()
        microns = self._to_stage_axes(microns)

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
        microns = self._to_stage_axes(microns)

        x_um = microns.get("x", microns.get("X", None))
        y_um = microns.get("y", microns.get("Y", None))
        z_um = microns.get("z", microns.get("Z", None))
        f = microns.get("F", DEFAULT_FEED_RATE)

        x_mm = self._current_position.x if x_um is None else (x_um / 1000.0)
        y_mm = self._current_position.y if y_um is None else (y_um / 1000.0)
        z_mm = self._current_position.z if z_um is None else (z_um / 1000.0)

        self._move_to(Position(x_mm, y_mm, z_mm), f)

    def soft_reset(self):
        print("ignoring soft_reset in dummy_stage controller")

    def set_on_start_location(self):
        print("ignoring set_on_start_location in dummy_stage controller")

    def get_autofocus(self):
        print(f"current autofocus value: {self._autofocus_offset}")
        return self._autofocus_offset

    def get_position(self):
        self._update_position()
        return self._to_gui_position(
            (
                self._current_position.x * 1000.0,
                self._current_position.y * 1000.0,
                self._current_position.z * 1000.0,
            )
        )

    def get_on_start_location(self):
        print("ignoring get_on_start_location in dummy_stage controller")

    def _move_to(self, pos: Position, feed_rate: float):
        """Internal method to move to a specific position."""
        res = self.omm.move_to(pos.x, pos.y, pos.z, feed_rate, blocking=BLOCKING_MOVE)
        self.omm.wait_for_stop()
        self._update_position()
        return res

    def get_bounds(self):
        stage_bounds = {
            "x": (-12 * 1000.0, 12 * 1000.0),
            "y": (-12 * 1000.0, 12 * 1000.0),
        }
        bounds = {"z": (-12 * 1000.0, self._z_max * 1000.0)}
        for gui_axis in ("x", "y"):
            stage_axis = self._axis[gui_axis]
            lo, hi = stage_bounds[stage_axis]
            if self._sign[stage_axis] < 0:
                lo, hi = -hi, -lo
            bounds[gui_axis] = (lo, hi)
        return {"x": bounds["x"], "y": bounds["y"], "z": bounds["z"]}

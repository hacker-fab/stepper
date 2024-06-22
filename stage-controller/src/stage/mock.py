import logging
import time
from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

from stage.base import BaseStage
from utilities import axes_to_array


class MissingStage(BaseStage):
    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        BaseStage.__init__(self)
        self._position = [0, 0, 0]
        self._n_axis = 3
        self._backlash = None

    @property
    def state(self):
        """The general state dictionary of the board."""
        state = {"position": self.position_map}
        return state

    @property
    def configuration(self):
        return {}

    def update_settings(self, config: dict):
        """Update settings from a config dictionary"""
        # Set backlash. Expects a dictionary with axis labels
        if "backlash" in config:
            # Construct backlash array
            backlash = axes_to_array(config["backlash"], ["x", "y", "z"], [0, 0, 0])
            self.backlash = backlash

    def read_settings(self) -> dict:
        """Return the current settings as a dictionary"""
        blsh = self.backlash.tolist()
        config = {"backlash": {"x": blsh[0], "y": blsh[1], "z": blsh[2]}}
        return config

    @property
    def n_axes(self):
        return self._n_axis

    @property
    def position(self):
        return self._position

    @property
    def backlash(self):
        if self._backlash is not None:
            return self._backlash
        else:
            return np.array([0] * self.n_axes)

    @backlash.setter
    def backlash(self, blsh):
        if blsh is None:
            self._backlash = None
        elif isinstance(blsh, Iterable):
            assert len(blsh) == self.n_axes
            self._backlash = np.array(blsh)
        else:
            self._backlash = np.array([int(blsh)] * self.n_axes, dtype=np.int)

    def move_rel(
        self,
        displacement: Union[int, Tuple[int, int, int]],
        axis: Optional[Literal["x", "y", "z"]] = None,
        backlash: bool = True,
    ):
        time.sleep(0.5)
        if axis:
            # Displacement MUST be an integer if axis name is specified
            if not isinstance(displacement, int):
                raise TypeError(
                    "Displacement must be an integer when axis is specified"
                )
            # Axis name MUST be x, y, or z
            if axis not in ("x", "y", "z"):
                raise ValueError("axis must be one of x, y, or z")
            move = (
                displacement if axis == "x" else 0,
                displacement if axis == "y" else 0,
                displacement if axis == "z" else 0,
            )
            displacement = move

        initial_move = np.array(displacement, dtype=np.integer)

        self._position = list(np.array(self._position) + np.array(initial_move))
        logging.debug(np.array(self._position) + np.array(initial_move))
        logging.debug("New position: %s", self._position)

    def move_abs(self, final, **kwargs):
        time.sleep(0.5)

        self._position = list(final)
        logging.debug("New position: %s", self._position)

    def zero_position(self):
        """Set the current position to zero"""
        self._position = [0, 0, 0]

    def close(self):
        pass

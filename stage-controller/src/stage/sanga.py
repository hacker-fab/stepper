import logging
import time
from collections.abc import Iterable
from types import GeneratorType
from typing import Optional, Tuple, Union

import numpy as np
from sangaboard import Sangaboard
from typing_extensions import Literal

from stage.base import BaseStage
from utilities import axes_to_array


def _displacement_to_array(
    displacement: int, axis: Literal["x", "y", "z"]
) -> np.ndarray:
    # Create the displacement array
    return np.array(
        [
            displacement if axis == "x" else 0,
            displacement if axis == "y" else 0,
            displacement if axis == "z" else 0,
        ]
    )


class SangaStage(BaseStage):
    """
    Sangaboard v0.2 and v0.3 powered Stage object

    Args:
        port (str): Serial port on which to open communication

    Attributes:
        board (:py:class:`openflexure_microscope.stage.sangaboard.Sangaboard`): Parent Sangaboard object.
        _backlash (list): 3-element (element-per-axis) list of backlash compensation in steps.
    """

    def __init__(self, port=None, **kwargs):
        """Class managing serial communications with the motors for an Openflexure stage"""
        BaseStage.__init__(self)

        self.port = port
        self.board = Sangaboard(port, **kwargs)

        # Initialise backlash storage, used by property setter/getter
        self._backlash = None
        self.settle_time = 0.2  # Default move settle time
        self._position_on_enter = None

    @property
    def state(self):
        """The general state dictionary of the board."""
        return {"position": self.position_map}

    @property
    def configuration(self):
        return {
            "port": self.port,
            "board": self.board.board,
            "firmware": self.board.firmware,
        }

    @property
    def n_axes(self):
        """The number of axes this stage has."""
        return 3

    @property
    def position(self) -> Tuple[int, int, int]:
        return self.board.position

    @property
    def backlash(self) -> np.ndarray:
        """The distance used for backlash compensation.
        Software backlash compensation is enabled by setting this property to a value
        other than `None`.  The value can either be an array-like object (list, tuple,
        or numpy array) with one element for each axis, or a single integer if all axes
        are the same.
        The property will always return an array with the same length as the number of
        axes.
        The backlash compensation algorithm is fairly basic - it ensures that we always
        approach a point from the same direction.  For each axis that's moving, the
        direction of motion is compared with ``backlash``.  If the direction is opposite,
        then the stage will overshoot by the amount in ``-backlash[i]`` and then move
        back by ``backlash[i]``.  This is computed per-axis, so if some axes are moving
        in the same direction as ``backlash``, they won't do two moves.
        """
        if isinstance(self._backlash, np.ndarray):
            return self._backlash
        elif isinstance(self._backlash, list):
            return np.array(self._backlash)
        elif isinstance(self._backlash, int):
            return np.array([self._backlash] * self.n_axes)
        else:
            return np.array([0] * self.n_axes)

    @backlash.setter
    def backlash(self, blsh):
        logging.debug("Setting backlash to %s", (blsh))
        if blsh is None:
            self._backlash = None
        elif isinstance(blsh, Iterable):
            assert len(blsh) == self.n_axes
            self._backlash = np.array(blsh)
        else:
            self._backlash = np.array([int(blsh)] * self.n_axes, dtype=np.int)

    def update_settings(self, config: dict):
        """Update settings from a config dictionary"""

        # Set backlash. Expects a dictionary with axis labels
        if "backlash" in config:
            # Construct backlash array
            backlash = axes_to_array(config["backlash"], ["x", "y", "z"], [0, 0, 0])
            self.backlash = np.array(backlash)
        if "settle_time" in config:
            self.settle_time = config.get("settle_time")

    def read_settings(self) -> dict:
        """Return the current settings as a dictionary"""
        if self.backlash is not None:
            blsh = self.backlash.tolist()
        else:
            blsh = None
        config = {
            "backlash": {"x": blsh[0], "y": blsh[1], "z": blsh[2]},
            "settle_time": self.settle_time,
        }

        return config

    def move_rel(
        self,
        displacement: Union[int, Tuple[int, int, int], np.ndarray],
        axis: Optional[Literal["x", "y", "z"]] = None,
        backlash: bool = True,
    ):
        """Make a relative move, optionally correcting for backlash.
        displacement: integer or array/list of 3 integers
        axis: None (for 3-axis moves) or one of 'x','y','z'
        backlash: (default: True) whether to correct for backlash.

        Backlash Correction:
        This backlash correction strategy ensures we're always approaching the
        end point from the same direction, while minimising the amount of extra
        motion.  It's a good option if you're scanning in a line, for example,
        as it will kick in when moving to the start of the line, but not for each
        point on the line.
        For each axis where we're moving in the *opposite*
        direction to self.backlash, we deliberately overshoot:

        """
        if True:
            logging.debug("Moving sangaboard by %s", displacement)
            # If we specify an axis name and a displacement int, convert to a displacement tuple
            if axis:
                # Displacement MUST be an integer if axis name is specified
                if not isinstance(displacement, int):
                    raise TypeError(
                        "Displacement must be an integer when axis is specified"
                    )
                # Axis name MUST be x, y, or z
                if axis not in ("x", "y", "z"):
                    raise ValueError("axis must be one of x, y, or z")
                # Calculate displacement array
                displacement_array: np.ndarray = _displacement_to_array(
                    displacement, axis
                )
            elif isinstance(displacement, np.ndarray):
                displacement_array = displacement
            elif isinstance(displacement, (list, tuple, GeneratorType)):
                # Convert our displacement tuple/generator into a numpy array
                displacement_array = np.array(list(displacement))
            else:
                raise TypeError(f"Unsupported displacement type {type(displacement)}")

            # Handle simple case, no backlash
            if not backlash or self.backlash is None:
                return self.board.move_rel(displacement_array)

            # Handle move with backlash correction
            # Calculate main movement
            initial_move: np.ndarray = np.copy(displacement_array)
            initial_move -= np.where(
                self.backlash * displacement_array < 0,
                self.backlash,
                np.zeros(self.n_axes, dtype=self.backlash.dtype),
            )
            # Make the main movement
            self.board.move_rel(initial_move)
            # Handle backlash if required
            if np.any(displacement_array - initial_move != 0):
                # If backlash correction has kicked in and made us overshoot, move
                # to the correct end position (i.e. the move we were asked to make)
                self.board.move_rel(displacement_array - initial_move)
        # Settle outside of the stage lock so that another move request
        # can just take over before settling
        time.sleep(self.settle_time)

    def move_abs(self, final: Union[Tuple[int, int, int], np.ndarray], **kwargs):
        """Make an absolute move to a position
        """
        if True:
            logging.debug("Moving sangaboard to %s", final)
            self.board.move_abs(final, **kwargs)
        # Settle outside of the stage lock so that another move request
        # can just take over before settling
        time.sleep(self.settle_time)

    def zero_position(self):
        """Set the current position to zero"""
        if True:
            self.board.zero_position()

    def close(self):
        """Cleanly close communication with the stage"""
        if hasattr(self, "board"):
            self.board.close()

    # Methods specific to Sangaboard
    def release_motors(self):
        """De-energise the stepper motor coils"""
        self.board.release_motors()

    def __enter__(self):
        """When we use this in a with statement, remember where we started."""
        self._position_on_enter = self.position
        return self

    def __exit__(self, type_, value, traceback):
        """The end of the with statement.  Reset position if it went wrong.
        NB the instrument is closed when the object is deleted, so we don't
        need to worry about that here.
        """
        if type_ is not None:
            print(
                "An exception occurred inside a with block, resetting position \
                to its value at the start of the with block"
            )
            try:
                time.sleep(0.5)
                self.move_abs(self._position_on_enter)
            except Exception as e:  # pylint: disable=W0703
                print(
                    "A further exception occurred when resetting position: {}".format(e)
                )
            print("Move completed, raising exception...")
            raise value  # Propagate the exception


class SangaDeltaStage(SangaStage):
    def __init__(
        self,
        port: Optional[str] = None,
        flex_h: int = 80,
        flex_a: int = 50,
        flex_b: int = 50,
        camera_angle: float = 0,
        **kwargs,
    ):
        self.flex_h: int = flex_h
        self.flex_a: int = flex_a
        self.flex_b: int = flex_b

        # Set up camera rotation relative to stage
        camera_theta: float = (camera_angle / 180) * np.pi
        self.R_camera: np.ndarray = np.array(
            [
                [np.cos(camera_theta), -np.sin(camera_theta), 0],
                [np.sin(camera_theta), np.cos(camera_theta), 0],
                [0, 0, 1],
            ]
        )

        logging.debug(self.R_camera)

        # Transformation matrix converting delta into cartesian
        x_fac: float = -1 * np.multiply(
            np.divide(2, np.sqrt(3)), np.divide(self.flex_b, self.flex_h)
        )
        y_fac: float = -1 * np.divide(self.flex_b, self.flex_h)
        z_fac: float = np.multiply(np.divide(1, 3), np.divide(self.flex_b, self.flex_a))

        self.Tvd: np.ndarray = np.array(
            [
                [-x_fac, x_fac, 0],
                [0.5 * y_fac, 0.5 * y_fac, -y_fac],
                [z_fac, z_fac, z_fac],
            ]
        )
        logging.debug(self.Tvd)

        self.Tdv: np.ndarray = np.linalg.inv(self.Tvd)
        logging.debug(self.Tdv)

        SangaStage.__init__(self, port=port, **kwargs)

    @property
    def raw_position(self) -> Tuple[int, int, int]:
        return self.board.position

    @property
    def position(self):
        # TODO: Account for camera rotation
        position: np.ndarray = np.dot(self.Tvd, self.raw_position)

        position: np.ndarray = np.dot(np.linalg.inv(self.R_camera), position)
        return [int(p) for p in position]

    def move_rel(
        self,
        displacement: Union[int, Tuple[int, int, int], np.ndarray],
        axis: Optional[Literal["x", "y", "z"]] = None,
        backlash: bool = True,
    ):
        # If we specify an axis name and a displacement int, convert to a displacement tuple
        if axis:
            # Displacement MUST be an integer if axis name is specified
            if not isinstance(displacement, int):
                raise TypeError(
                    "Displacement must be an integer when axis is specified"
                )
            # Axis name MUST be x, y, or z
            if axis not in ("x", "y", "z"):
                raise ValueError("axis must be one of x, y, or z")
            # Calculate displacement array
            cartesian_displacement_array: np.ndarray = _displacement_to_array(
                displacement, axis
            )
        elif isinstance(displacement, np.ndarray):
            cartesian_displacement_array = displacement
        elif isinstance(displacement, (list, tuple, GeneratorType)):
            # Convert our displacement tuple/generator into a numpy array
            cartesian_displacement_array = np.array(list(displacement))
        else:
            raise TypeError(f"Unsupported displacement type {type(displacement)}")

        # Transform into camera coordinates
        camera_displacement_array: np.ndarray = np.dot(
            self.R_camera, cartesian_displacement_array
        )

        # Transform into delta coordinates
        delta_displacement_array: np.ndarray = np.dot(
            self.Tdv, camera_displacement_array
        )

        logging.debug("Delta displacement: %s", (delta_displacement_array))

        # Do the move
        SangaStage.move_rel(
            self, delta_displacement_array, axis=None, backlash=backlash
        )

    def move_rel_delta(
        self,
        displacement: Union[int, Tuple[int, int, int], np.ndarray]
    ):
        # Do the move
        SangaStage.move_rel(
            self, displacement, axis=None, backlash=None
        )
        
    def move_abs(self, final: Union[Tuple[int, int, int], np.ndarray], **kwargs):
        # Transform into camera coordinates
        camera_final_array: np.ndarray = np.dot(self.R_camera, final)

        # Transform into delta coordinates
        delta_final_array: np.ndarray = np.dot(self.Tdv, camera_final_array)

        logging.debug("Delta final: %s", (final))

        # Do the move
        SangaStage.move_abs(self, delta_final_array, **kwargs)

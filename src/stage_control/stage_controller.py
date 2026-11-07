# Hacker Fab
# J. Kent Wirant
# Stage Controller Interface


from core.engine_module import EngineModule
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stage_control.calibration import StageCalibration


class UnsupportedCommand(Exception):
    pass


# This will be an abstract interface for stage positioning.
class StageController(EngineModule):

    def __init__(self):
        super().__init__()
        self._calibration: Optional["StageCalibration"] = None
        self.__true_target_coord: list[float] = [0.0, 0.0, 0.0]

    def set_calibration(self, calibration: Optional["StageCalibration"]) -> None:
        """Sets or clears the stage coordinate calibration."""
        self._calibration = calibration

    def get_calibration(self) -> Optional["StageCalibration"]:
        """Returns the currently active stage coordinate calibration, if any."""
        return self._calibration

    @abstractmethod
    def has_homing(self) -> bool:
        pass

    def home(self) -> bool:
        self.__true_target_coord = [0.0, 0.0, 0.0]

    def move_relative(self, microns: dict[str, float], z_calib_only: bool = False) -> bool:
        dx = microns.get("x", microns.get("X", 0.0))
        dy = microns.get("y", microns.get("Y", 0.0))
        dz = microns.get("z", microns.get("Z", 0.0))
        self.__true_target_coord[0] += dx
        self.__true_target_coord[1] += dy
        self.__true_target_coord[2] += dz

        target = {
            "x": self.__true_target_coord[0],
            "y": self.__true_target_coord[1],
            "z": self.__true_target_coord[2],
        }
        # pass over the feed number
        for k in ("F", "f"):
            if k in microns:
                target[k] = microns[k]
        return self.move_absolute(target, z_calib_only=z_calib_only)

    def move_absolute(self, microns: dict[str, float], z_calib_only: bool = False) -> bool:
        dx = microns.get("x", microns.get("X", None))
        dy = microns.get("y", microns.get("Y", None))
        dz = microns.get("z", microns.get("Z", None))

        if dx is not None:
            self.__true_target_coord[0] = dx
        if dy is not None:
            self.__true_target_coord[1] = dy
        if dz is not None:
            self.__true_target_coord[2] = dz

        if self._calibration is not None and self._calibration.enabled:
            raw_target = self._calibration.corr_to_raw(self.__true_target_coord, z_calib_only=z_calib_only)
            for k in ("F", "f"):
                if k in microns:
                    raw_target[k] = microns[k]
            return self._move_absolute_raw(raw_target)
        raw_cmd = {
            'x': self.__true_target_coord[0],
            'y': self.__true_target_coord[1],
            'z': self.__true_target_coord[2],
        }
        for k in ("F", "f"):
            if k in microns:
                raw_cmd[k] = microns[k]
        return self._move_absolute_raw(raw_cmd)

    def get_position(self) -> tuple[float, float, float]:
        return (self.__true_target_coord[0], self.__true_target_coord[1], self.__true_target_coord[2])

    def move_absolute_raw(self, microns: dict[str, float], set_internal_coord: bool = False) -> bool:
        """Directly commands raw machine coordinates, bypassing calibration."""
        if set_internal_coord:
            dx = microns.get("x", microns.get("X", None))
            dy = microns.get("y", microns.get("Y", None))
            dz = microns.get("z", microns.get("Z", None))

            if dx is not None:
                self.__true_target_coord[0] = dx
            if dy is not None:
                self.__true_target_coord[1] = dy
            if dz is not None:
                self.__true_target_coord[2] = dz

        return self._move_absolute_raw(microns)

    def move_relative_raw(self, microns: dict[str, float], set_internal_coord: bool = False) -> bool:
        """Directly commands raw machine relative move, bypassing calibration."""
        if set_internal_coord:
            dx = microns.get("x", microns.get("X", 0.0))
            dy = microns.get("y", microns.get("Y", 0.0))
            dz = microns.get("z", microns.get("Z", 0.0))
            self.__true_target_coord[0] += dx
            self.__true_target_coord[1] += dy
            self.__true_target_coord[2] += dz

        return self._move_relative_raw(microns)

    def get_raw_position(self) -> tuple[float, float, float]:
        """Returns the raw machine position without calibration correction."""
        return self._get_position_raw()

    @abstractmethod
    def _move_relative_raw(self, microns: dict[str, float]) -> bool:
        pass

    @abstractmethod
    def _move_absolute_raw(self, microns: dict[str, float]) -> bool:
        pass

    @abstractmethod
    def get_bounds(self) -> Optional[dict[str, tuple[float, float]]]:
        pass





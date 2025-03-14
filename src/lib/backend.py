# Hacker Fab
# Luca Garlati, 2024
# backend library for gui_lib as well as other utilities

# region: imports
from __future__ import annotations

from typing import Callable, Literal

from PIL import Image

from .img import slice_image
from .tuple import *

# endregion


# TODO add calibration function
# TODO add option to prevent leaving FoV
# TODO add move to function
# class to manage global stage coordinates.
# can specify list of functions to call when updating coords
# can add debug widget to print info at various verbosity levels:
# <=0: no info
#   1: basic info
#   2: basic info + function calls
class Stage_Controller:
    dir_t = Literal["+", "-"]
    dirs_t = tuple[dir, dir]
    update_funcs: dict[Literal["x", "y", "z", "any"], dict[str, Callable]]
    debug: Debug | None
    step_size: tuple[float, float, float]
    __coords__: tuple[float, float, float]
    __verbosity__: int
    __locked__: bool = False
    # function that returns a location as percentage of fov from top left
    # recommended to use the gui_lib.GUI_Controller.get_coords() function
    __location_query__: Callable[[], tuple[float, float]] | None = None

    def __init__(
        self,
        starting_coords: tuple[float, float, float] = (0, 0, 0),
        step_size: tuple[float, float, float] = (1, 1, 1),
        debug: Debug | None = None,
        location_query: Callable[[], tuple[float, float]] | None = None,
        verbosity: int = 1,
    ):
        self.update_funcs = {"x": {}, "y": {}, "z": {}, "any": {}}
        self.__coords__ = starting_coords
        self.__changed_coords__ = starting_coords
        self.step_size = step_size
        self.debug = debug
        self.__location_query__ = location_query
        self.__verbosity__ = verbosity

    def __str2key__(self, axis: str) -> Literal["x", "y", "z", "any"] | None:
        match axis[-1]:
            case "x":
                return "x"
            case "y":
                if axis == "any":
                    return "any"
                else:
                    return "y"
            case "z":
                return "z"

    def __call_funcs__(self, axis: str):
        # convert arbitrary string to literal
        key: Literal["x", "y", "z", "any"] | None = self.__str2key__(axis)
        if key == None:
            return
        # call all functions
        for func in self.update_funcs.get(key, {}):
            self.update_funcs.get(key, {}).get(func, lambda: None)()

    # region: Convenience Getters
    def x(self) -> float:
        return self.__coords__[0]

    def y(self) -> float:
        return self.__coords__[1]

    def z(self) -> float:
        return self.__coords__[2]

    def xy(self) -> tuple[float, float]:
        return (self.__coords__[0], self.__coords__[1])

    def xz(self) -> tuple[float, float]:
        return (self.__coords__[0], self.__coords__[2])

    def yz(self) -> tuple[float, float]:
        return (self.__coords__[1], self.__coords__[2])

    def xyz(self) -> tuple[float, float, float]:
        return self.__coords__

    # endregion

    # wrapper to add function to update_funcs
    def add_callback(
        self, axis: Literal["x", "y", "z", "any"], name: str, func: Callable
    ):
        self.update_funcs[axis][name] = func

    # calibration function to equate camera view to stage step increments
    # region: params
    # @param step_size: amount to move stage by for calibration
    # @param calibrate_backlash:
    #          'None': no backlash calibration
    #          'symmetric': assume backlash is the same in both directions
    #          'bidirectional': calibrate backlash in both directions
    # @param return_to_start: return to starting position after calibration
    # endregion
    # region: fields
    # conversion ratio from camera fov percentage to stage steps
    __conversion_ratio__: tuple[float, float] | None = None
    # backlash calibration [[X back, X forward], [Y back, Y forward]] in stage steps
    __backlash__: tuple[tuple[float, float], tuple[float, float]] | None = None
    # location of point used to calibrate
    __calibrate_point__: tuple[float, float] | None = None
    # last directions moved in for backlash compensation
    __last_dir__: dirs_t = ("+", "+")

    # endregion
    def calibrate(
        self,
        step_size: tuple[float | int, float | int] | float | int,
        calibrate_backlash: Literal["None", "symmetric", "bidirectional"] = "None",
        return_to_start: bool = True,
    ):
        # region: general structure is as follows:
        # 1. Eliminate backlash biasing by moving
        #    get starting location from camera
        #    see which was is towards center of view as we don't want it moving out of view
        #    move stage in that direction by step_size, record that direction
        # 2. get conversion ratio
        #    get new location from camera as starting point of calibration
        #    move stage by step_size in same direction as before
        #    get new location from camera
        #    do math to get conversion ratio
        # 3. get backlash
        #    if backlash is enabled, move backwards
        #    get new location from camera
        #    do the math using the previous move amount as reference
        # 4. get backlash in other direction
        #    if bidirectional backlash is enabled, move forward
        #    get new location from camera
        #    do the math using the previous move amount as reference
        #    if not, just assign the previous backlash value to the other direction
        # 5. return to starting position
        # 6. ask user to click on where stage finally is
        #
        # The required number of clicks is as follows:
        # total = 3 + 1 if calibrate_backlash + 1 if bidirectional_backlash + 1 if return_to_start
        # endregion

        def convert_to_dir(dir: tuple[int, int, int]) -> self.dirs_t:
            return ("+" if dir[0] > 0 else "-", "+" if dir[1] > 0 else "-")

        def step(dir: tuple[int, int]) -> None:
            for i in range(2):
                match dir[i]:
                    case 1:
                        self.step(axis=("x" if i == 0 else "y"), size=step_size[i])
                    case -1:
                        self.step(axis=("-x" if i == 0 else "-y"), size=step_size[i])

        def bad_location(loc: tuple[float, float]) -> bool:
            if loc[0] < 0 or loc[0] > 1 or loc[1] < 0 or loc[1] > 1:
                if self.debug != None:
                    self.debug.error(
                        "Clicked out of bounds: "
                        + str(loc)
                        + "\n  Aborting calibration"
                    )
                    return True
            return False

        # 0. check inputs
        if type(step_size) != tuple:
            step_size = (step_size, step_size)
        if self.debug != None:
            self.debug.info("Calibrating: click on the same point as it moves")
        # 1. Eliminate backlash biasing by moving
        step_size: tuple[float, float] = abs_tuple(step_size)
        # get starting location from camera
        starting_location: tuple[float, float] = self.__location_query__()
        if bad_location(starting_location):
            return
        # see which way is towards center of view, remember to flip y
        main_dir: tuple[int, int, int] = (
            -1 if starting_location[0] >= 0.5 else 1,
            1 if starting_location[1] >= 0.5 else -1,
            0,
        )
        inv_dir: tuple[int, int, int] = neg_tuple(main_dir)
        # move stage in that direction by step_size
        step(main_dir)

        # 2. get conversion ratio
        # get new location from camera as starting point of calibration
        starting_location = self.__location_query__()
        if bad_location(starting_location):
            return
        # move stage by step_size in same direction as before
        step(main_dir)
        self.__last_dir__ = convert_to_dir(main_dir)
        # get new location from camera
        new_location: tuple[float, float] = self.__location_query__()
        if bad_location(new_location):
            return
        self.__calibrate_point__ = new_location
        # do math to get conversion ratio
        distance: tuple[float, float] = sub(new_location, starting_location)
        if distance[0] == 0 or distance[1] == 0:
            if self.debug != None:
                self.debug.error(
                    "Clicked in same location twice\n  Aborting calibration"
                )
            return
        self.__conversion_ratio__ = abs_tuple(div(step_size, distance))

        # 3. get backlash
        if calibrate_backlash != "None":
            starting_location = new_location
            # if backlash is enabled, move backwards
            step(inv_dir)
            self.__last_dir__ = convert_to_dir(inv_dir)
            # get new location from camera
            new_location = self.__location_query__()
            if bad_location(new_location):
                return
            self.__calibrate_point__ = new_location
            # do the math using the previous move amount as reference
            # backlash = | (-distance - (new_location - starting_location)) * conversion_ratio |
            backlash: tuple[float, float] = abs_tuple(
                mult(
                    sub(neg_tuple(distance), sub(new_location, starting_location)),
                    self.__conversion_ratio__,
                )
            )
            self.__backlash__ = ((backlash[0], backlash[0]), (backlash[1], backlash[1]))

        # 4. get backlash in other direction
        if calibrate_backlash == "bidirectional":
            starting_location = new_location
            backlash_list: list[list[float]] = [
                [backlash[0], backlash[0]],
                [backlash[1], backlash[1]],
            ]
            # if bidirectional backlash is enabled, move forward
            step(main_dir)
            self.__last_dir__ = convert_to_dir(main_dir)
            # get new location from camera
            new_location = self.__location_query__()
            if bad_location(new_location):
                return
            self.__calibrate_point__ = new_location
            # do the math using the previous move amount as reference
            backlash: tuple[float, float] = abs_tuple(
                mult(
                    sub(distance, sub(new_location, starting_location)),
                    self.__conversion_ratio__,
                )
            )
            # apply the backlash to the other direction only
            for i in range(2):
                match main_dir[i]:
                    case 1:
                        backlash_list[i][1] = backlash[i]
                    case -1:
                        backlash_list[i][0] = backlash[i]
            self.__backlash__ = (
                (backlash_list[0][0], backlash_list[0][1]),
                (backlash_list[1][0], backlash_list[1][1]),
            )

        # 5. return to starting position
        if return_to_start:
            if calibrate_backlash != "symmetric":
                step(inv_dir)
            step(inv_dir)
            self.__last_dir__ = convert_to_dir(inv_dir)
            self.__calibrate_point__ = new_location

        if self.debug != None:
            self.debug.info("Calibration complete")

    # check if stage is calibrated
    def is_calibrated(self) -> bool:
        return self.__conversion_ratio__ != None

    # move stage to a location in camera fov, must calibrate stage before use
    # @param location: move to location without querying user
    def goto(self, location: tuple[float, float] | None = None):
        # check if calibrated
        if self.__conversion_ratio__ == None:
            if self.debug != None:
                self.debug.error("Must calibrate before using goto function")
            return
        # get location from camera
        if location == None:
            location: tuple[float, float] = self.__location_query__()
        # get delta
        delta: list[float] = list(
            mult(self.__conversion_ratio__, sub(location, self.__calibrate_point__))
        )
        self.__calibrate_point__ = location
        # apply backlash
        if self.__backlash__ != None:
            new_dir: self.dirs_t = (
                "+" if delta[0] > 0 else "-",
                "+" if delta[1] > 0 else "-",
            )
            for i in range(2):
                if self.__last_dir__[i] != new_dir[i]:
                    if new_dir[i] == "+":
                        delta[i] -= self.__backlash__[i][1]
                    else:
                        delta[i] += self.__backlash__[i][0]
        # move stage
        self.incr(*(*delta, 0))

    # lock stage to prevent movement
    def lock(self):
        self.__locked__ = True

    # unlock stage
    def unlock(self):
        self.__locked__ = False

    # is stage locked
    def is_locked(self) -> bool:
        return self.__locked__

    # return true if stage has changed since last call to this function
    # default behavior is to reset changed comparison on every call
    # set query_only to retain previous reference
    __changed_coords__: tuple[float, float, float]

    def changed(self, query_only: bool = False) -> bool:
        result: bool = self.__changed_coords__ != self.__coords__
        if not query_only:
            self.__changed_coords__ = self.__coords__
        return result

    # step stage in a direction by step_size, mostly for convenience
    def step(
        self,
        axis: Literal["-x", "x", "+x", "-y", "y", "+y", "-z", "z", "+z"],
        size: float = 0,
        update: bool = True,
    ):
        if self.__locked__:
            if self.debug != None:
                self.debug.warn("Tried to move stage while locked")
            return
        delta: tuple[float, float, float] = (0, 0, 0)
        if size == 0:
            match axis[-1]:
                case "x":
                    delta = (self.step_size[0], 0, 0)
                case "y":
                    delta = (0, self.step_size[1], 0)
                case "z":
                    delta = (0, 0, self.step_size[2])
        else:
            match axis[-1]:
                case "x":
                    delta = (size, 0, 0)
                case "y":
                    delta = (0, size, 0)
                case "z":
                    delta = (0, 0, size)
        if axis[0] == "-":
            delta = mult(delta, -1)
        self.__coords__ = add(self.__coords__, delta)
        if update:
            self.__call_funcs__(axis)
            self.__call_funcs__("any")
        # region: debug
        if self.debug != None and self.__verbosity__ > 0:
            debug_str: str = ""
            if self.__verbosity__ >= 1:
                debug_str += (
                    "stage stepped " + str(delta) + " to " + str(self.__coords__)
                )
            if self.__verbosity__ >= 2 and update:
                debug_str += " and called:"
                # convert arbitrary string to literal
                key: Literal["x", "y", "z", "any"] | None = self.__str2key__(axis)
                if key != None:
                    for func in self.update_funcs.get(key, {}):
                        debug_str += "\n  " + axis[-1] + ": " + func
                for func in self.update_funcs.get("any", {}):
                    debug_str += "\n  any: " + func
            self.debug.info(debug_str)
        # endregion: debug

    # set coords from a list of floats
    def set(self, x: float, y: float, z: float, update: bool = True):
        if self.__locked__:
            if self.debug != None:
                self.debug.warn("Tried to move stage while locked")
            return
        self.__coords__ = (x, y, z)
        if update:
            self.__call_funcs__("x")
            self.__call_funcs__("y")
            self.__call_funcs__("z")
            self.__call_funcs__("any")
        # region: debug
        if self.debug != None and self.__verbosity__ > 0:
            debug_str: str = ""
            if self.__verbosity__ >= 1:
                debug_str += "stage set to " + str((x, y, z))
            if self.__verbosity__ >= 2 and update:
                debug_str += " and called:"
                for func in self.update_funcs.get("x", {}):
                    debug_str += "\n  x: " + func
                for func in self.update_funcs.get("y", {}):
                    debug_str += "\n  y: " + func
                for func in self.update_funcs.get("z", {}):
                    debug_str += "\n  z: " + func
                for func in self.update_funcs.get("any", {}):
                    debug_str += "\n  any: " + func
            self.debug.info(debug_str)
        # endregion

    # call set() with a delta, for convenience
    def incr(self, x: float = 0, y: float = 0, z: float = 0, update: bool = True):
        return self.set(*add(self.__coords__, (x, y, z)), update=update)


# Controls a set of N stage controllers
# allows naming of each, and toggling between them
# also supports group moving of selected stages
class Multi_Stage:
    # region: fields
    __controllers__: dict[str, Stage_Controller]
    __names__: list[str]
    __selected__: list[str]
    step_size: tuple[float, float, float] = (1, 1, 1)
    verbosity: int
    # endregion

    def __init__(
        self,
        names: list[str],
        debug: Debug | None = None,
        verbosity: int = 1,
        step_size: tuple[float, float, float] = (1, 1, 1),
        location_query: Callable[[], tuple[float, float]] | None = None,
    ):
        self.__names__ = names
        self.verbosity = verbosity
        self.debug = debug
        self.__controllers__ = {}
        for name in names:
            self.__controllers__[name] = Stage_Controller(
                debug=debug,
                verbosity=verbosity,
                location_query=location_query,
                step_size=step_size,
            )
        self.__selected__ = []

    # add an update function to all stages
    # enable force to overwrite existing functions with same name
    def batch_add_update_func(
        self,
        axis: Literal["x", "y", "z", "any"],
        name: str,
        func: Callable,
        force: bool = False,
    ):
        # check if function already exists to prevent overwriting
        if not force:
            for stage_name in self.__names__:
                if name in self.__controllers__[stage_name].update_funcs[axis]:
                    if self.debug != None:
                        self.debug.warn(
                            "attempted to overwrite existing function: "
                            + stage_name
                            + "."
                            + axis
                            + "."
                            + name
                            + "\n  use force=True to overwrite"
                        )
                    return

    # remove an update function from all stages if it exists
    # enable strict to throw error if a stage doesn't have the function
    def batch_remove_update_func(
        self, axis: Literal["x", "y", "z", "any"], name: str, strict: bool = True
    ):
        # if strict, check that the function exists in all stages
        if strict:
            for stage_name in self.__names__:
                if name not in self.__controllers__[stage_name].update_funcs[axis]:
                    if self.debug != None:
                        self.debug.warn(
                            "Tried to remove non-existant function: "
                            + stage_name
                            + "."
                            + axis
                            + "."
                            + name
                            + "\n  use strict=False to ignore"
                        )
                    return
        # remove function from all stages
        for stage_name in self.__names__:
            if name in self.__controllers__[stage_name].update_funcs[axis]:
                self.__controllers__[stage_name].update_funcs[axis].pop(name)

    # get a stage by name or name by stage
    # str search is O(1), Stage_Controller search is O(n)
    def get(self, name: str | Stage_Controller) -> Stage_Controller | None:
        if type(name) == str:
            if name not in self.__names__:
                if self.debug != None:
                    self.debug.error("Tried to get non-existant stage " + name)
                return None
            return self.__controllers__[name]
        if type(name) == Stage_Controller:
            for key in self.__controllers__:
                if self.__controllers__[key] == name:
                    return key
            if self.debug != None:
                self.debug.error("Tried to get non-existant stage")
            return None

    # toggle a stage between on and off
    # optionally specify force_on or force_off to ensure a stage is on or off
    # specify one stage as string, multiple as list, or None to toggle all
    def toggle(
        self,
        names: None | str | list[str] = None,
        force: Literal["on", "off", "None"] = "None",
    ):
        # get list of names to toggle
        name_list: list[str]
        if type(names) == str:
            name_list = [names]
        elif type(names) == list:
            name_list = names
        else:
            name_list = self.__names__.copy()

        # toggle each name
        for name in name_list:
            # check existence
            if name not in self.__names__:
                if self.debug != None:
                    self.debug.error("Tried to select non-existant stage " + name)
                continue
            # actually toggle
            if force == "on" and name not in self.__selected__:
                self.__selected__.append(name)
            elif force == "off" and name in self.__selected__:
                self.__selected__.remove(name)
            elif name in self.__selected__:
                self.__selected__.remove(name)
            elif name not in self.__selected__:
                self.__selected__.append(name)
            else:
                if self.debug != None:
                    self.debug.error(
                        "Execution reached unexpected point in Multi_Stage.toggle()"
                    )
            # success
            if self.debug != None and self.verbosity > 0:
                self.debug.info(
                    ("Enabled " if name in self.__selected__ else "Disabled ")
                    + name
                    + " stage"
                )

    # rename a stage
    def rename(self, old_name: str, new_name: str):
        # check existence
        if old_name not in self.__names__:
            if self.debug != None:
                self.debug.error("Tried to rename non-existant stage " + old_name)
            return
        # check name not taken
        if new_name in self.__names__:
            if self.debug != None:
                self.debug.error(
                    "Tried to rename "
                    + old_name
                    + " to "
                    + new_name
                    + " but "
                    + new_name
                    + " already exists"
                )
            return
        # actually rename
        self.__controllers__[new_name] = self.__controllers__.pop(old_name)
        self.__names__[self.__names__.index(old_name)] = new_name
        if old_name in self.__selected__:
            self.__selected__[self.__selected__.index(old_name)] = new_name
        # success
        if self.debug != None and self.verbosity > 0:
            self.debug.info("renamed " + old_name + " to " + new_name)

    # set coords from a list of floats
    def set_coords(self, name: str, x: float, y: float, z: float, update: bool = True):
        for name in self.__selected__:
            self.__controllers__[name].set(x, y, z, update)

    # step selected stages in a direction
    # updates children with step_size
    def step(
        self,
        axis: Literal["-x", "x", "+x", "-y", "y", "+y", "-z", "z", "+z"],
        size: float = 0,
        update: bool = True,
    ):
        for name in self.__selected__:
            child = self.__controllers__[name]
            child.step_size = self.step_size
            child.step(axis, size, update)

    # lock all stages, optionally only lock selected
    def lock(self, only_selected: bool = False):
        if only_selected:
            for name in self.__selected__:
                self.__controllers__[name].lock()
        else:
            for name in self.__names__:
                self.__controllers__[name].lock()

    # unlock all stages, optionally only unlock selected
    def unlock(self, only_selected: bool = False):
        if only_selected:
            for name in self.__selected__:
                self.__controllers__[name].unlock()
        else:
            for name in self.__names__:
                self.__controllers__[name].unlock()

    # return selected stage names list copy
    def get_enabled(self) -> list[Stage_Controller]:
        return self.__selected__.copy()

    # return name list copy
    def get_names(self) -> list[str]:
        return self.__names__.copy()

    # calibrate stage via name, list of names
    def calibrate(
        self,
        name: str | list[str],
        step_size: tuple[float, float],
        calibrate_backlash: Literal["None", "symmetric", "bidirectional"] = "None",
        return_to_start: bool = True,
    ) -> None:
        if type(name) == str:
            if name not in self.__controllers__ and self.debug != None:
                self.debug.error("Tried to calibrate stage with invalid name")
            else:
                if self.debug != None:
                    self.debug.info("Calibrating " + name + "...")
                self.__controllers__[name].calibrate(
                    step_size=step_size,
                    calibrate_backlash=calibrate_backlash,
                    return_to_start=return_to_start,
                )
            return
        elif type(name) == list:
            for n in name:
                if n not in self.__controllers__ and self.debug != None:
                    self.debug.error("Tried to calibrate stage with invalid name")
                else:
                    if self.debug != None:
                        self.debug.info("Calibrating " + n + "...")
                    self.__controllers__[n].calibrate(
                        step_size=step_size,
                        calibrate_backlash=calibrate_backlash,
                        return_to_start=return_to_start,
                    )
            return
        else:
            if self.debug != None:
                self.debug.error("Tried to calibrate stage with invalid name")


# Class takes an image and slicing parameters and returns slices
class Slicer:
    # pattern types
    pattern_type = Literal["snake", "row major", "col major"]
    pattern_list: list[Literal["snake", "row major", "col major"]] = [
        "snake",
        "row major",
        "col major",
    ]
    # fields
    __full_image__: Image.Image | None = None
    __sliced_images__: tuple[Image.Image, ...] = ()
    __index__: int = 0
    __pattern__: pattern_type
    __horizontal_slices__: int = 0
    __vertical_slices__: int = 0
    __grid_size__: tuple[int, int] = (0, 0)
    debug: Debug | None

    def __init__(
        self,
        image: Image.Image | None = None,
        horizontal_tiles: int = 0,
        vertical_tiles: int = 0,
        tiling_pattern: pattern_type = "snake",
        debug: Debug | None = None,
    ):
        if horizontal_tiles >= 0:
            self.__horizontal_slices__ = horizontal_tiles
        if vertical_tiles >= 0:
            self.__vertical_slices__ = vertical_tiles
        self.__pattern__ = tiling_pattern
        if image is not None:
            self.__full_image__ = image.copy()
            (self.__grid_size__, self.__sliced_images__) = slice_image(
                self.__full_image__,
                self.__horizontal_slices__,
                self.__vertical_slices__,
            )
        self.debug = debug

    # convert internal index counter to specified pattern index
    def __convert_index__(self, index: int = -1) -> int:
        if index == -1:
            index = self.__index__
        match self.__pattern__:
            case "row major":
                return index
            case "col major":
                return (
                    self.__grid_size__[0] * (index % self.__grid_size__[1])
                    + index // self.__grid_size__[1]
                )
            case "snake":
                row: int = index // self.__grid_size__[0]
                if row % 2 == 0:
                    return index
                else:
                    return (
                        self.__grid_size__[0] * (row + 1)
                        - (index % self.__grid_size__[0])
                        - 1
                    )
        return 0

    # increment index, false if at end of list
    def next(self, increment: int = 1) -> bool:
        if increment < 1:
            return False
        elif self.__index__ + increment >= len(self.__sliced_images__):
            return False
        else:
            self.__index__ += increment
            return True

    # decrement index, false if at beginning of list
    def prev(self, decrement: int = 1) -> bool:
        if decrement < 1:
            return False
        elif self.__index__ - decrement < 0:
            return False
        else:
            self.__index__ -= decrement
            return True

    # returns current image
    def image(self) -> Image.Image:
        result: Image.Image = self.__sliced_images__[self.__convert_index__()]
        return result

    # returns next image, if possible, without incrementing index
    def peek(self) -> Image.Image | None:
        result = None
        if self.next():
            result = self.image()
            self.prev()
        return result

    # returns number of tiles
    def tile_count(self) -> int:
        return len(self.__sliced_images__)

    # rests tile index to 0
    def restart(self):
        self.__index__ = 0

    # update slicer parameters
    # will reset index, so calling with no args is equivalent to resetting
    def update(
        self,
        image: Image.Image | None = None,
        horizontal_tiles: int = 0,
        vertical_tiles: int = 0,
        tiling_pattern: pattern_type = "snake",
    ):
        reslice: bool = False
        self.__index__ = 0

        if image is not None:
            self.__full_image__ = image.copy()
            reslice = True

        if self.__horizontal_slices__ != horizontal_tiles and horizontal_tiles >= 0:
            self.__horizontal_slices__ = horizontal_tiles
            reslice = True

        if self.__vertical_slices__ != vertical_tiles and vertical_tiles >= 0:
            self.__vertical_slices__ = vertical_tiles
            reslice = True

        if self.__pattern__ != tiling_pattern:
            self.__pattern__ = tiling_pattern
            reslice = True

        if reslice and self.__full_image__ != None:
            (self.__grid_size__, self.__sliced_images__) = slice_image(
                self.__full_image__,
                self.__horizontal_slices__,
                self.__vertical_slices__,
            )


# class to elegantly manage multi-function calls
# can be bound to a button to allow for dynamic functionality of button presses
class Func_Manager:
    __funcs__: dict[str, tuple[Callable, bool]]
    __total_enabled__: int
    name: str
    debug: Debug | None

    def __init__(
        self,
        name: str = "Unnamed Func Manager",
        debug: Debug | None = None,
    ):
        self.name = name
        self.__funcs__ = {}
        self.debug = debug
        self.__total_enabled__ = 0

    def __call__(self):
        if len(self.__funcs__) == 0 or self.__total_enabled__ == 0:
            return
        if self.debug != None:
            self.debug.info(self.name + " Func Manager called:")

        for key in self.__funcs__.keys():
            entry: tuple[Callable, bool] = self.__funcs__[key]
            if entry[1]:
                self.__funcs__[key][0]()
                if self.debug != None:
                    self.debug.info("| " + key)

    def add(self, name: str, func: Callable, enabled: bool = True):
        if enabled:
            self.__total_enabled__ += 1
        self.__funcs__[name] = (func, enabled)
        if self.debug != None:
            self.debug.info("added " + name + " to " + self.name + " Func Manager")

    def remove(self, name: str):
        if self.__funcs__.get(name, [False, False])[1]:
            self.__total_enabled__ -= 1
        self.__funcs__.pop(name, None)
        if self.debug != None:
            self.debug.info("removed " + name + " from " + self.name + " Func Manager")

    def enable(self, name: str):
        if self.__funcs__.get(name, [True, True])[1]:
            return
        else:
            self.__total_enabled__ += 1
            self.__funcs__[name] = (self.__funcs__[name][0], True)
            if self.debug != None:
                self.debug.info(
                    "enabled " + name + " in " + self.name + " Func Manager"
                )

    def disable(self, name: str):
        if not self.__funcs__.get(name, [True, True])[1]:
            return
        else:
            self.__total_enabled__ -= 1
            self.__funcs__[name] = (self.__funcs__[name][0], False)
            if self.debug != None:
                self.debug.info(
                    "disabled " + name + " in " + self.name + " Func Manager"
                )

    def is_enabled(self, name: str) -> bool:
        return self.__funcs__[name][1]

    def is_func(self, name: str) -> bool:
        return name in self.__funcs__

    def disable_all(self):
        for key in self.__funcs__.keys():
            self.disable(key)
        self.__total_enabled__ = 0
        if self.debug != None:
            self.debug.info("disabled all functions in " + self.name + " Func Manager")

    def enable_all(self):
        self.__total_enabled__ = 0
        for key in self.__funcs__.keys():
            self.enable(key)
            self.__total_enabled__ += 1
        if self.debug != None:
            self.debug.info("enabled all functions in " + self.name + " Func Manager")

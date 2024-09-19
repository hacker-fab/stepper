# Hacker Fab
# J. Kent Wirant
# GRBL Stage Controller

from stage_control.stage_controller import StageController
from typing import List
import io
import time

def clamp(value, lo, hi):
    if value > hi:
        return hi
    elif value < lo:
        return lo
    else:
        return value

class GrblStage(StageController):
    # only x, y, and z axes are supported by this interface
    # may support alternative axes schemes in the future
    # controller_target must be an open file (may be serial port for example)
    def __init__(self, 
                 controller_target, 
                 scale_factor: float, # units per micron
                 bounds: tuple[tuple[float,float],tuple[float,float],tuple[float,float]] = ((-10,10),(-10,10),(-10,10)), 
                 ):
        self.controller_target = controller_target
        self.scale_factor = scale_factor

        try:
            # set origin based on initial position
            self.controller_target.write(b'G92 0 0 0\n') 
        except:
            print(f'Error: Could not open motor control file "{controller_target}"')

        self.bounds = bounds
        self.axes = ('x', 'y', 'z')
        self.position = [0.0, 0.0, 0.0]

        for b in bounds:
            assert b[0] <= b[1]

    def __del__(self):
        self.controller_target.write(b'G91\n') # set to relative mode for safety
    
    def _move_relative(self, microns: tuple[float, float, float]):
        x, y, z = tuple(m * self.scale_factor for m in microns)
        self.controller_target.write(b'G91\n') # set relative movement mode
        self.controller_target.write(bytes(f'G0 x{x} y{y} z{z}\n', encoding='utf-8')) # move by this amount
    
    def _move_absolute(self, microns: tuple[float, float, float]):
        x, y, z = tuple(m * self.scale_factor for m in microns)
        self.controller_target.write(b'G90\n') # set absolute movement mode
        self.controller_target.write(bytes(f'G0 x{x} y{y} z{z}\n', encoding='utf-8')) # move by this amount

    # pass in list of amounts to move by. Dictionary in "axis: amount" format
    def move_by(self, amounts: dict[str, float]):
        # first make sure axes are valid
        if self.__axes_valid__(list(amounts.keys())):
            x, y, z = self.__adjust_coordinates__(amounts, True)
            self._move_relative((x, y, z))
            # if that worked, update internal position
            self.position[0] += x
            self.position[1] += y
            self.position[2] += z
            print(f"moved by {x} {y} {z}")
        else:
            print('Error: tried to move on invalid axis')

    def move_to(self, amounts: dict[str, float]):
        # first make sure axes are valid
        if self.__axes_valid__(list(amounts.keys())):
            x, y, z = self.__adjust_coordinates__(amounts, False)
            self._move_absolute((x, y, z))
            # if that worked, update internal position
            self.position[0] = x
            self.position[1] = y
            self.position[2] = z

    def __axes_valid__(self, axes):
        for axis in axes:
            if axis not in self.axes or (axis != 'x' and axis != 'y' and axis != 'z'):
                return False
        return True
    
    def __adjust_coordinates__(self, amounts: dict[str, float], relative: bool):
        coords = [0.0, 0.0, 0.0]
        clamped_amt = [0.0, 0.0, 0.0]
        coords[0] = amounts.get('x')
        coords[1] = amounts.get('y')
        coords[2] = amounts.get('z')

        for i in range(0, len(coords)):
            bounds_lo, bounds_hi = self.bounds[i]
            if coords[i] == None:
                if relative:
                    coords[i] = 0
                else:
                    coords[i] = self.position[i]
            else:
                # if bounds exceeded, set target coordinate to the boundary
                if relative:
                    clamped_amt[i] = clamp(coords[i] + self.position[i], bounds_lo, bounds_hi) - self.position[i]
                else:
                    clamped_amt[i] = clamp(coords[i], bounds_lo, bounds_hi)
        print('a')
        print(self.position)
        print(coords)
        print(clamped_amt)
        return clamped_amt

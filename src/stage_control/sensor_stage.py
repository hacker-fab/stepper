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

class SensorStage(StageController):
    # only x, y, and z axes are supported by this interface
    # may support alternative axes schemes in the future
    # controller_target must be an open file (may be serial port for example)
    def __init__(self, controller_target):
        self.controller_target = controller_target
        
        self.um_to_steps = 3.2

        self.axes = ('x', 'y', 'z')
        self.position = [0.0, 0.0, 0.0]
       
    def _refresh_log(self):
        if self.controller_target.in_waiting > 0:
            b = self.controller_target.read(self.controller_target.in_waiting)
            while (s := b.find(b'$')) != -1:
                print(b[:s].decode('utf-8'), end='')
                b = b[s:]
                while b'\n' not in b:
                    while self.controller_target.in_waiting == 0:
                        pass
                    self.controller_target.read(self.controller_target.in_waiting)
                e = b.index(b'\n')
                position = b[:e+1].decode('ascii')
                x, y = position[1:].strip().split(',')
                x = int(x)
                y = int(y)
                print(f'Position: {x}, {y}')
                b = b[e+1:]
            print(b.decode('utf-8'), end='')
    
    def query_position(self):
        self.controller_target.write(b'q\n')
        self._refresh_log()

    def _move_relative(self, microns: tuple[float, float, float]):
        self._refresh_log()
        x, y, z = tuple(round(m * self.um_to_steps) for m in microns)
        if x != 0 or y != 0:
            self.controller_target.write(f's {x} {y}\n'.encode('ascii'))
        if z != 0:
            self.controller_target.write(f'z {z}\n'.encode('ascii'))
    
    def _move_absolute(self, microns: tuple[float, float, float]):
        self._refresh_log()
        # TODO: Z
        z = microns[2] * self.um_to_steps

        x, y = round(microns[0]), round(microns[1])

        self.controller_target.write(f'a {x} {y}\n'.encode('ascii'))

    # pass in list of amounts to move by. Dictionary in "axis: amount" format
    def move_by(self, amounts: dict[str, float], **kwargs):
        # first make sure axes are valid
        self._move_relative((amounts.get('x', 0), amounts.get('y', 0), amounts.get('z', 0)))
        '''
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
        '''

    def move_to(self, amounts: dict[str, float], **kwargs):
        self._move_absolute((amounts['x'], amounts['y'], amounts['z']))
        '''
        # first make sure axes are valid
        if self.__axes_valid__(list(amounts.keys())):
            x, y, z = self.__adjust_coordinates__(amounts, False)
            self._move_absolute((x, y, z))
            # if that worked, update internal position
            self.position[0] = x
            self.position[1] = y
            self.position[2] = z
        '''

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

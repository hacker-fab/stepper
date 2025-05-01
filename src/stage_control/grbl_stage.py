# Hacker Fab
# J. Kent Wirant
# GRBL Stage Controller

import time

from stage_control.stage_controller import StageController, UnsupportedCommand


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
    def __init__(self, controller_target, enable_homing):
        self.controller_target = controller_target
        self.enable_homing = enable_homing

        time.sleep(3.0) # allow time for grbl to boot
        print(self.controller_target.read_all())

        self.axes = ("x", "y", "z")

        self.resp_buffer = b""

        print(self._query_state())

    def _fill_resp_buffer(self):
        self.resp_buffer += self.controller_target.read_all()

    def _send_msg(self, msg: bytes):
        self.controller_target.write(msg)
        while b"\r\n" not in self.resp_buffer:
            self._fill_resp_buffer()
        resp, self.resp_buffer = self.resp_buffer.split(b"\r\n", maxsplit=1)
        if resp != b"ok":
            raise Exception(f"not ok!!! {resp}")

    def _query_state(self):
        self.controller_target.write(b"?")
        while b">\r\n" not in self.resp_buffer:
            self._fill_resp_buffer()
        resp, self.resp_buffer = self.resp_buffer.split(b">\r\n", maxsplit=1)
        resp = resp.decode("ascii")
        print(repr(resp))
        print(len(self.resp_buffer), self.controller_target.in_waiting)
        idle = False
        for part in resp.split("|"):
            if "Idle" in part:
                idle = True
            elif part.startswith("MPos:"):
                x, y, z = part.removeprefix("MPos:").split(",")
                position = (float(x), float(y), float(z))

        return idle, position

    def __del__(self):
        self._send_msg(b"G91\n")

    def has_homing(self):
        return self.enable_homing

    def home(self):
        if self.enable_homing:
            self._send_msg(b"$H\n")
        else:
            raise UnsupportedCommand()

    def _move(self, microns: dict[str, float], relative):
        if relative:
            self._send_msg(b"G91\n")
        else:
            self._send_msg(b"G90\n")

        msg = "G0"
        if "x" in microns:
            x_mm = microns["x"] / 1000.0
            msg += f" x{x_mm:.3f}"
        if "y" in microns:
            y_mm = microns["y"] / 1000.0
            msg += f" y{y_mm:.3f}"
        if "z" in microns:
            z_mm = microns["z"] / 1000.0
            msg += f" z{z_mm:.3f}"
        msg += "\n"

        self._send_msg(msg.encode("ascii"))

    def move_relative(self, microns: dict[str, float]):
        print("moving relative", microns)
        self._move(microns, relative=True)

    def move_absolute(self, microns: dict[str, float]):
        print("moving absolute", microns)
        self._move(microns, relative=False)

    def move_by(self, amounts):
        self.move_relative(amounts)

    def move_to(self, amounts):
        self.move_absolute(amounts)

    """
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
    """

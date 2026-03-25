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

# Define potential Exception errors
class GrblError(Exception):
    pass

class GrblAlarmError(GrblError):
    pass

class GrblSoftLimitError(GrblError):
    pass

class GrblSoftLimitAlarm(GrblError):
    pass



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

        print(self._query_state()) # queries for current position and state

        self.configuration = self._query_config() # queries for current configuration settings
        
    def unlock(self):
        self.controller_target.write(b"$X\n")
        
    def _fill_resp_buffer(self):
        self.resp_buffer += self.controller_target.read_all()

    def _send_msg(self, msg: bytes):

        self.controller_target.write(msg)
        while b"\r\n" not in self.resp_buffer:
            self._fill_resp_buffer()
        resp, self.resp_buffer = self.resp_buffer.split(b"\r\n", maxsplit=1)
        response = resp.decode("ascii", errors="replace").strip()
        
        if response == 'ok':
            return # happy path
        
        # GRBL errors -> no movement
        if response.contains("error:"):
            code = response.split(":", 1)[1]
            print(f"error: {code} -> {response}")
            if code == "15":
                raise GrblSoftLimitError("Soft limit reached. Move away from the boundary.")
            raise GrblError(f"GRBL error: {code}\n-> Please visit this site for debugging alarm codes: https://docs.lightburnsoftware.com/legacy/Troubleshooting/GRBLErrors")
        
        # GRBL alarms -> movement occured
        if response.startswith("ALARM:"):
            code = response.split(":", 1)[1]
            print(f"ALARM: {code} -> {response}")
            
            if code == "2":
                print("code 2")
                self.exit_exceptions()
                self.resp_buffer = b""
                time.sleep(0.1)
                # _, pos = self._query_state()
                # self.stage_setpoint = (
                #     pos[0] * 1000,
                #     pos[1] * 1000,
                #     pos[2] * 1000
                # )
                raise GrblSoftLimitAlarm("Soft limit alarm. Move away from the boundary.")
            else:
                raise GrblAlarmError(f"GRBL alarm: Code -> {code}\n-> Please visit this site for debugging alarm codes: https://docs.lightburnsoftware.com/legacy/Troubleshooting/GRBLErrors")
        print("GrblError")
        raise GrblError(f"Unexpected GRBL response: {response}\n-> Please visit this site for debugging alarm codes: https://docs.lightburnsoftware.com/legacy/Troubleshooting/GRBLErrors")

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

    def _query_config(self):
        """
        queries for entire grbl settings ->
        $0=10 (Step pulse time, microseconds) \n
        $1=25 (Step idle delay, msec) \n
        $2=0 (Step pulse invert, mask) \n
        $3=0 (Step direction invert, mask) \n
        $4=0 (Invert step enable pin, boolean)\n
        $5=0 (Invert limit pins, boolean)\n
        $6=0 (Invert probe pin, boolean)\n
        $10=1 (Status report options, mask) \n
        $11=0.010 (Junction deviation, mm) \n
        $12=0.002 (Arc tolerance, mm) \n
        $13=0 (Report inches, boolean) \n
        $20=0 (Soft limits enable, boolean) \n
        $21=0 (Hard limits enable, boolean) \n
        $22=0 (Homing cycle enable, boolean) \n
        $23=0 (Homing direction invert, mask) \n
        $24=25.000 (Homing locate feed rate, mm/min) \n
        $25=500.000 (Homing search seek rate, mm/min) \n
        $26=250 (Homing switch debounce delay, msec) \n
        $27=1.000 (Homing switch pull-off distance, mm) \n
        $30=1000 (Max spindle speed, RPM) \n
        $31=0 (Min spindle speed, RPM) \n
        $32=0 (Laser mode enable, boolean) \n
        $100=250.000 (X steps/mm) \n
        $101=250.000 (Y steps/mm) \n
        $102=250.000 (Z steps/mm) \n
        $110=500.000 (X max rate, mm/min) \n
        $111=500.000 (Y max rate, mm/min) \n
        $112=500.000 (Z max rate, mm/min) \n
        $120=10.000 (X acceleration, mm/sec^2) \n
        $121=10.000 (Y acceleration, mm/sec^2) \n
        $122=10.000 (Z acceleration, mm/sec^2) \n
        $130=200.000 (X max travel, mm) \n
        $131=200.000 (Y max travel, mm) \n
        $132=200.000 (Z max travel, mm) \n
        """
        self.controller_target.write(b"$$\n")
        lines = []

        # fetch entire configuration for grbl
        while True:
            while b"\r\n" not in self.resp_buffer:
                self._fill_resp_buffer()
            raw, self.resp_buffer = self.resp_buffer.split(b"\r\n", maxsplit=1)
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line == "ok":
                break
            if line.startswith("error:"):
                raise Exception(f"GRBL error: query config failed -> {line}") # double check this
            lines.append(line)
        
        settings = {}
        for line in lines:
            if line.startswith("$") and "=" in line:
                key, value = line.split("=", 1)
                key = int(key[1:])
                settings[int(key)] = float(value)
        return settings

    def __del__(self):
        self._send_msg(b"G91\n")

    def exit_exceptions(self):
        print("in exit_exceptions")
        self.controller_target.write(b"\x18")  # soft reset
        print("soft reset")
        time.sleep(0.1)
        self.controller_target.write(b"$X\n")  # unlock
        print("unlock")
        self.resp_buffer = b""
        
    def has_homing(self):
        return self.enable_homing

    def home(self):
        if self.enable_homing:
            self._send_msg(b"$H\n")
        else:
            raise UnsupportedCommand()

        # G10 P0 L20 X0 Y0 Z0
    
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

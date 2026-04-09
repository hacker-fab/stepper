# Hacker Fab
# J. Kent Wirant
# GRBL Stage Controller

from collections import defaultdict
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
    def __init__(self, controller_target, enable_homing, enable_tiling, autofocus_offset):
        self.controller_target = controller_target
        self.enable_homing = enable_homing # homes on start
        self.enable_tiling = enable_tiling # allow for tiling feature that requires homing
        self.autofocus_estimate = autofocus_offset # a value offset that estimates current focus z-coordinates
        self.valid_position = True
        self.configuration = None
        self.on_start_location = (0,0,0)

        # Doing checks to ensure quality of following features
        if self.enable_tiling and not self.enable_homing: 
            raise RuntimeError("Error: Tiling enabled, but homing is not. Homing is required. Please change config and restart.")
    
        if self.autofocus_estimate != 0 and not self.enable_homing:
            raise RuntimeError("Error: Autofocus set, but homing is not. Homing is required. Please change config and restart.")

        time.sleep(3.0) # allow time for grbl to boot
        print(self.controller_target.read_all())

        self.axes = ("x", "y", "z")
        self.resp_buffer = b""
        
        print(f"WPos startup (mm): {self._query_state()}") # queries for current position and state
        self._send_msg(b'$X\n') # exit out of any alarms

        self._query_config() # query for grbl config settings, which we will need to determine bounds and other features

    def _fill_resp_buffer(self):
        self.resp_buffer += self.controller_target.read_all()
    
    def _wait_for_idle(self, timeout=10):
        """
        To ensure we are accurantely moving the stage, we'll only send
        GRBL the next move command when it is ready to take in more commands
        This means we will return if we're no longer idle
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            while b"\r\n" not in self.resp_buffer:
                self._fill_resp_buffer()

            raw, self.resp_buffer = self.resp_buffer.split(b"\r\n", 1)
            line = raw.decode("ascii", errors="replace").strip()
            self.resp_buffer = b""

            if not line:
                continue

            if line.startswith("<") and "Idle" in line:
                return

            # ignore everything else (but don't lose it if you need it!)
            raise TimeoutError("Stage did not reach idle")

    def _handle_alarms(self, response):
        """
        GRBL Alarm codes:
            ALARM:1  - Hard limit triggered
            ALARM:2  - Soft limit triggered  
            ALARM:3  - Reset while in motion
            ALARM:4  - Probe fail (initial state open)
            ALARM:5  - Probe fail (contact not detected)
            ALARM:6  - Homing fail (reset during cycle)
            ALARM:7  - Homing fail (door opened during cycle)
            ALARM:8  - Homing fail (failed to clear limit switch)
            ALARM:9  - Homing fail (could not find limit switch)
        """

        code = None
        if ":" in response:
            try:
                code = int(response.split(":")[1].strip())
            except ValueError:
                pass  
        
        try:
            self.soft_reset() # soft reset and unlock first
        except (Exception, RuntimeError, TimeoutError, ValueError) as e:
            print(f"Error: {str(e)}")
            return


        # GRBL alarms -> Once in alarm-mode, Grbl will lock out and shut down everything until the user issues a reset
        # --- Soft limit (ALARM:2) ---
        if code == 2:
            self.resp_buffer = b""
            raise RuntimeError(
                "Soft limit reached — motion outside machine bounds. "
                "Stage has been reset. Move away from boundary before retrying."
            )

        # --- Hard limit (ALARM:1) ---
        elif code == 1:
            self.resp_buffer = b""
            raise RuntimeError(
                "Hard limit triggered — a physical limit switch was hit. "
                "Check stage position and mechanical state before continuing."
            )

        # --- Homing failures (ALARM:6-9) ---
        elif code in (6, 7, 8, 9):
            self.resp_buffer = b""
            raise RuntimeError(
                f"Homing failure ({response}) — homing cycle did not complete. "
                "Check limit switches and wiring."
            )

        # --- Reset while in motion (ALARM:3) ---
        elif code == 3:
            self.resp_buffer = b""
            self.valid_position = False
            raise RuntimeError(
                "GRBL was reset while the stage was moving. "
                "Position may be lost — rehome before continuing."
            )

        # --- Probe failures (ALARM:4-5) ---
        elif code in (4, 5):
            # probe alarms don't necessarily require a full reset
            # since your lithostepper may not use probing, raise clearly
            self.resp_buffer = b""
            raise RuntimeError(
                f"Probe failure ({response}) — check probe wiring and initial state."
            )

        # --- Any other that we didn't cover ---
        else:
            self.resp_buffer = b""
            raise RuntimeError(
                f"Unknown GRBL alarm: {response!r}. "
                "See https://docs.lightburnsoftware.com/legacy/Troubleshooting/GRBLErrors"
            )

    def _send_msg(self, msg: bytes):
        """
        sending g-code command
        Note: to learn more about error and alarm handling mechanisms, please
        visit https://github.com/grbl/grbl/wiki//Interfacing-with-Grbl#streaming-a-g-code-program-to-grbl
        to understand how to unlock out of locked stage, or how to interface with grbl via g-code

        Message Handling: \\
        Most of the feedback from Grbl fits into nice categories. Here's how they are organized:

            - ok: Standard all-is-good response to a single line sent to Grbl.
            - error: Standard error response to a single line sent to Grbl.
            - ALARM: A critical error message that occurred. All processes stopped until user acknowledgment.
            - []: All feedback messages are sent in brackets (parameter and g-code parser state print-outs)
            - <>: Status reports are sent in chevrons.
        """
        self.controller_target.write(msg) # write gcode command
        deadline = time.time() + 30.0
        while True:
            while b"\r\n" not in self.resp_buffer: # sometimes grbl may take time to respond, so we wait until
                self._fill_resp_buffer()             # the response actually arrives. This is really important.
                if time.time() > deadline:
                    raise TimeoutError("No response from GRBL")      
                time.sleep(0.01)  # yield the CPU

            resp = self.resp_buffer.split(b"\r\n")
            self.resp_buffer = b""
            response = [res.decode("ascii", errors="replace").strip() for res in resp]

            for item in response:
                if not item:
                    continue  # skip blank lines
                elif item.startswith("[") or item.startswith("<"):
                    print(f"[GRBL feedback]: {item}")
                    continue  # keep waiting for ok/error
                if "ok" in item:
                    print("Received OK")
                    return  # happy path, command completed, successful
                elif item.startswith("error:"):
                    print(f"[GRBL error]: {item}")
                    return # not going to block normal operations, command blocked, session stays active
                elif item.upper().startswith("ALARM:"):
                    self._handle_alarms(item)
                    return
                else:
                    print(f"[GRBL unknown]: {item}")
                    continue
        
    def _query_state(self):
        """
        Status Report: \
        
        - MPos:0.000,0.000,0.000: Machine position listed as X,Y,Z coordinates. 
            - Machine Position (MPos): This is the "Absolute" coordinate system
            - defined entirely by your homing cycle -> often will see negative 
            - values here because (0,0,0) is location where all proximity sensors light up
            - and so it "bounces" away from that a bit and that's your homing process
            - this "bounce" is defined by ($27)

        - WPos:0.000,0.000,0.000: Work position listed as X,Y,Z coordinates. 
            - Work Position = Machine Position - Work Offset

        - Buf:0: Number of motions queued in Grbl's planner buffer.
        - RX:0: Number of characters queued in Grbl's serial RX receive buffer.

        $10=0 => WPos only
        $10=1 => MPos only
        $10=2 => MPos + WCO (work coordinate offset)

        Example response: `<Idle,MPos:0.000,0.000,0.000,WPos:0.000,0.000,0.000>`
        """
        self.controller_target.write(b"?\n")

        while b">\r\n" not in self.resp_buffer:
            self._fill_resp_buffer()

        resp_raw = self.resp_buffer.split(b"<")[1].split(b">")[0]
        buff = resp_raw.decode("ascii", errors="replace").strip()
        self.resp_buffer = b""

        idle = False
        position = None
        work_position = None

        for part in buff.split("|"):
            if "Idle" in part:
                idle = True
            elif part.startswith("MPos:"):
                x, y, z = part.removeprefix("MPos:").split(",")
                position = (float(x), float(y), float(z))
            elif part.startswith("WPos:"):
                x, y, z = part.removeprefix("WPos:").split(",")
                work_position = (float(x), float(y), float(z))
        
        resolved_position = position or work_position
        print(f"resolved position: {resolved_position}, Idle: {idle}")
        if resolved_position is None:
            raise ValueError(f"GRBL status response contained no position data: {buff!r}")
        return idle, resolved_position
    
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
        print("Querying GRBL configurations")
        self.configuration = {}
        self.resp_buffer = b""
        self.controller_target.write(b"$$\n")
        lines = []

        # fetch entire configuration for grbl
        while b"ok" not in self.resp_buffer:
            self._fill_resp_buffer()
        raw = self.resp_buffer.split(b"\r\n")
        self.resp_buffer = b""
        buff = [r.decode("ascii", errors="replace").strip() for r in raw]
        
        # parse the buffer
        for setting in buff:
            if not setting or '$' not in setting:
                continue
            if "ok" in setting:
                break
            if setting.startswith("error:"):
                raise Exception(f"GRBL error: query config failed -> {setting}") # double check this
            lines.append(setting)
        
        # parse string splits
        for line in lines:
            if "$" in line:
                part = line.split("$")[-1].strip() # => [0=10]
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = int(key.strip())
                    value = value.strip()
                    if value == "":
                        value = None
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                    self.configuration[key] = value
        print("Done extracting configuration")
        print(self.configuration)

    def __del__(self):
        self._send_msg(b"G91\n")

    def _move(self, microns: dict[str, float], relative):

        # Note: depending on $10, G91 and G90 will move wrt WPos or MPos
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
        """
        Moves relative to WPos

        Pre-conditions if homing enabled:
        0 >= microns[0] >= -$130   (for X)
        0 >= microns[1] >= -$131   (for Y)
        0 >= microns[2] >= -$132   (for Z)
        """
        if not self.valid_position and self.enable_tiling: # only block if users are doing tiling
            raise RuntimeError("Position is invalid — please exit and re-open the application.")
        
        print("moving relative", microns)
        self._move(microns, relative=True)

    def move_absolute(self, microns: dict[str, float]):
        """
        Moves to absolute location in WPos

        Pre-conditions if homing enabled:
        0 >= microns[0] >= -$130   (for X)
        0 >= microns[1] >= -$131   (for Y)
        0 >= microns[2] >= -$132   (for Z)
        """
        if not self.valid_position and self.enable_tiling: # only block if users are doing tiling
            raise RuntimeError("Position is invalid — please exit and re-open the application.")
        
        print("moving absolute", microns)
        self._move(microns, relative=False)
    
    def soft_reset(self):
        """ 
        This is used when handling GrblAlarm that locks GRBL interface and disallows
        us from sending more g-code messages, thereby creating a freezing gui interface. 
        Steps to unlock include a full soft reset, and an unlock g-code command to escape out
        """
        self.controller_target.write(b"\x18")  # soft reset
        
        deadline = time.time() + 5.0
        startup_seen = False

        while time.time() < deadline:
            self.resp_buffer += self.controller_target.read_all()
            if b"Grbl" in self.resp_buffer:
                startup_seen = True
                break
            time.sleep(0.05)
        
        if not startup_seen:
            raise RuntimeError("GRBL did not send startup greeting after reset — check connection.")

        self.resp_buffer = b""

        deadline = time.time() + 3.0
        while time.time() < deadline:
            self.resp_buffer += self.controller_target.read_all()
            if b"\r\n" in self.resp_buffer:
                break
            time.sleep(0.05)

        resp, self.resp_buffer = self.resp_buffer.split(b"\r\n", maxsplit=1)
        response = resp.decode("ascii", errors="replace").strip()

        if "ok" not in response:
            raise RuntimeError(f"GRBL unlock ($X) failed — got: {response!r}")

        self.resp_buffer = b""
        print("Stage Reset and Unlocked")
    
    def home(self):
        """
        Homes the stage (assume proximity sensors exist for each axis, then sets WPos
        to the homed location that is $27 mm away from absolute MPos (0,0,0). Now
        the users will see the location post-homing as (0,0,0)

        NOTE: Maximum distance of homing seek travel per axis is 1.5 times its configured max travel,
        which basically guarantees that we can find home if soft limits are set. So ensure soft limits
        are enabled to home deterministically (works 99% of the time)
        - To change max travel: set $130-$132
        - To change soft limits: set $20=1
        
        When $H runs, GRBL moves all axes toward their endstops until the limit switches trigger. 
        Then it pulls off by $27 (pull-off distance, default 1mm). After that sequence completes, 
        GRBL automatically sets MPos to (0,0,0) at the endstop location internally.

        Because of pull-off dinstance, you may see MPos at a negative location upon sending a ?
        right after homing. 

        Because MPos can sometimes contain random values (which makes sense), we want to set a 
        WPos, which allows us to set the (0,0,0) position to any location we're at. This is just
        so developers and gui users find it easier to work off of coordinates with an origin of (0,0,0)

        For more info, please visit GRBL documentation wiki. 

        """
        print("Sending Home Command...")
        if self.enable_homing:
            self._send_msg(b"$H\n")
            print("Storing Location to Home...")

            # "pinning" WPos (0,0,0) to the physical end-stops (limit switch locations)
            # MPos = (some triplet of negative values depending on $27)
            # WPos = (0,0,0) -> establish current position as home (0,0,0)
            self._send_msg(b"G10 L20 P1 X0 Y0 Z0\n") 
            self.valid_position = True
                
        else:
            raise UnsupportedCommand()
    
    def set_on_start_location(self):
        self.on_start_location = self.get_position()
        
    def has_homing(self) -> bool:
        return self.enable_homing

    def get_autofocus(self) -> float:
        """
        Returns offset from z-axis home coordinate
        that can get the stage to reach an estimated
        focused position. 

        Note: developers will still need to run their custom 
        autofocus function that can fine-tune the focus score,
        but this provides a good point to start the gradient
        descent search. 
        """
        return -1*self.autofocus_estimate
        
    def get_position(self) -> tuple[float, float, float]:
        """
        Returns GUI / Wpos of stage in microns
        IMPORTANT: WPos is different from MPos, so
        ensure that $10 is set to reveal WPos instead of MPos
        """
        _, positions = self._query_state() # in microns
        micron_x = positions[0] * 1000
        micron_y = positions[1] * 1000
        micron_z = positions[2] * 1000
        return (micron_x, micron_y, micron_z)
    
    def get_on_start_location(self) -> tuple[float, float, float]:
        return self.on_start_location
    
    def get_bounds(self):
        cfg = self.configuration
        mask    = int(cfg[23])   # homing direction invert mask
        pulloff = cfg[27] * 1000 # mm → your units

        def axis_bounds(travel_param, bit):
            travel = cfg[travel_param] * 1000
            homes_to_max = bool(mask & (1 << bit))
            if homes_to_max:
                # origin at max end → coords are negative
                return (-travel + pulloff, 0.0)
            else:
                # origin at min end → coords are positive
                return (0.0, travel - pulloff)
        x = axis_bounds(130, 0)
        y = axis_bounds(131, 1)
        z = axis_bounds(132, 2)
        return {
            'x': [-x[0], -x[1]],
            'y': [-y[0], -y[1]],
            'z': [-z[0], -z[1]],
        }

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

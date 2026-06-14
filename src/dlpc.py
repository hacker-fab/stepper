from __future__ import annotations

import usb.core
import usb.util

TI_VID = 0x0451  # Texas Instruments

DEST_COMMON = 1
DEST_SYSTEM = 4

HDR_READ = 1 << 7
HDR_REPLY = 1 << 6
HDR_BUSY = 1 << 7
HDR_ERROR = 1 << 6
HDR_DEST_MASK = 0x07

OP_MODE = 0x00
OP_CONTROLLER_INFO = 0x00
OP_ILLUMINATION_ENABLE = 0x80   # Table 19-90: 1 data byte
OP_ILLUMINATION_CURRENT = 0x84  # Table 19-91: 6 data bytes, 2 per channel LE

ILLUM_OFF = 0x00
ILLUM_RED = 0b001
ILLUM_GREEN = 0b010
ILLUM_BLUE = 0b100
ILLUM_ALL = 0b111


class DLPC:
    """USB bulk-transfer driver for DLPC6540 (DLPDLCR471TPEVM). Used to interface with the
    projector's DLPC chip as per: https://www.ti.com/lit/ug/dlpu110b/dlpu110b.pdf"""

    INTERFACE = 0  # Projector Control per §15.3

    def __init__(self, dev: usb.core.Device, timeout_ms: int = 1000):
        self.dev = dev
        self.timeout_ms = timeout_ms
        self._reattach = False

        try:
            if dev.is_kernel_driver_active(self.INTERFACE):
                dev.detach_kernel_driver(self.INTERFACE)
                self._reattach = True
        except (NotImplementedError, usb.core.USBError):
            pass

        dev.set_configuration()
        cfg = dev.get_active_configuration()
        intf = cfg[(self.INTERFACE, 0)]

        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
                and usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK
            ),
        )
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
                and usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK
            ),
        )
        if self.ep_out is None or self.ep_in is None:
            raise IOError("Did not find bulk OUT and IN endpoints on interface 0")

    def close(self) -> None:
        usb.util.dispose_resources(self.dev)
        if self._reattach:
            try:
                self.dev.attach_kernel_driver(self.INTERFACE)
            except usb.core.USBError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send_write_command(
        self, destination: int, opcode: int, data: bytes = b"", *, reply: bool = False
    ) -> bytes | None:
        header = (HDR_REPLY if reply else 0) | (destination & HDR_DEST_MASK)
        self.ep_out.write(bytes([header, opcode]) + data, timeout=self.timeout_ms)

        if not reply:
            return None

        resp = bytes(self.ep_in.read(64, timeout=self.timeout_ms))
        if len(resp) < 1:
            raise IOError("Empty response from controller")
        resp_header = resp[0]
        if resp_header & HDR_ERROR:
            err_code = resp[1] if len(resp) > 1 else 0xFF
            raise RuntimeError(
                f"Controller error {err_code} (see Table 16-5). Header=0x{resp_header:02X}"
            )
        return resp[1:]

    def send_read_command(self, destination: int, opcode: int, response_len: int) -> bytes:
        header = HDR_READ | (destination & HDR_DEST_MASK)
        self.ep_out.write(bytes([header, opcode]), timeout=self.timeout_ms)

        wanted = response_len + 1
        resp = bytes(self.ep_in.read(max(wanted, 64), timeout=self.timeout_ms))

        if len(resp) < 1:
            raise IOError("Empty response from controller")
        resp_header = resp[0]
        if resp_header & HDR_ERROR:
            err_code = resp[1] if len(resp) > 1 else 0xFF
            raise RuntimeError(
                f"Controller error {err_code} (see Table 16-5). Header=0x{resp_header:02X}"
            )
        if resp_header & HDR_BUSY:
            raise RuntimeError(
                f"BUSY bit set on USB response (header 0x{resp_header:02X})"
            )
        return resp[1:wanted]

    def set_illumination_enable(self, mask: int) -> None:
        """Enable/disable LEDs. mask: ILLUM_RED | ILLUM_GREEN | ILLUM_BLUE."""
        if not 0 <= mask <= 7:
            raise ValueError(f"mask must be 0-7, got {mask}")
        self.send_write_command(DEST_SYSTEM, OP_ILLUMINATION_ENABLE, bytes([mask]))

    def get_illumination_enable(self) -> int:
        """Return the current illumination enable bitmask (0-7)."""
        data = self.send_read_command(DEST_SYSTEM, OP_ILLUMINATION_ENABLE, 1)
        return data[0]

    def set_led_drive_level(self, red: int, green: int, blue: int) -> None:
        """Set drive current level for each LED (0-874).
        OutputCurrent = ((level + 1) / 1024) * (0.15 / R_LIM) A.
        Do not use when Dynamic Black or WPC is enabled."""
        for name, val in (("red", red), ("green", green), ("blue", blue)):
            if not 0 <= val <= 874:
                raise ValueError(f"{name} must be 0-874, got {val}")
        payload = (
            red.to_bytes(2, "little")
            + green.to_bytes(2, "little")
            + blue.to_bytes(2, "little")
        )
        self.send_write_command(DEST_SYSTEM, OP_ILLUMINATION_CURRENT, payload)

    def get_led_drive_level(self) -> tuple[int, int, int]:
        """Return (red, green, blue) drive levels in range 0-874."""
        data = self.send_read_command(DEST_SYSTEM, OP_ILLUMINATION_CURRENT, 6)
        red = int.from_bytes(data[0:2], "little")
        green = int.from_bytes(data[2:4], "little")
        blue = int.from_bytes(data[4:6], "little")
        return red, green, blue


def find_devices(vid: int, pid: int | None = None) -> list[usb.core.Device]:
    """Return all USB devices matching vid (and pid if given)."""
    kw: dict = {"idVendor": vid}
    if pid is not None:
        kw["idProduct"] = pid
    return list(usb.core.find(find_all=True, **kw))


def connect(pid: int | None = None) -> DLPC | None:
    """Find the first TI DLPC device and return a connected DLPC, or None."""
    devices = find_devices(TI_VID, pid)
    if not devices:
        return None
    return DLPC(devices[0])

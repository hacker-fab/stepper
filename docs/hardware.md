# Hardware Setup & Peripherals Guide

This document describes the hardware components supported by the Hacker Fab Stepper V2 software, including motion stages, cameras, and the projector controller, along with their installation and configuration steps.

---

## 1. Motion Stages

The stepper software provides an extensible stage controller interface supporting physical CNC/micropositioning hardware and simulated stages. The stage is selected via `[stage]` in your `config.toml`.

### GRBL Stage (Arduino CNC)

GRBL controls 3-axis stepper stages via an Arduino over USB serial.

#### Stage Axes & Polarity
Stage axes are defined from the **point of view of the projector/camera**:
- **X and Y axes**: Pan the stage relative to the optical axis.
- **Z axis**: Controls focal distance. Movement in **+Z** moves the stage closer to the objective lens (focusing in).
- If any axis direction is inverted, swap the stepper motor coil wiring or invert directions in GRBL settings.

#### Firmware Configuration (`config.h`)
If physical limit switches or proximity sensors are installed, modify GRBL's `config.h` before flashing to the Arduino:

1. **Configure Homing Cycles**:
   Replace the default homing definitions to home X/Y together first, then Z:
   ```c
   #define HOMING_CYCLE_0 ((1 << X_AXIS)|(1<<Y_AXIS))
   #define HOMING_CYCLE_1 (1 << Z_AXIS)
   ```

2. **Enable Z-Axis Limit Switch**:
   Comment out `VARIABLE_SPINDLE` to free up the hardware pin for the Z limit switch:
   ```c
   //#define VARIABLE_SPINDLE
   ```

#### Runtime Configuration (EEPROM)
Connect to the Arduino using a serial terminal (e.g. Arduino IDE Serial Monitor or `picocom`) at **115200 baud** to configure GRBL settings:

| Parameter | Recommended Value | Description |
|---|---|---|
| `$22` | `1` | Enable homing cycle (requires limit sensors) |
| `$23` | `7` | Homing direction invert mask (7 inverts X, Y, and Z for negative travel limit switches) |
| `$24` | `10.0` | Homing feed rate (mm/min) |
| `$25` | `50.0` | Homing seek rate (mm/min) |
| `$27` | `0.5` | Homing pull-off distance (mm) |
| `$100`, `$101`, `$102` | `3200.0` | Steps per mm for X, Y, Z (based on 8x microstepping, 200 steps/rev, 0.5mm pitch) |
| `$110`, `$111`, `$112` | `120.0` | Max travel rate (mm/min) |
| `$120`, `$121`, `$122` | `5.0` | Max acceleration (mm/sec²) |
| `$130`, `$131`, `$132` | `15.0` | Max travel limit (mm) |

#### TOML Configuration
```toml
[stage]
type = "grbl"

[stage.grbl]
port = "COM6"         # Serial port (e.g., /dev/ttyUSB0 on Linux)
baud-rate = 115200    # Default GRBL baud rate
homing = false        # Set to true if homing sensors are installed
```

---

### Open Micro Manipulator (OMM) Stage

The OMM stage provides high-precision microstepping control through the `open-micro-stage-api`.

#### Installation
Install the required OMM API library:
```bash
pip install '.[omm]'
```

#### TOML Configuration
```toml
[stage]
type = "omm"

[stage.omm]
port = "COM6"         # Serial port
baud-rate = 921600    # High-speed baud rate
z-max = 50.0          # Max allowable Z travel limit (mm)
```

---

### Dummy Stage (Simulation Mode)

For software development or UI testing without physical hardware, the dummy stage simulates smooth motion and delay.

```toml
[stage]
type = "dummy"

[stage.dummy]
delay = 0.01          # Command execution delay in seconds
speed = 500.0         # Simulated travel speed
```

---

## 2. Cameras

The camera feed provides live alignment viewing, pattern calibration, and capture capabilities.

### Supported Camera Types

| Camera Type | Driver Key | Prerequisites |
|---|---|---|
| **Webcam** | `"webcam"` | `opencv-python` (default dependency) |
| **Basler (Pylon)** | `"pylon"` or `"basler"` | Basler Pylon Software Suite + `pypylon` (`pip install '.[basler]'`) |
| **FLIR** | `"flir"` | `flir-private` submodule (Python ≤ 3.10) |
| **AmScope** | `"amscope"` | AmScope SDK |
| **Dummy** | `"dummy"` | None (generates test pattern) |
| **None** | `"none"` | Disables camera preview |

---

### Basler Camera Setup (Pylon)

1. Download and install the **Pylon Software Suite** from [Basler's Website](https://www.baslerweb.com/en-us/downloads/software/).
2. Reboot your computer when prompted to allow driver initialization.
3. Install the Python bindings:
   ```bash
   pip install '.[basler]'
   # or: pip install pypylon
   ```
4. Configuration in `config.toml`:
   ```toml
   [camera]
   type = "pylon"

   [camera.pylon]
   index = 0
   ```

---

### FLIR Camera Setup

FLIR Machine Vision cameras require proprietary vendor libraries and Python ≤ 3.10:

1. Clone the private FLIR driver into `src/camera/flir`:
   ```bash
   cd src/camera
   git clone git@github.com:hacker-fab/flir-private.git flir
   ```
2. Ensure you are running Python 3.10:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Configuration in `config.toml`:
   ```toml
   [camera]
   type = "flir"
   ```

---

### Generic USB Webcam

Standard USB webcams and USB microscope cameras are driven via OpenCV:

```toml
[camera]
type = "webcam"

[camera.webcam]
index = 0             # 0 for primary webcam, 1+ for external USB cams
```

---

### Dummy Camera

Generates an animated test pattern with timestamps for development and CI testing without hardware attached:

```toml
[camera]
type = "dummy"

[camera.dummy]
width = 640
height = 480
```

---

## 3. Projector & DLPC6540 LED Controller

The lithography projector projects patterns onto the photoresist. To prevent unintended exposure or background leakage:
- During **alignment (Red Mode)**: The UV/blue LED channel is disabled.
- During **exposure (UV Mode)**: The Red LED channel is disabled.

The software can control the Texas Instruments **DLPC6540** projector controller directly over USB.

### Prerequisites
Install `pyusb`:
```bash
pip install '.[dlpc]'
```

> **Note for Linux Users**: You may need to grant USB permissions by adding a udev rule for the Texas Instruments VID `0x0451`.

### TOML Configuration
```toml
[projector]
# Enable direct USB hardware control of LED channels
dlpc_enabled = false

# Optional: Specific USB Product ID (defaults to auto-detect first TI device)
# dlpc_pid = 0x6401

# UV LED current drive level (0 - 400). Safe default: 150.
# Current formula: OutputCurrent = ((DriveLevel + 1) / 1024) * (0.15 / 0.036)
uv_led_drive_level = 150
```

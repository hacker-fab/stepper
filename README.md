# Hacker Fab - Stepper V2

This repository contains the source code for The Hacker Fab's open source stepper software. It contains a user interface for pattern selection, stage alignment, timed ultraviolet exposure, and optionally a live camera preview.

![stepper-gui](https://github.com/user-attachments/assets/3687a777-6f2b-4d9b-b7dc-8fc08cd7d4bf) ![stepper-assembly](https://github.com/user-attachments/assets/6211e7e7-3368-4a26-bbe2-425e88622b5c)

For more information about software setup, hardware assembly, or the Hacker Fab in general, please visit our [Gitbook](https://hacker-fab.gitbook.io/hacker-fab-space/fab-toolkit/patterning/lithography-stepper-v2-build-work-in-progress) or our [website](https://hackerfab.ece.cmu.edu/).

# GRBL Setup

This GUI is designed to be used with an Arduino running [GRBL](https://github.com/gnea/grbl) to move the stage,
possibly coupled with homing sensors for absolute positioning.

## Axes

The X/Y/Z axes of the stage are labeled according to the **point of view of the projector/camera**.
If any of the axis directions below are incorrect, swap and/or reverse the appropriate stepper connections.

Movement on the X and Y axes should pan the camera's view,
while movement on the Z axis should adjust focus.

The polarity of the X/Y axes is not currently used by the GUI; any direction is acceptable.

Movement in +Z should bring the camera and the chip's surface closer together.
The Z axis should be used for focus adjustment (i.e. in/out).

## Build Configuration (Sensors Only)

If proximity sensors or limit switches are installed on each axis,
some GRBL settings must be changed before flashing to the Arduino.
Open your GRBL folder's `config.h` and make the following changes.

Remove any existing defines for `HOMING_CYCLE_0` and `HOMING_CYCLE_1` and replace them
with the defines listed below. This enables homing.

```c
#define HOMING_CYCLE_0 ((1 << X_AXIS)|(1<<Y_AXIS))
#define HOMING_CYCLE_1 (1 << Z_AXIS)
```

Comment the `VARIABLE_SPINDLE` define.
This allows the Z axis limit switch to be used.

```c
//#define VARIABLE_SPINDLE
```

## Runtime Configuration

Once GRBL is flashed to an Arduino, you can use the serial monitor in the Arduino IDE
to adjust its configuration.
More detail on these settings is available
[on GRBL's wiki](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration).

Enable homing (if you have sensors).

```
$22=1
```

Adjust the homing direction invert mask (if you have sensors).
Bits 0, 1, and 2 in this value correspond to the X, Y, and Z axes.
For each axis, check if the limit switch is reached by moving in the positive direction or the negative direction.
If the axis requires negative movement, set the corresponding bit.
On CMU's setup, the limit switches are all reached by traveling in the negative direction,
so we use a value of 7 (invert all axes).

```
$23=7
```

Adjust the homing feed and homing seek (if you have sensors).

```
$24=10
$25=50
```

Adjust the homing pull-off (may need to be increased depending on your sensors' hysteresis).

```
$27=0.5
```

Adjust the steps per mm for the X, Y, and Z axes.
For CMU's setup, there is 8x microstepping, 200 steps per revolution, and 0.5mm per revolution,
so this value is 3200.

```
$100=3200
$101=3200
$102=3200
```

Adjust the max rate in mm per minute.
These numbers are empirical; you may need lower (or may be able to use higher) ones on your stage.

```
$110=120.000
$111=120.000
$112=120.000
```

Adjust the maximum acceleration in mm per sec^2.
Same caveats apply as for the max rate.

```
$120=5.000
$121=5.000
$122=5.000
```

Adjust the maximum travel of each axis in mm.
You should measure the travel and then back it off by 0.1mm,
or you could just approximate it as 15mm and hope for the best.

```
$130=15.000
$131=15.000
$132=15.000
```

Once these steps are complete, you should be able to home your stage using the `$H` command (if you have sensors)
and you are ready to use the GUI.

**Set `homing=true` under `[stage]` in your config file to enable the use of sensors.**

# Software Setup

## Python Setup with UV (Recommended)

The [UV project manager](https://github.com/astral-sh/uv) can be used to handle the Python version and packaging.

```bash
uv run src/gui.py
```

TODO: Doesn't currently work on MacOS due to TKinter problems

## Python Setup with venv

[virtual environment](https://docs.python.org/3/library/venv.html).

```bash
python -m venv venv       # name may change depending on system python
source venv/bin/activate  # depending on your shell, see venv docs
python --version          # ensure version is correct (<=3.10)
pip install -r requirements.txt
```

### Using a Basler (Pylon) camera

To use a Basler camera with the GUI, you will need to install `pylon` from
[Basler's website](https://www.baslerweb.com/en-us/downloads/software/).

Select the "Software Suite" download under Pylon, as it contains useful tools for debugging the camera view.

After a few steps, the installer will prompt you to **restart the computer**.
Do not delay this restart as the Pylon installer has several more steps that are run only after restarting.

### Using a FLIR camera

If you are using the FLIR camera with your stage,
ask in the Hacker Fab Discord for an invitation to the FLIR repository.

Then, update the git submodules to add support for the FLIR:

```bash
git submodule init
git submodule update
```

If the submodule update doesn't work, clone the `flir-private` repo as follows:

```bash
cd src/camera
git clone git@github.com:hacker-fab/flir-private.git flir
```

In order to use the FLIR camera, you will need to use a version of Python **at or before 3.10.**
Specify a version when creating your venv to make this work:

```bash
python3.10 -m venv venv   # name may change depending on system python
```

This restriction does not apply to other camera brands.

> [!NOTE]
> This software is in active development, and features are subject to change. Though each change to the main branch has been tested, there remains a chance that some bugs are undetected. To report a bug or to suggest additional features, please create an issue on this repository.

## Configuration

The GUI uses a `config.toml` file (in the [TOML](https://toml.io/en/) format) for configuration.
A sample configuration file with an explanation of the settings is shown below.

```toml
# This section configures the camera used for the GUI's preview.
[camera]
# Available options are:
# "webcam" (for a generic USB camera),
# "basler" (for a Basler/Pylon camera)
# "flir" (for a FLIR camera)
# "none" (to disable camera)
type = "webcam"
# The index field is optional and is only used to select which of multiple webcams should be used,
# e.g. on a laptop where there may be a builtin webcam in addition to an external USB camera.
index = 1
# The output from the camera is typically too large to show at full resolution.
# This parameter adjusts the size of the camera feed before it is displayed in the GUI.
gui-scale = 0.25
# The following two values adjust the *camera* exposure in microseconds when viewing red or UV light.
# Note that using values that are not a multiple of 4167 can lead to flickering.
red-exposure = 4167.0
uv-exposure = 25000.0

# This section configures the motion stage
[stage]
# Set enabled to false to disable all motion.
enabled = true
# Set homing to false if your stage does not have limit sensors
homing = true
# Select the correct serial port for the device running GRBL.
# The correct serial port can be checked with Device Manager on Windows.
port = "COM6"
# GRBL's baud rate is almost always 115200, do not change this value
baud-rate = 115200

# This section configures alignment marker detection
[alignment]
# Enable or disable real-time detection of alignment markers
enabled = false
# Path to the YOLO model weights file
model_path = "best.pt"
# Alignment marker reference coordinates (in pixels)
right_marker_x = 1634.0  # x-coordinate for markers on the right side
top_marker_y = 117.5     # y-coordinate for markers on the top
bottom_marker_y = 1001.5 # y-coordinate for markers on the bottom
left_marker_x = 0.0      # x-coordinate for markers on the left side
# Scaling factors for converting normalized differences to stage movements (in µm)
x_scale_factor = -1040   # Scaling factor for x-axis movements
y_scale_factor = -580    # Scaling factor for y-axis movements
```

# Hacker Fab - Stepper V2

This repository contains the source code for The Hacker Fab's open source stepper software. It contains a user interface for pattern selection, stage alignment, timed ultraviolet exposure, and optionally a live camera preview.

![stepper-gui](https://github.com/user-attachments/assets/3687a777-6f2b-4d9b-b7dc-8fc08cd7d4bf)   ![stepper-assembly](https://github.com/user-attachments/assets/6211e7e7-3368-4a26-bbe2-425e88622b5c)

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

It is strongly recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html).

```bash
python3 -m venv venv
source venv/bin/activate  # depending on your shell, see venv docs
pip install -r requirements.txt
```

> [!NOTE]
> This software is in active development, and features are subject to change. Though each change to the main branch has been tested, there remains a chance that some bugs are undetected. To report a bug or to suggest additional features, please create an issue on this repository.

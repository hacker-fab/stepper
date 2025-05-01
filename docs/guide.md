# Stepper Software Guide

This is a guide to the photolithography stepper software contained in this repository.

## Overview

TODO

## Quick Start

TODO

## Configuration

TODO

## Tiling

TODO

## Autofocus

TODO

## Detection

The GUI allows for alignment marker detection using `ultralytics`.

Any `ultralytics` YOLO model is supported. In this repository, we include the `best.pt` file in the root directory, which contains the weights of a finetuned 2.6M parameter YOLOv11n model, which is capable of detecting alignment markers in real time.

To train a YOLO model on your own data using `ultralytics`, see [this guide](model.md).

There are two configuration options for detection, housed under the `alignment` section in the TOML config.

These are:

```toml
[alignment]
# Enable or disable real-time detection of alignment markers
enabled = false
# Path to the YOLO model weights file
model_path = "best.pt"
```

The GUI has a checkbox which can be used to enable or disable real-time alignment.

## Alignment

The alignment system allows for automatic positioning of chips on the stage for layered patterning. The system uses the YOLO model to detect alignment markers and calculates the necessary stage movements to align the patterns precisely.

#### How Alignment Works

1. The system detects alignment markers in the camera image using the YOLO model specified in the configuration.
2. For each detected marker, the system calculates its position relative to predefined reference coordinates.
3. Based on these differences, the system computes the necessary stage movements to align the chip correctly.

#### Alignment Parameters

```toml
[alignment]
# ...
# Alignment marker reference coordinates (in pixels)
right_marker_x = 1820.0  # x-coordinate for markers on the right side
top_marker_y = 269.0     # y-coordinate for markers on the top
bottom_marker_y = 1075.0 # y-coordinate for markers on the bottom
left_marker_x = 280.0    # x-coordinate for markers on the left side
# Scaling factors for converting normalized differences to stage movements (in µm)
x_scale_factor = -1100   # Scaling factor for x-axis movements
y_scale_factor = 800     # Scaling factor for y-axis movements
```

- **Reference Coordinates**:

  - **right_marker_x**: The expected x-coordinate (in pixels) for alignment markers on the right side of the image.
  - **left_marker_x**: The expected x-coordinate for markers on the left side.
  - **top_marker_y**: The expected y-coordinate for markers at the top of the image.
  - **bottom_marker_y**: The expected y-coordinate for markers at the bottom.

  These coordinates represent the "ideal" positions where markers should be in the camera frame when the chip is perfectly aligned.

- **Scaling Factors**:

  - **x_scale_factor**: Converts the normalized x-coordinate differences (between detected markers and reference positions) into stage movement distances in micrometers (µm).
  - **y_scale_factor**: Similar conversion factor for the y-axis.

#### Calibrating Alignment Parameters

To calibrate the alignment system for your specific setup:

1. **Reference Coordinates**: Place a perfectly aligned reference chip on the stage. Note the pixel coordinates of the alignment markers in the camera image and update the reference coordinate values accordingly.

2. **Scaling Factors**: These values depend on your camera's field of view and the mechanical properties of your stage. To calibrate:
   - Move the stage by a known distance (e.g., 100 µm)
   - Measure how many pixels the markers moved in the camera image
   - Calculate the scaling factor as: `(stage movement in µm) / (marker movement in normalized coordinates)`
   - Adjust the sign (positive or negative) based on the relative directions of stage movement and camera coordinates

Proper calibration of these parameters is crucial for accurate alignment.

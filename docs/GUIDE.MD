# Stepper Software Guide

This is a guide to the photolithography stepper software contained in this repository.

### Overview

TODO

### Quick Start

TODO

### Configuration

TODO

### Tiling

TODO

### Autofocus

TODO

### Detection

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

### Alignment

TODO

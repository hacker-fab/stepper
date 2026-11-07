# Stepper Software User Guide

This guide covers the operation of the Hacker Fab Stepper V2 software, including user interface navigation, chip project management, stage positioning, pattern alignment, and exposure workflows.

---

## 1. Application Overview

The Stepper V2 software provides a unified desktop interface for operating the Hacker Fab photolithography stepper. It integrates:
- **Optical Inspection**: Live camera feed with digital zoom, pan, and measurement crosshairs.
- **Pattern Projection**: Projector control with automated LED channel gating.
- **Motion Control**: Real-time 3-axis stage manipulation, homing, and click-to-move positioning.
- **Computer Vision Alignment**: Automated chip alignment using YOLO marker detection.
- **Exposure Management**: Timed exposure execution with visual exposure mapping and project persistence.

---

## 2. User Interface Layout

The main window is organized into functional panels:

### Camera Viewport
- **Live Feed**: Displays the real-time optical view from the configured camera.
- **Pan & Zoom**: Use the mouse wheel to zoom in/out and drag to pan across the field of view. The **Reset Zoom** button restores the default view scale.
- **Dual Crosshairs**: Includes a primary center crosshair and an optional secondary crosshair to measure distances across the chip surface directly in pixel space.
- **Snapshot Tool**: Click the snapshot button to save the current camera frame as a timestamped PNG image in the `stepper_captures/` folder.

### Projector Preview
- Shows the current exposure mask or alignment pattern loaded into the project.
- Indicates the active illumination mode (**Red Alignment Mode** vs. **UV Exposure Mode**).

### Interactive Stage Map
- Provides a bird's-eye 2D view of the stage travel bounds and current position.
- **Click-to-Move**: Click anywhere on the stage map to command the stage to translate to that coordinate.
- **Exposure History**: Renders color-coded boxes showing the positions and dimensions of all completed exposures in the active project.

### Machine Control Panel
- **Axis Jogging**: Manual jog controls for the X, Y, and Z axes.
- **Step Sizes**: Selectable step increments (e.g., 0.001 mm, 0.01 mm, 0.1 mm, 1.0 mm, 5.0 mm).
- **Homing**: Trigger the automated hardware homing routine (for stages equipped with limit sensors).
- **Status Indicators**: Real-time display of current stage coordinates and hardware connection status.

### Workflow Panel
- Step-by-step workflow tabs for project loading, pattern selection, alignment, and exposure sequence execution.

---

## 3. Project & Exposure Workflow

### Step 1: Initialize or Load a Project
1. Open or create a Chip Project (`.chip` file) from the Workflow Panel.
2. The project preserves stage calibration, pattern assignments, exposure logs, and alignment coordinates.

### Step 2: Load a Pattern Mask
1. Load a pattern image (`.png`, `.jpg`, or `.bmp`).
2. If working with IC layout files, convert your GDSII file to high-resolution PNG using the bundled conversion utility:
   ```bash
   python gds2png.py layout.gds -o mask.png --dpi 2400
   ```
3. The pattern will appear in the Projector Preview panel.

### Step 3: Align and Focus the Stage
1. Switch to **Red Mode** to illuminate the chip with non-actinic red light.
2. Use the Machine Control Panel to jog X and Y to locate your alignment targets or die origin.
3. Adjust the Z axis until the surface features and alignment markers are sharp and in focus.

### Step 4: Execute Exposure
1. Set the desired **Exposure Duration** (in seconds) in the Workflow Panel.
2. Click **Start Exposure**. The software will:
   - Hardware-gate the projector to activate the UV light source.
   - Hold the exposure for the programmed duration with a live progress bar.
   - Safely turn off the UV light source upon completion.
   - Record the exposure footprint onto the Interactive Stage Map.

---

## 4. Alignment Marker Detection & Auto-Alignment

The stepper features real-time computer vision detection to align chips automatically across multiple photolithography layers.

### How Alignment Works
1. A finetuned YOLOv11 model detects alignment markers within the live camera frame.
2. The software calculates the centroid offset of detected markers against calibrated reference coordinates.
3. The stage controller executes closed-loop correction moves along the X and Y axes to center the markers accurately.

### Configuration Parameters
Alignment parameters are specified in the `[alignment]` section of `config.toml`:

```toml
[alignment]
# Enable real-time YOLO detection
enabled = false

# Path to trained YOLO weights
model_path = "ckpts/best.pt"

# Ideal marker coordinates in camera pixel space
right_marker_x = 1820.0
top_marker_y = 269.0
bottom_marker_y = 1075.0
left_marker_x = 280.0

# Conversion factors: Normalized camera offset -> stage movement (in µm)
x_scale_factor = -1100
y_scale_factor = 800
```

### Calibrating Alignment Parameters
1. **Reference Coordinates**:
   - Place a bare silicon wafer or blank chip on the stage.
   - In Red Mode, project a pattern containing alignment markers.
   - Enable detection in the GUI to view marker centroid outputs.
   - Update `right_marker_x`, `left_marker_x`, `top_marker_y`, and `bottom_marker_y` with the observed pixel coordinates.

2. **Scaling Factors (`x_scale_factor`, `y_scale_factor`)**:
   - Command the stage to move a known distance (e.g., 100 µm).
   - Observe the marker displacement in camera pixels.
   - Compute: `scale = (stage displacement in µm) / (marker displacement in normalized coordinates)`.
   - Update the sign depending on whether the camera view is inverted relative to stage motion.

To train custom marker detection models, consult the [Alignment Model Training Guide](alignment_model_training.md).

---

## 5. Tiling (Step-and-Repeat)

For patterning larger substrates or exposing multi-die arrays:

1. Enable `[tiling]` in `config.toml`:
   ```toml
   [tiling]
   enabled = true
   ```
2. In the Workflow Panel, configure the tiling matrix:
   - **Rows & Columns**: Number of exposures along the grid.
   - **X & Y Pitch**: Center-to-center distance between dies (in mm).
3. The software will automatically step the stage, execute the exposure sequence, and record all exposures sequentially across the substrate.

---

## 6. Autofocus

For automated focus compensation:
```toml
[autofocus]
enabled = false
```
When enabled, the autofocus operation evaluates image sharpness across incremental Z steps to establish optimal focal distance before exposing.

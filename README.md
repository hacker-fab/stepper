# Hacker Fab - Stepper V2

This repository contains the software for The Hacker Fab's open source photolithography stepper. It provides a desktop interface for pattern projection, micropositioning stage control, real-time computer vision alignment, timed ultraviolet (UV) exposure, and exposure footprint mapping.

![stepper-gui](https://github.com/user-attachments/assets/3687a777-6f2b-4d9b-b7dc-8fc08cd7d4bf) ![stepper-assembly](https://github.com/user-attachments/assets/6211e7e7-3368-4a26-bbe2-425e88622b5c)

For more information regarding physical hardware assembly, lithography toolkits, or the Hacker Fab initiative, visit our [GitBook](https://hacker-fab.gitbook.io/hacker-fab-space/fab-toolkit/patterning/lithography-stepper-v2-build-work-in-progress) and our [website](https://hackerfab.ece.cmu.edu/).

---

## Documentation Index

Detailed guides are available in the [`docs/`](docs/) directory:

- 📖 **[User Guide](docs/user_guide.md)**: Full GUI walkthrough, pattern loading, stage navigation, exposure workflows, alignment calibration, and tiling.
- 🔧 **[Hardware Setup Guide](docs/hardware.md)**: Physical assembly configurations, GRBL firmware flashing/EEPROM setup, OMM stage setup, camera drivers, and DLPC6540 projector USB control.
- 🎯 **[Alignment Model Training Guide](docs/alignment_model_training.md)**: Capturing dataset images, annotating markers via Roboflow, and fine-tuning YOLOv11 alignment models with Ultralytics.
- 💻 **[Developer Guide](docs/developer_guide.md)**: Software architecture, abstraction layers, and contribution guidelines for both human and AI agents.

---

## Quick Start

### Python Setup with UV (Recommended)

The [UV project manager](https://github.com/astral-sh/uv) handles virtual environment creation, Python runtime management (Python ≥ 3.10), and dependency resolution automatically:

```bash
uv run src/gui.py
```

To run with a custom configuration file:
```bash
uv run src/gui.py custom_config.toml
```

### Python Setup with Standard venv

Ensure you are using **Python 3.10 or newer**:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To install optional hardware drivers (e.g. Basler camera, GRBL, OMM stage, alignment YOLO models, latent vision):
```bash
pip install -e '.[grbl,omm,alignment,basler,dlpc,latent_vision]'
```

Launch the GUI:
```bash
python src/gui.py
```

---

## Configuration Overview

The software uses a [TOML](https://toml.io/en/) configuration file to configure hardware drivers and exposure operations. On startup, the application looks for `default.toml` in the repository root or accepts a `.toml` path as a command-line argument.

```toml
# Camera configuration
[camera]
type = "webcam"       # Options: "webcam", "pylon", "flir", "amscope", "dummy", "none"

[camera.webcam]
index = 0

# Motion stage configuration
[stage]
type = "omm"          # Options: "omm", "grbl", "dummy"

[stage.omm]
port = "COM6"
baud-rate = 921600

# Projector and LED hardware channel gating
[projector]
dlpc_enabled = false
uv_led_drive_level = 150

# Automated operations
[alignment]
enabled = false
model_path = "ckpts/best.pt"

[tiling]
enabled = false

[autofocus]
enabled = false
```

Detailed descriptions and tuning instructions for all hardware and operational parameters can be found in the documentation below.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

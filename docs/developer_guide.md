# Developer & AI Agent Architecture Guide

This document outlines the software architecture, design patterns, and engineering rules for human developers and AI coding agents working on the Hacker Fab Stepper V2 codebase.

---

## 1. Architectural Overview

The application follows a decoupled, event-driven architecture designed to separate core lithography logic and hardware communication from the graphical user interface.

```
┌────────────────────────────────────────────────────────┐
│                   PySide6 UI Layer                     │
│ (MainWindow, CameraView, StageMap, WorkflowPanel, etc.)│
└───────────────────────────┬────────────────────────────┘
                            │ Qt Signals & Slots
┌───────────────────────────▼────────────────────────────┐
│                    QtEngineBridge                      │
│      (Translates Core EventBus events to Qt signals)   │
└───────────────────────────┬────────────────────────────┘
                            │ Thread-safe EventBus
┌───────────────────────────▼────────────────────────────┐
│                   Core Engine Layer                    │
│      StepperEngine  ───  ExecutionContext ─── Project  │
│                           │                            │
│                  OperationManager                      │
│     (Exposure, Alignment, Tiling, Autofocus, Jog)      │
└───────────────────────────┬────────────────────────────┘
                            │ Abstract Classes
┌───────────────────────────▼────────────────────────────┐
│              Hardware Abstraction Layer (HAL)          │
│   StageController        CameraModule        Projector │
│(GRBL / OMM / Dummy)   (Webcam/Pylon/Dummy)   (DLPC6540)│
└────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems & Patterns

### A. Headless Engine & Execution Context (`src/core/`)
- **`StepperEngine`**: The central coordinator. It initializes hardware controllers, loads projects, and manages the execution context and operations. It has no GUI dependencies and can run headlessly.
- **`ExecutionContext`**: A container passed to operations containing references to `stage`, `camera`, `projector`, and the active `project`.
- **`ChipProject`**: The project persistence model representing the wafer state, loaded patterns, exposure history, and coordinate origins.

### B. Event-Driven Communication (`src/core/events.py`, `src/ui/bridge.py`)
- **`EventBus`**: Provides thread-safe publish-subscribe messaging for engine-level events (e.g., `STAGE_POSITION_CHANGED`, `EXPOSURE_PROGRESS`, `FRAME_ACQUIRED`, `WARNING_MESSAGE`).
- **`QtEngineBridge`**: Acts as an adapter between the core `EventBus` and the PySide6 UI thread, re-emitting engine events as Qt signals to prevent cross-thread UI violations.

### C. Operations Pattern (`src/operations/`)
All long-running tasks (exposures, auto-alignments, autofocus sweeps, tiling sequences) are encapsulated as discrete `Operation` subclasses:
- Every operation implements `execute(context: ExecutionContext, progress_callback: Callable[[float, str], None]) -> Any`.
- Operations are executed through the `OperationManager`, allowing background execution, progress reporting, and graceful cancellation.

### D. Hardware Abstraction Layer (HAL)
Hardware interactions are abstracted behind standard interfaces:
- **`StageController` (`src/stage_control/`)**: Abstract base class defining 3-axis motion primitives (`move_absolute`, `move_relative`, `home`, `get_position`, `is_moving`, `stop`). Implementations: `GrblStage`, `OMMStage`, `DummyStage`.
- **`CameraModule` (`src/camera/`)**: Abstract base class defining acquisition primitives (`get_frame`, `is_open`, `close`). Implementations: `Webcam`, `BaslerPylon`, `FlirCamera`, `AmscopeCamera`, `DummyCamera`.
- **Factory Discovery**: Implementations are resolved dynamically via `get_stage_controller(config)` and `get_camera(config)`, with fallbacks to simulated dummy devices.

---

## 3. Configuration Architecture (`default.toml`)

Configuration is managed via TOML files and organized into modular sections:
- **`[camera]`**: Configures camera type and sub-tables (`[camera.dummy]`, `[camera.webcam]`, `[camera.pylon]`).
- **`[stage]`**: Configures stage type (`omm`, `grbl`, `dummy`) and sub-tables (`[stage.omm]`, `[stage.grbl]`, `[stage.dummy]`).
- **`[projector]`**: Hardware LED channel gating via DLPC6540.
- **`[alignment]`**, **`[tiling]`**, **`[autofocus]`**: Parameters for operations.

Configs are injected directly into operations and hardware factories during initialization rather than being stored as global mutable singletons.

---

## 4. Engineering & Coding Rules

When modifying or extending this codebase, adhere to the following principles to keep the code clean and readable:

1. **Single Source of Truth**
   - Events should be emitted only from the hardware or data classes that "generates" events, not from UI. Data should be kept in the chip project, do not store multiple copies at multiple places.

2. **Reusable Operations**
   - Almost all action (like moving, aligning, exposure, etc.) should be implemented as operations, and they can be ran either through the UI or by other operations.
   - Do not perform sequential hardware moves or timed loops directly inside UI widget callbacks.
   - Encapsulate the sequence in a subclass of `Operation` inside `src/operations/` and run it through `OperationManager`.

3. **GUI & Core Separation**:
   - **Never** import `PySide6`, `Qt`, or UI widgets inside `src/core/`, `src/operations/`, `src/stage_control/`, or `src/camera/`.
   - Core components must communicate with the UI exclusively via the `EventBus` and `QtEngineBridge`.
   
4. **Thread Safety & Non-Blocking Execution**:
   - Hardware communication and operations must not block the Qt UI event loop.
   - Use background threads/workers managed by the bridge or operation manager.

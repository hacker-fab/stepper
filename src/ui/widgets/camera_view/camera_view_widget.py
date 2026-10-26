import time
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Callable, Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera import CameraModule
from core.engine import StepperEngine
from ui.bridge import QtEngineBridge
from .camera_viewport import CameraViewport

try:
    from lib.latent_vision_model import (
        DEFAULT_LATENT_VISION_CHECKPOINT,
        LatentVisionCNN,
        LatentVisionWorker,
        load_latent_vision_model,
        resolve_checkpoint_path,
    )
    LATENT_VISION_AVAILABLE = True
except ImportError:
    LatentVisionCNN = None  # type: ignore
    LatentVisionWorker = None  # type: ignore
    load_latent_vision_model = None  # type: ignore
    DEFAULT_LATENT_VISION_CHECKPOINT = Path("checkpoints/latent_vision/latest_model.pth")
    resolve_checkpoint_path = lambda p: Path(p)  # type: ignore
    LATENT_VISION_AVAILABLE = False


# Latent vision overlay opacity constant (easy to change)
LATENT_VISION_OPACITY: float = 0.5
_resolve_checkpoint_path = resolve_checkpoint_path


class MaskRenderWorker(threading.Thread):
    """Background worker that converts model output masks into renderable QImages.

    Performs heavy image processing on a separate thread to keep event handlers responsive.
    """

    def __init__(self, on_image_ready: Callable[[QImage], None]):
        super().__init__(daemon=True)
        self.on_image_ready = on_image_ready
        self._pending_mask: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()

    def submit_mask(self, mask: np.ndarray):
        with self._lock:
            self._pending_mask = mask
        self._wake_event.set()

    def stop(self):
        self._stop_event.set()
        self._wake_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self._wake_event.wait()
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            with self._lock:
                mask_to_process = self._pending_mask
                self._pending_mask = None

            if mask_to_process is None:
                continue

            try:
                h, w = mask_to_process.shape[:2]
                if mask_to_process.ndim == 3:
                    mask_2d = mask_to_process[:, :, 0]
                else:
                    mask_2d = mask_to_process

                alpha = np.clip(mask_2d * 255.0, 0.0, 255.0).astype(np.uint8)
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[:, :, 1] = alpha  # Green channel premultiplied
                overlay[:, :, 3] = alpha  # Alpha channel
                mask_qimage = QImage(
                    overlay.data, w, h, overlay.strides[0], QImage.Format_RGBA8888_Premultiplied
                ).copy()

                self.on_image_ready(mask_qimage)
            except Exception as e:
                print(f"[MaskRenderWorker] Error processing mask: {e}")


class CameraViewWidget(QWidget):
    """Live camera view with crosshairs, FPS tracking, zooming, and snapshot capture."""

    mask_rendered_signal = Signal(object)

    def __init__(
        self,
        engine: StepperEngine,
        bridge: QtEngineBridge,
        camera: Optional[CameraModule] = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self.camera = camera

        self.current_frame: Optional[np.ndarray] = None
        self.current_qimage: Optional[QImage] = None
        self.show_crosshairs = True
        self.is_moving_crosshair = False

        # Latent Vision state
        self.latent_vision_enabled: bool = False
        self._latest_latent_mask: Optional[np.ndarray] = None
        self._latest_latent_mask_qimage: Optional[QImage] = None
        self._latent_mask_lock = threading.RLock()
        self._latent_worker: Optional[LatentVisionWorker] = None
        self._mask_render_worker: Optional[MaskRenderWorker] = None
        self.mask_rendered_signal.connect(self._on_mask_rendered)

        # FPS metrics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        if self.camera and not self.camera.is_open():
            if not self.camera.open():
                print("Warning: Camera failed to open")

        self._init_ui()

        # Listen for camera frame ready event from engine bridge
        self.bridge.camera_frame_ready.connect(self._on_frame_ready)

    @property
    def latest_latent_mask(self) -> Optional[np.ndarray]:
        with self._latent_mask_lock:
            return self._latest_latent_mask

    @latest_latent_mask.setter
    def latest_latent_mask(self, mask: Optional[np.ndarray]):
        with self._latent_mask_lock:
            self._latest_latent_mask = mask
            self._latest_latent_mask_qimage = None
        if self.latent_vision_enabled and mask is not None:
            if self._mask_render_worker is not None and self._mask_render_worker.is_alive():
                self._mask_render_worker.submit_mask(mask)
            else:
                self.viewport.set_latent_mask(mask)
        else:
            self.viewport.set_latent_mask(None)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header bar with controls
        header = QHBoxLayout()
        header.setSpacing(6)

        self.crosshair_cb = QCheckBox("Crosshair")
        self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(self._on_crosshair_toggled)
        header.addWidget(self.crosshair_cb)

        self.latent_vision_cb = QCheckBox("Latent Vision")
        self.latent_vision_cb.setChecked(False)
        if LATENT_VISION_AVAILABLE:
            self.latent_vision_cb.setToolTip("Toggle latent vision model overlay")
        else:
            self.latent_vision_cb.setEnabled(False)
            self.latent_vision_cb.setToolTip(
                "PyTorch is not installed. Install with: pip install '.[latent_vision]'"
            )
        self.latent_vision_cb.toggled.connect(self._on_latent_vision_toggled)
        header.addWidget(self.latent_vision_cb)

        self.move_crosshair_btn = QPushButton("Move Crosshair")
        self.move_crosshair_btn.setCheckable(True)
        self.move_crosshair_btn.setStyleSheet("font-weight: bold;")
        self.move_crosshair_btn.toggled.connect(self._on_move_crosshair_toggled)
        header.addWidget(self.move_crosshair_btn)

        self.center_crosshair_btn = QPushButton("Center")
        self.center_crosshair_btn.setToolTip("Recenter crosshair to image center")
        self.center_crosshair_btn.clicked.connect(self._on_center_crosshair_clicked)
        header.addWidget(self.center_crosshair_btn)

        header.addSpacing(6)

        # Zoom Controls
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(26)
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        header.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("100%")
        self.btn_zoom_reset.setFixedWidth(52)
        self.btn_zoom_reset.setToolTip("Reset Zoom and Pan (Fit)")
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        header.addWidget(self.btn_zoom_reset)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(26)
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        header.addWidget(self.btn_zoom_in)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        header.addWidget(self.snapshot_btn)

        header.addStretch()

        self.res_label = QLabel("No Camera")
        self.res_label.setStyleSheet("font-size: 11px;")
        header.addWidget(self.res_label)

        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet("font-size: 11px;")
        header.addWidget(self.fps_label)

        layout.addLayout(header)

        # Main viewport canvas
        self.viewport = CameraViewport(self)
        self.viewport.zoom_changed.connect(self._on_viewport_zoom_changed)
        layout.addWidget(self.viewport, stretch=1)

    def _on_viewport_zoom_changed(self, zoom_level: float):
        pct = int(round(zoom_level * 100))
        self.btn_zoom_reset.setText(f"{pct}%")

    def _on_crosshair_toggled(self, checked: bool):
        self.show_crosshairs = checked
        self.move_crosshair_btn.setEnabled(checked)
        self.center_crosshair_btn.setEnabled(checked)
        if not checked and self.move_crosshair_btn.isChecked():
            self.move_crosshair_btn.setChecked(False)
        self.viewport.set_show_crosshairs(checked)

    def _on_latent_vision_toggled(self, checked: bool):
        self.latent_vision_enabled = checked
        if checked:
            if self._latent_worker is None or not self._latent_worker.is_alive():
                self._start_latent_worker()
            if self._mask_render_worker is None or not self._mask_render_worker.is_alive():
                self._start_mask_render_worker()
            if self.current_frame is not None and self._latent_worker is not None:
                self._latent_worker.submit_frame(self.current_frame)
            with self._latent_mask_lock:
                mask = self._latest_latent_mask
            if mask is not None and self._mask_render_worker is not None:
                self._mask_render_worker.submit_mask(mask)
        else:
            with self._latent_mask_lock:
                self._latest_latent_mask = None
                self._latest_latent_mask_qimage = None
            self.viewport.set_latent_mask(None)

    def _start_latent_worker(self):
        if not LATENT_VISION_AVAILABLE or LatentVisionWorker is None:
            return
        self._latent_worker = LatentVisionWorker(
            checkpoint_path=DEFAULT_LATENT_VISION_CHECKPOINT,
            on_mask_ready=self._on_latent_mask_ready,
        )
        self._latent_worker.start()

    def _start_mask_render_worker(self):
        if self._mask_render_worker is None or not self._mask_render_worker.is_alive():
            self._mask_render_worker = MaskRenderWorker(on_image_ready=self.mask_rendered_signal.emit)
            self._mask_render_worker.start()

    def _on_latent_mask_ready(self, mask: np.ndarray):
        """Worker callback: saves mask and offloads heavy image processing to MaskRenderWorker."""
        with self._latent_mask_lock:
            self._latest_latent_mask = mask
        if self.latent_vision_enabled:
            if self._mask_render_worker is None or not self._mask_render_worker.is_alive():
                self._start_mask_render_worker()
            self._mask_render_worker.submit_mask(mask)

    def _on_mask_rendered(self, mask_qimage: QImage):
        """GUI thread slot: receives pre-rendered QImage and updates the viewport with zero processing."""
        with self._latent_mask_lock:
            self._latest_latent_mask_qimage = mask_qimage
        if self.latent_vision_enabled:
            self.viewport.set_latent_mask(mask_qimage)

    def _on_move_crosshair_toggled(self, checked: bool):
        self.is_moving_crosshair = checked
        self.viewport.set_moving_crosshair(checked)

    def _on_center_crosshair_clicked(self):
        self.viewport.crosshair_u = 0.5
        self.viewport.crosshair_v = 0.5
        self.viewport.has_secondary_crosshair = False
        self.viewport.update()

    def _zoom_in(self):
        self.viewport.zoom(1.25)

    def _zoom_out(self):
        self.viewport.zoom(1.0 / 1.25)

    def _zoom_reset(self):
        self.viewport.reset_zoom()

    def _update_zoom_label(self):
        pct = int(round(self.viewport.zoom_level * 100))
        self.btn_zoom_reset.setText(f"{pct}%")

    def _take_snapshot(self):
        if self.current_frame is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        capture_dir = Path("stepper_captures")
        capture_dir.mkdir(exist_ok=True)
        filename = capture_dir / f"manual_snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), self.current_frame)
        print(f"Saved snapshot to {filename}")
        self.bridge.status_message.emit(f"Snapshot saved: {filename.name}")

    def cleanup(self):
        try:
            self.bridge.camera_frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass
        if self._mask_render_worker is not None:
            self._mask_render_worker.stop()
            self._mask_render_worker = None
        if self._latent_worker is not None:
            self._latent_worker.stop()
            self._latent_worker = None
        if self.camera and self.camera.is_open():
            try:
                self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def _on_frame_ready(self, frame: Optional[np.ndarray]):
        if frame is None:
            return

        self.current_frame = frame

        if self.latent_vision_enabled:
            if self._latent_worker is not None and self._latent_worker.is_alive():
                self._latent_worker.submit_frame(frame)
            if self.viewport.latent_mask_image is None and self._latest_latent_mask is not None:
                with self._latent_mask_lock:
                    mask = self._latest_latent_mask_qimage or self._latest_latent_mask
                self.viewport.set_latent_mask(mask)

        self._render_frame(frame)

        # FPS calculate
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            self.current_fps = self.frame_count / dt
            self.fps_label.setText(f"{self.current_fps:.1f} FPS")
            self.frame_count = 0
            self.last_fps_time = now

    def _render_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self.viewport.set_frame(frame)
        self.current_qimage = self.viewport.current_qimage
        self.res_label.setText(f"{w}x{h}")

    def get_display_image(self) -> Optional[QImage]:
        """Returns the current display image rendered by the viewport."""
        return self.viewport.get_display_image()



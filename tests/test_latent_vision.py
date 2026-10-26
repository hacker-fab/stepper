import os
from pathlib import Path
import sys
import time
import unittest

import cv2
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
try:
    import torch
    from lib.latent_vision_model import (
        LatentVisionCNN,
        load_latent_vision_model,
    )
    LATENT_VISION_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    LatentVisionCNN = None  # type: ignore
    load_latent_vision_model = None  # type: ignore
    LATENT_VISION_AVAILABLE = False

from projector import ProjectorController
from stage_control import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.camera_view.camera_view_widget import (
    DEFAULT_LATENT_VISION_CHECKPOINT,
    LATENT_VISION_OPACITY,
    CameraViewWidget,
    LatentVisionWorker,
    MaskRenderWorker,
    _resolve_checkpoint_path,
)


class MockProjector(ProjectorController):
    def projector_size(self):
        return 1920, 1080

    def update_display(self):
        pass

# Qt headless mode for test environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication

_APP = None


def get_qapp():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class TestLatentVisionModel(unittest.TestCase):
    @unittest.skipUnless(LATENT_VISION_AVAILABLE, "PyTorch not installed")
    def test_model_forward_and_predict_shapes(self):
        import torch

        model = LatentVisionCNN()
        # Input shape: (B, 3, H, W)
        x = torch.rand(1, 3, 240, 320)
        y = model.predict(x)
        # Expected output shape: (1, 1, 60, 80)
        self.assertEqual(y.shape, (1, 1, 60, 80))
        # Sigmoid output in [0, 1]
        self.assertTrue((y >= 0.0).all() and (y <= 1.0).all())

    @unittest.skipUnless(LATENT_VISION_AVAILABLE, "PyTorch not installed")
    def test_load_checkpoint(self):
        ckpt_path = _resolve_checkpoint_path(DEFAULT_LATENT_VISION_CHECKPOINT)
        self.assertTrue(ckpt_path.is_file(), f"Checkpoint {ckpt_path} not found")
        model = load_latent_vision_model(ckpt_path)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LatentVisionCNN)

    @unittest.skipUnless(LATENT_VISION_AVAILABLE, "PyTorch not installed")
    def test_load_nonexistent_checkpoint(self):
        model = load_latent_vision_model("nonexistent_model.pth")
        self.assertIsNone(model)


class TestLatentVisionWorker(unittest.TestCase):
    @unittest.skipUnless(LATENT_VISION_AVAILABLE, "PyTorch not installed")
    def test_worker_processing(self):
        ckpt_path = _resolve_checkpoint_path(DEFAULT_LATENT_VISION_CHECKPOINT)
        received_masks = []

        def on_mask(mask):
            received_masks.append(mask)

        worker = LatentVisionWorker(
            checkpoint_path=ckpt_path,
            on_mask_ready=on_mask,
            device="cpu",
        )
        worker.start()

        try:
            # Submit a test frame 480x640
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            worker.submit_frame(frame)

            # Wait for mask callback
            for _ in range(50):
                if received_masks:
                    break
                time.sleep(0.05)

            self.assertGreater(len(received_masks), 0)
            mask = received_masks[-1]
            self.assertEqual(mask.shape, (480, 640))
        finally:
            worker.stop()
            worker.join(timeout=2.0)


class TestCameraViewWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qapp()

    def setUp(self):
        self.stage = DummyStage()
        self.projector = MockProjector()
        self.camera = DummyCamera(640, 480)
        self.engine = StepperEngine(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
        )
        self.bridge = QtEngineBridge(self.engine)
        self.widget = CameraViewWidget(self.engine, self.bridge, self.camera)

    def tearDown(self):
        self.widget.cleanup()

    def test_ui_toggle_exists(self):
        self.assertIsNotNone(self.widget.latent_vision_cb)
        self.assertEqual(self.widget.latent_vision_cb.text(), "Latent Vision")
        self.assertFalse(self.widget.latent_vision_cb.isChecked())

    def test_toggle_latent_vision(self):
        self.assertFalse(self.widget.latent_vision_enabled)
        self.widget.latent_vision_cb.setChecked(True)
        self.assertTrue(self.widget.latent_vision_enabled)

        # Toggle back off
        self.widget.latent_vision_cb.setChecked(False)
        self.assertFalse(self.widget.latent_vision_enabled)
        self.assertIsNone(self.widget.latest_latent_mask)

    def test_latent_vision_blending(self):
        # Set a synthetic mask
        h, w = 480, 640
        fake_mask = np.ones((h, w), dtype=np.float32)  # value 1.0 everywhere
        with self.widget._latent_mask_lock:
            self.widget.latest_latent_mask = fake_mask
        self.widget.latent_vision_enabled = True

        # Blue=50, Green=50, Red=50
        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        self.widget._on_frame_ready(frame)

        # Unmodified camera image is in current_qimage (no heavy computation in event loop)
        raw_qimg = self.widget.current_qimage
        self.assertIsNotNone(raw_qimg)
        self.assertEqual(raw_qimg.width(), w)
        self.assertEqual(raw_qimg.height(), h)
        raw_pixel = raw_qimg.pixelColor(320, 240)
        self.assertEqual(raw_pixel.red(), 50)
        self.assertEqual(raw_pixel.green(), 50)
        self.assertEqual(raw_pixel.blue(), 50)

        # Image merge is performed in the camera viewport drawing function: final = cam*(1-mask) + green*mask
        display_img = self.widget.viewport.get_display_image()
        self.assertIsNotNone(display_img)
        self.assertEqual(display_img.width(), w)
        self.assertEqual(display_img.height(), h)

        # Inspect pixel at center (320, 240): mask=1.0 -> final is pure green (0, 255, 0)
        pixel_color = display_img.pixelColor(320, 240)
        self.assertEqual(pixel_color.red(), 0)
        self.assertEqual(pixel_color.green(), 255)
        self.assertEqual(pixel_color.blue(), 0)

    def test_grayscale_frame_overlay(self):
        h, w = 480, 640
        fake_mask = np.ones((h, w), dtype=np.float32)
        with self.widget._latent_mask_lock:
            self.widget.latest_latent_mask = fake_mask
        self.widget.latent_vision_enabled = True

        # Grayscale frame 2D
        gray_frame = np.full((h, w), 50, dtype=np.uint8)
        self.widget._on_frame_ready(gray_frame)

        display_img = self.widget.viewport.get_display_image()
        self.assertIsNotNone(display_img)
        pixel_color = display_img.pixelColor(320, 240)
        self.assertEqual(pixel_color.red(), 0)
        self.assertEqual(pixel_color.green(), 255)
        self.assertEqual(pixel_color.blue(), 0)

    def test_previous_mask_reused_when_model_slower_than_camera(self):
        h, w = 480, 640
        fake_mask = np.full((h, w), 0.5, dtype=np.float32)
        with self.widget._latent_mask_lock:
            self.widget.latest_latent_mask = fake_mask
        self.widget.latent_vision_enabled = True

        # Simulate 5 rapid camera frames arriving while model output stays the same
        for i in range(5):
            frame = np.full((h, w, 3), 40 + i, dtype=np.uint8)
            self.widget._on_frame_ready(frame)
            # final = cam * (1 - 0.5) + green * 0.5
            expected_rb = int(round((40.0 + i) * 0.5))
            expected_green = int(round((40.0 + i) * 0.5 + 127.5))
            display_img = self.widget.viewport.get_display_image()
            pixel_color = display_img.pixelColor(10, 10)
            self.assertAlmostEqual(pixel_color.red(), expected_rb, delta=1)
            self.assertAlmostEqual(pixel_color.green(), expected_green, delta=1)
            self.assertAlmostEqual(pixel_color.blue(), expected_rb, delta=1)

    def test_toggle_off_restores_clean_frame(self):
        h, w = 480, 640
        self.widget.latent_vision_cb.setChecked(True)
        fake_mask = np.ones((h, w), dtype=np.float32)
        with self.widget._latent_mask_lock:
            self.widget.latest_latent_mask = fake_mask

        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        self.widget._on_frame_ready(frame)

        # Now toggle off
        self.widget.latent_vision_cb.setChecked(False)
        self.widget._on_frame_ready(frame)
        self.assertIsNone(self.widget.latest_latent_mask)
        display_img = self.widget.viewport.get_display_image()
        pixel_color = display_img.pixelColor(320, 240)
        self.assertEqual(pixel_color.red(), 50)
        self.assertEqual(pixel_color.green(), 50)
        self.assertEqual(pixel_color.blue(), 50)


class TestMaskRenderWorker(unittest.TestCase):
    def test_mask_render_worker_thread(self):
        rendered_images = []

        def on_image(qimg):
            rendered_images.append(qimg)

        worker = MaskRenderWorker(on_image_ready=on_image)
        worker.start()

        try:
            # Submit a test mask
            h, w = 240, 320
            fake_mask = np.full((h, w), 0.75, dtype=np.float32)
            worker.submit_mask(fake_mask)

            # Wait for output
            for _ in range(50):
                if rendered_images:
                    break
                time.sleep(0.02)

            self.assertGreater(len(rendered_images), 0)
            img = rendered_images[-1]
            self.assertEqual(img.width(), w)
            self.assertEqual(img.height(), h)
            self.assertFalse(img.isNull())
        finally:
            worker.stop()
            worker.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()

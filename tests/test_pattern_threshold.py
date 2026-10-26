import os
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.chip_project import ChipLayer, ChipProject
from core.events import Event
from core.engine import StepperEngine
from projector import DummyProjector
from camera.camera_module import DummyCamera
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.workflow_panel import WorkflowPanelWidget
from PySide6.QtWidgets import QApplication


class TestPatternThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test pattern image with various RGB values:
        # Pixel (0, 0): [51, 0, 0]   -> Red channel > 50
        # Pixel (0, 1): [0, 51, 0]   -> Green channel > 50
        # Pixel (0, 2): [0, 0, 51]   -> Blue channel > 50
        # Pixel (1, 0): [50, 50, 50] -> None > 50
        # Pixel (1, 1): [10, 20, 30] -> None > 50
        # Pixel (1, 2): [0, 0, 0]    -> None > 50
        cls.test_img_path = str(Path(__file__).resolve().parent / "test_thresh_temp.png")
        img = np.zeros((2, 3, 3), dtype=np.uint8)
        # BGR format for cv2.imwrite:
        img[0, 0] = [0, 0, 51]      # RGB: 51, 0, 0
        img[0, 1] = [0, 51, 0]      # RGB: 0, 51, 0
        img[0, 2] = [51, 0, 0]      # RGB: 0, 0, 51
        img[1, 0] = [50, 50, 50]    # RGB: 50, 50, 50
        img[1, 1] = [30, 20, 10]    # RGB: 10, 20, 30
        img[1, 2] = [0, 0, 0]       # RGB: 0, 0, 0
        cv2.imwrite(cls.test_img_path, img)

        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    def test_default_threshold_value(self):
        layer = ChipLayer()
        self.assertEqual(layer.threshold, 50)

    def test_thresholding_logic_default_50(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)

        pattern = layer.get_pattern_image()
        self.assertIsNotNone(pattern)

        # Expected RGB values with threshold = 50:
        # [51, 0, 0] -> White [255, 255, 255]
        np.testing.assert_array_equal(pattern[0, 0], [255, 255, 255])
        # [0, 51, 0] -> White [255, 255, 255]
        np.testing.assert_array_equal(pattern[0, 1], [255, 255, 255])
        # [0, 0, 51] -> White [255, 255, 255]
        np.testing.assert_array_equal(pattern[0, 2], [255, 255, 255])
        # [50, 50, 50] -> Black [0, 0, 0]
        np.testing.assert_array_equal(pattern[1, 0], [0, 0, 0])
        # [10, 20, 30] -> Black [0, 0, 0]
        np.testing.assert_array_equal(pattern[1, 1], [0, 0, 0])
        # [0, 0, 0] -> Black [0, 0, 0]
        np.testing.assert_array_equal(pattern[1, 2], [0, 0, 0])

    def test_threshold_disabled_minus_one(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)
        layer.set_threshold(-1)

        pattern = layer.get_pattern_image()
        self.assertIsNotNone(pattern)

        # Should preserve raw BGR values
        np.testing.assert_array_equal(pattern[0, 0], [0, 0, 51])
        np.testing.assert_array_equal(pattern[0, 1], [0, 51, 0])
        np.testing.assert_array_equal(pattern[0, 2], [51, 0, 0])
        np.testing.assert_array_equal(pattern[1, 0], [50, 50, 50])
        np.testing.assert_array_equal(pattern[1, 1], [30, 20, 10])
        np.testing.assert_array_equal(pattern[1, 2], [0, 0, 0])

    def test_threshold_custom_value(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)

        # Threshold 25: 51 > 25 (white), 50 > 25 (white), 30 > 25 (white), [0,0,0] <= 25 (black)
        layer.set_threshold(25)
        pattern = layer.get_pattern_image()
        np.testing.assert_array_equal(pattern[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(pattern[1, 0], [255, 255, 255])
        np.testing.assert_array_equal(pattern[1, 1], [255, 255, 255])
        np.testing.assert_array_equal(pattern[1, 2], [0, 0, 0])

        # Threshold 51: none of 51, 50, 30, 0 is > 51 -> all black
        layer.set_threshold(51)
        pattern = layer.get_pattern_image()
        self.assertTrue(np.all(pattern == 0))

    def test_serialization(self):
        layer = ChipLayer(name="Thresh Layer", threshold=120)
        data = layer.to_disk()
        self.assertEqual(data["threshold"], 120)

        restored = ChipLayer.from_disk(data)
        self.assertEqual(restored.name, "Thresh Layer")
        self.assertEqual(restored.threshold, 120)

        # Default fallback when missing in serialized dict
        restored_default = ChipLayer.from_disk({"name": "Legacy Layer"})
        self.assertEqual(restored_default.threshold, 50)

    def test_ui_threshold_spinbox_and_interactions(self):
        stage = DummyStage()
        projector = DummyProjector(size=(1280, 720))
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        widget = WorkflowPanelWidget(engine, bridge)
        layer_subpanel = widget.layer_subpanel

        # Verify initial UI value matches default threshold 50
        self.assertEqual(layer_subpanel.spin_threshold.value(), 50)

        # Set pattern path
        engine.project.active_layer.set_pattern_path(self.test_img_path)
        engine.event_bus.emit(Event.PROJECT_CHANGED, engine.project)

        # Change threshold via UI spinbox
        layer_subpanel.spin_threshold.setValue(100)
        self.assertEqual(engine.project.active_layer.threshold, 100)

        # Disable threshold via UI spinbox
        layer_subpanel.spin_threshold.setValue(-1)
        self.assertEqual(engine.project.active_layer.threshold, -1)

        # Switch layers and ensure spinbox updates to new layer's threshold
        l2 = engine.project.add_layer("Layer 2")
        l2.threshold = 75
        engine.project.select_layer(1)
        self.assertEqual(layer_subpanel.spin_threshold.value(), 75)


if __name__ == "__main__":
    unittest.main()

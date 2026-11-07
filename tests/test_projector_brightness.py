import os
import unittest
from unittest.mock import MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import ColorMode, Event, EventBus, ProjectorImageSource
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.machine_control_panel.manual_control_tab import ManualControlTabWidget
from ui.widgets.machine_control_panel import MachineControlPanelWidget


class TestProjectorBrightness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_default_brightness(self):
        projector = DummyProjector()
        self.assertEqual(projector.brightness, 1.0)

    def test_set_brightness_clamping(self):
        projector = DummyProjector()
        projector.set_brightness(0.5)
        self.assertAlmostEqual(projector.brightness, 0.5)

        projector.set_brightness(-0.2)
        self.assertAlmostEqual(projector.brightness, 0.0)

        projector.set_brightness(1.5)
        self.assertAlmostEqual(projector.brightness, 1.0)

    def test_brightness_event_emitted(self):
        events = EventBus()
        projector = DummyProjector()
        projector.event_bus = events

        received = []
        events.add_listener(Event.PROJECTOR_BRIGHTNESS_CHANGED, lambda val, *_: received.append(val))

        projector.set_brightness(0.6)
        self.assertEqual(len(received), 1)
        self.assertAlmostEqual(received[0], 0.6)

    def test_brightness_multiplies_image(self):
        projector = DummyProjector(size=(100, 100))
        projector.set_color_mode(ColorMode.RED)
        projector.set_image_source(ProjectorImageSource.SOLID)

        # At default brightness 1.0, solid red image has 255 in red channel, 0 in blue & green
        img_full = projector._displayed_image_cache
        self.assertIsNotNone(img_full)
        self.assertEqual(img_full[50, 50, 2], 255)
        self.assertEqual(img_full[50, 50, 0], 0)
        self.assertEqual(img_full[50, 50, 1], 0)

        # Set brightness to 0.5 -> red channel should be ~128
        projector.set_brightness(0.5)
        img_half = projector._displayed_image_cache
        self.assertIsNotNone(img_half)
        self.assertAlmostEqual(int(img_half[50, 50, 2]), 128, delta=1)
        self.assertEqual(img_half[50, 50, 0], 0)
        self.assertEqual(img_half[50, 50, 1], 0)

        # Set brightness to 0.0 -> all black
        projector.set_brightness(0.0)
        img_dark = projector._displayed_image_cache
        self.assertIsNotNone(img_dark)
        self.assertEqual(int(img_dark[50, 50, 2]), 0)
        self.assertEqual(int(img_dark[50, 50, 0]), 0)
        self.assertEqual(int(img_dark[50, 50, 1]), 0)

    def test_manual_control_tab_brightness_widget(self):
        stage = DummyStage()
        projector = DummyProjector()
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        tab = ManualControlTabWidget(engine=engine, bridge=bridge)
        self.assertAlmostEqual(tab.spin_projector_brightness.value(), 1.0)
        self.assertAlmostEqual(tab.spin_projector_brightness.minimum(), 0.0)
        self.assertAlmostEqual(tab.spin_projector_brightness.maximum(), 1.0)

        # Changing UI spinbox updates projector brightness
        tab.spin_projector_brightness.setValue(0.7)
        self.assertAlmostEqual(projector.brightness, 0.7)

        # Changing projector brightness via set_brightness updates UI via sync
        tab._sync_brightness(0.3)
        self.assertAlmostEqual(tab.spin_projector_brightness.value(), 0.3)

        # Lock state disables and enables the spinbox
        tab.update_lock_state(is_busy=True)
        self.assertFalse(tab.spin_projector_brightness.isEnabled())

        tab.update_lock_state(is_busy=False)
        self.assertTrue(tab.spin_projector_brightness.isEnabled())

    def test_machine_control_panel_syncs_brightness(self):
        stage = DummyStage()
        projector = DummyProjector()
        projector.brightness = 0.8
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        panel = MachineControlPanelWidget(engine=engine, bridge=bridge)
        self.assertAlmostEqual(panel.manual_tab.spin_projector_brightness.value(), 0.8)

        # Emitting brightness changed signal through bridge updates manual tab
        projector.set_brightness(0.4)
        self.assertAlmostEqual(panel.manual_tab.spin_projector_brightness.value(), 0.4)


if __name__ == "__main__":
    unittest.main()

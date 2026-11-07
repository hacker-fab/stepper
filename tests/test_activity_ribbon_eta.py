import unittest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.activity_ribbon.activity_ribbon_widget import ActivityRibbonWidget


class TestActivityRibbonETA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.engine = StepperEngine(stage=self.stage, projector=self.projector, camera=self.camera)
        self.bridge = QtEngineBridge(self.engine)
        self.ribbon = ActivityRibbonWidget(self.engine, self.bridge)

    def test_format_eta(self):
        self.assertEqual(ActivityRibbonWidget.format_eta(None), "ETA: --:--")
        self.assertEqual(ActivityRibbonWidget.format_eta(-10), "ETA: --:--")
        self.assertEqual(ActivityRibbonWidget.format_eta(0), "ETA: 00:00")
        self.assertEqual(ActivityRibbonWidget.format_eta(45), "ETA: 00:45")
        self.assertEqual(ActivityRibbonWidget.format_eta(125), "ETA: 02:05")
        self.assertEqual(ActivityRibbonWidget.format_eta(3600), "ETA: 01:00:00")
        self.assertEqual(ActivityRibbonWidget.format_eta(3665), "ETA: 01:01:05")

    def test_initial_state(self):
        self.assertEqual(self.ribbon.eta_label.text(), "")
        self.assertFalse(self.ribbon._eta_timer.isActive())

    def test_operation_start_and_finish_lifecycle(self):
        self.ribbon._on_operation_started("TestOp")
        self.assertEqual(self.ribbon.eta_label.text(), "ETA: --:--")
        self.assertTrue(self.ribbon._eta_timer.isActive())

        self.ribbon._on_operation_finished("TestOp")
        self.assertEqual(self.ribbon.eta_label.text(), "")
        self.assertFalse(self.ribbon._eta_timer.isActive())

    def test_operation_abort_lifecycle(self):
        self.ribbon._on_operation_started("TestOp")
        self.assertTrue(self.ribbon._eta_timer.isActive())

        self.ribbon._on_operation_aborted("TestOp")
        self.assertEqual(self.ribbon.eta_label.text(), "")
        self.assertFalse(self.ribbon._eta_timer.isActive())

    def test_eta_calculation_with_advancing_progress(self):
        self.ribbon._on_operation_started("TestOp")

        # Simulate 10 seconds passing and reaching 50% progress
        t0 = 100.0
        self.ribbon._progress_history.clear()
        self.ribbon._progress_history.append((t0, 0.0))
        self.ribbon._progress_history.append((t0 + 10.0, 0.50))
        self.ribbon._current_progress = 0.50

        # Run ETA update
        # Mock time.monotonic to return t0 + 10.0
        with unittest.mock.patch("time.monotonic", return_value=t0 + 10.0):
            self.ribbon._update_eta()

        # dp = 0.5, dt = 10.0s -> speed = 0.05/s. rem = 0.5 -> ETA = 10s
        self.assertEqual(self.ribbon.eta_label.text(), "ETA: 00:10")

    def test_eta_updates_when_progress_freezes(self):
        self.ribbon._on_operation_started("TestOp")

        t0 = 100.0
        self.ribbon._progress_history.clear()
        self.ribbon._progress_history.append((t0, 0.0))
        self.ribbon._progress_history.append((t0 + 10.0, 0.50))
        self.ribbon._current_progress = 0.50

        # 10s later (t = t0 + 20.0), progress is still 0.50 (frozen)
        with unittest.mock.patch("time.monotonic", return_value=t0 + 20.0):
            self.ribbon._update_eta()

        # dp = 0.5, dt = 20.0s -> speed = 0.025/s. rem = 0.5 -> ETA = 20s
        self.assertEqual(self.ribbon.eta_label.text(), "ETA: 00:20")

        # Another 10s later (t = t0 + 30.0), still frozen at 0.50
        with unittest.mock.patch("time.monotonic", return_value=t0 + 30.0):
            self.ribbon._update_eta()

        # dp = 0.5, dt = 30.0s -> speed = 0.01667/s. rem = 0.5 -> ETA = 30s
        self.assertEqual(self.ribbon.eta_label.text(), "ETA: 00:30")

    def test_eta_stalled_when_frozen_longer_than_window(self):
        self.ribbon._on_operation_started("TestOp")

        t0 = 100.0
        # All samples in the past 60s are at 0.50
        self.ribbon._progress_history.clear()
        self.ribbon._progress_history.append((t0, 0.50))
        self.ribbon._progress_history.append((t0 + 30.0, 0.50))
        self.ribbon._current_progress = 0.50

        with unittest.mock.patch("time.monotonic", return_value=t0 + 60.0):
            self.ribbon._update_eta()

        # dp = 0.0 -> stalled
        self.assertEqual(self.ribbon.eta_label.text(), "ETA: --:--")


if __name__ == "__main__":
    unittest.main()


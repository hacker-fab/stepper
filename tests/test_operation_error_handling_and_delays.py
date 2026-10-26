import io
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    _app = QApplication(["test", "-platform", "offscreen"])

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import ColorMode, Event
from core.operation import ExecutionContext, Operation
from operations import AutofocusConfig, AutofocusOperation, OpticsCalibrationOperation
from stage_control.dummy_stage import DummyStage
from projector import DummyProjector
from ui.bridge import QtEngineBridge
from ui.widgets.activity_ribbon.activity_ribbon_widget import ActivityRibbonWidget


class TestOperationErrorHandlingAndDelays(unittest.TestCase):
    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.engine = StepperEngine(self.stage, self.projector, self.camera)
        self.bridge = QtEngineBridge(self.engine)
        self.ribbon = ActivityRibbonWidget(self.engine, self.bridge)

    def test_failed_operation_prints_error_and_updates_ui_without_overwriting(self):
        class BrokenOp(Operation):
            def __init__(self):
                super().__init__("BrokenOp")

            def execute(self, context, report_progress):
                return "Hardware communication failure"

        op = BrokenOp()
        failed_signals = []
        self.bridge.operation_failed.connect(lambda name, err: failed_signals.append((name, err)))

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.bridge.start_operation(op)
            self.bridge.threadpool.waitForDone()
            QApplication.processEvents()
            stdout_text = mock_stdout.getvalue()

        # Check explicit terminal output
        self.assertIn("[Operation Error]", stdout_text)
        self.assertIn("BrokenOp", stdout_text)
        self.assertIn("Hardware communication failure", stdout_text)

        # Check signal received
        self.assertEqual(len(failed_signals), 1)
        self.assertEqual(failed_signals[0], ("BrokenOp", "Hardware communication failure"))

        # Check UI status shows failure and is NOT overwritten with "Ready"
        self.assertEqual(self.ribbon.status_label.text(), "Failed (BrokenOp): Hardware communication failure")
        self.assertIn("#ef4444", self.ribbon.status_icon.styleSheet())  # Red color for error

    def test_aborted_operation_prints_abort_and_updates_ui(self):
        class AbortOp(Operation):
            def __init__(self):
                super().__init__("AbortOp")

            def execute(self, context, report_progress):
                return "Operation aborted"

        op = AbortOp()
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.engine.operations.abort_current()  # Should print no op running
            def run_sync(worker):
                self.engine.operations.abort_current()  # Abort before execution
                worker()

            self.engine.operations.start_operation(op, run_async_callback=run_sync)
            QApplication.processEvents()
            stdout_text = mock_stdout.getvalue()

        self.assertIn("[Operation Aborted]", stdout_text)
        self.assertIn("AbortOp", stdout_text)
        self.assertEqual(self.ribbon.status_label.text(), "Aborted: AbortOp")
        self.assertIn("#ef4444", self.ribbon.status_icon.styleSheet())

    def test_autofocus_delays_before_ar_detection_are_at_least_1_second(self):
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        proj = DummyProjector()
        cam = MagicMock()
        cam.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        delays_before_detection = []
        last_delay = [None]
        def track_delay(s):
            last_delay[0] = s

        ctx = ExecutionContext(
            stage=stage,
            projector=proj,
            camera=cam,
            delay_func=track_delay,
        )

        with patch("operations.autofocus.detect_ar_tags") as mock_detect:
            def on_detect(*args, **kwargs):
                delays_before_detection.append(last_delay[0])
                return 0, [], None
            mock_detect.side_effect = on_detect

            cfg = AutofocusConfig(enabled=True, min_detection_rate=0.9)
            op = AutofocusOperation(config=cfg)
            op.execute(ctx, lambda p, m: None)

        self.assertTrue(len(delays_before_detection) > 0)
        for d in delays_before_detection:
            self.assertGreaterEqual(d, 1.0, f"Encountered delay {d}s < 1.0s before AR detection in AutofocusOperation")

    def test_optics_calibration_delays_before_ar_detection_are_at_least_1_second(self):
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        proj = DummyProjector()
        cam = MagicMock()
        cam.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        delays_before_detection = []
        last_delay = [None]
        def track_delay(s):
            last_delay[0] = s

        ctx = ExecutionContext(
            stage=stage,
            projector=proj,
            camera=cam,
            delay_func=track_delay,
        )

        with patch("operations.optics_calibration.detect_ar_tags") as mock_detect:
            def on_detect(*args, **kwargs):
                delays_before_detection.append(last_delay[0])
                return 0, [], None
            mock_detect.side_effect = on_detect

            op = OpticsCalibrationOperation()
            op.execute(ctx, lambda p, m: None)

        self.assertTrue(len(delays_before_detection) > 0)
        for d in delays_before_detection:
            self.assertGreaterEqual(d, 1.0, f"Encountered delay {d}s < 1.0s before AR detection in OpticsCalibrationOperation")

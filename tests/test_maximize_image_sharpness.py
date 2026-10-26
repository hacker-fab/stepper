import os
import unittest
from unittest.mock import MagicMock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np

from core.operation import ExecutionContext
from operations import MaximizeImageSharpnessOperation
from operations.maximize_image_sharpness import SharpnessOptimizationResult

from stage_control.dummy_stage import DummyStage


class TestMaximizeImageSharpness(unittest.TestCase):
    def test_unimodal_peak_convergence(self):
        """Simulate a unimodal focus peak at Z = 53.2 within initial Z = 50.0 +- 15.0."""
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        peak_z = 53.2

        cam = MagicMock()

        def fake_camera_image() -> np.ndarray:
            curr_z = stage.get_position()[2]
            val = int(max(0.0, 255.0 - abs(curr_z - peak_z) * 15.0))
            img = np.zeros((20, 20, 3), dtype=np.uint8)
            img[:10, :, 2] = val  # Red channel edge
            return img

        cam.get_latest_frame.side_effect = fake_camera_image

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(
            z_range=15.0,
            threshold=0.5,
            max_iterations=10,
            settle_delay=0.0,
        )

        progress_calls = []
        err = op.execute(ctx, lambda p, m: progress_calls.append((p, m)))

        self.assertIsNone(err)
        self.assertIsNotNone(op.result)
        assert op.result is not None
        self.assertTrue(op.result.converged)
        self.assertAlmostEqual(op.result.best_z, peak_z, delta=0.5)
        # Stage must end up parked at best_z
        self.assertAlmostEqual(stage.get_position()[2], op.result.best_z, places=4)

    def test_point_caching_avoids_redundant_measurements(self):
        """Verify that previously evaluated endpoints are cached and not re-evaluated."""
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        moves = []
        orig_move_abs = stage.move_absolute

        def tracked_move(pos: dict[str, float]) -> bool:
            moves.append(round(pos["z"], 4))
            return orig_move_abs(pos)

        stage.move_absolute = tracked_move

        cam = MagicMock()

        def fake_cam():
            curr_z = stage.get_position()[2]
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            val = int(max(0.0, 255.0 - abs(curr_z - 50.0) * 10.0))
            img[:5, :, 2] = val
            return img


        cam.get_latest_frame.side_effect = fake_cam

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(
            z_range=10.0,
            threshold=0.5,
            max_iterations=5,
            settle_delay=0.0,
        )

        err = op.execute(ctx, lambda p, m: None)

        self.assertIsNone(err)
        self.assertIsNotNone(op.result)
        # Check that unique moves during search match the number of moves minus repositioning at end
        search_moves = moves[:-1]
        self.assertEqual(len(search_moves), len(set(search_moves)), "Duplicate Z moves detected during search")

    def test_explicit_tuple_range(self):
        """Verify passing (z_min, z_max) as z_range."""
        stage = DummyStage(initial_position=(0.0, 0.0, 100.0))
        cam = MagicMock()

        def fake_cam():
            curr_z = stage.get_position()[2]
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            score_factor = max(0, int(255 - abs(curr_z - 25.0) * 10))
            img[:5, :, 2] = score_factor
            return img

        cam.get_latest_frame.side_effect = fake_cam

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(
            z_range=(20.0, 30.0),
            threshold=0.5,
            max_iterations=8,
            settle_delay=0.0,
        )

        err = op.execute(ctx, lambda p, m: None)

        self.assertIsNone(err)
        self.assertIsNotNone(op.result)
        assert op.result is not None
        self.assertTrue(op.result.converged)
        self.assertAlmostEqual(op.result.best_z, 25.0, delta=0.5)

    def test_stage_boundary_limit_failure(self):
        """Verify handling when stage cannot move to target."""
        stage = DummyStage(initial_position=(0.0, 0.0, 10.0))
        warnings = []

        def failing_move(pos: dict[str, float]) -> bool:
            if pos["z"] < 0.0:
                return False
            return True

        stage.move_absolute = failing_move

        cam = MagicMock()
        cam.get_latest_frame.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            warning_callback=lambda msg: warnings.append(msg),
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(
            z_range=20.0,  # 10.0 - 20.0 = -10.0 -> fails
            threshold=0.5,
            settle_delay=0.0,
        )

        err = op.execute(ctx, lambda p, m: None)

        self.assertIsNotNone(err)
        self.assertGreater(len(warnings), 0)
        self.assertIn("boundary limit reached", warnings[0])

    def test_abort_handling(self):
        """Verify cooperative abort stops the routine immediately."""
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        cam = MagicMock()
        cam.get_latest_frame.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(settle_delay=0.0)
        op.abort()

        err = op.execute(ctx, lambda p, m: None)
        self.assertIsNotNone(err)
        self.assertIn("aborted", err)

    def test_golden_section_sample_efficiency(self):
        """Verify that after sampling x1 and x2 initially, each iteration only evaluates 1 new point."""
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        measured_zs = []
        orig_move_abs = stage.move_absolute

        def tracked_move(pos: dict[str, float]) -> bool:
            measured_zs.append(pos["z"])
            return orig_move_abs(pos)

        stage.move_absolute = tracked_move

        cam = MagicMock()
        cam.get_latest_frame.side_effect = lambda: np.full((10, 10, 3), 128, dtype=np.uint8)

        ctx = ExecutionContext(
            stage=stage,
            projector=MagicMock(),
            camera=cam,
            delay_func=lambda _: None,
        )

        op = MaximizeImageSharpnessOperation(
            z_range=10.0,
            threshold=0.5,
            max_iterations=4,
            settle_delay=0.0,
        )

        err = op.execute(ctx, lambda p, m: None)
        self.assertIsNone(err)
        assert op.result is not None

        # During search, we have 2 initial points + 1 point per iteration
        # Plus 1 final move returning to best_z
        search_evals = len(op.result.measured_points)
        self.assertEqual(search_evals, 2 + op.result.iterations_completed)

    def test_golden_section_asymmetric_peaks(self):
        """Verify Golden Section search converges to peaks located on both sides of the range."""
        for peak_z in (43.5, 56.5):
            stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
            cam = MagicMock()

            def fake_cam():
                curr_z = stage.get_position()[2]
                img = np.zeros((10, 10, 3), dtype=np.uint8)
                val = int(max(10.0, 255.0 - abs(curr_z - peak_z) * 20.0))
                img[:5, :, 2] = val
                return img

            cam.get_latest_frame.side_effect = fake_cam

            ctx = ExecutionContext(
                stage=stage,
                projector=MagicMock(),
                camera=cam,
                delay_func=lambda _: None,
            )

            op = MaximizeImageSharpnessOperation(
                z_range=10.0,
                threshold=0.5,
                max_iterations=10,
                settle_delay=0.0,
            )

            err = op.execute(ctx, lambda p, m: None)
            self.assertIsNone(err)
            assert op.result is not None
            self.assertTrue(op.result.converged)
            self.assertAlmostEqual(op.result.best_z, peak_z, delta=0.5)

    def test_machine_control_panel_sharpness_widget(self):
        """Verify MachineControlPanelWidget contains sharpness controls and triggers operation."""
        from PySide6.QtWidgets import QApplication, QScrollArea
        from camera.camera_module import DummyCamera
        from core.chip_project import ChipProject
        from core.engine import StepperEngine
        from core.events import EventBus
        from core.operation import OperationManager
        from projector import DummyProjector
        from ui.bridge import QtEngineBridge
        from ui.widgets.machine_control_panel import MachineControlPanelWidget

        app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

        events = EventBus()
        stage = DummyStage()
        cam = DummyCamera()
        proj = DummyProjector()
        engine = StepperEngine(stage=stage, projector=proj, camera=cam)
        bridge = QtEngineBridge(engine)

        panel = MachineControlPanelWidget(engine, bridge)


        # Verify scroll areas are used for tabs
        self.assertIsInstance(panel.tabs.widget(0), QScrollArea)
        self.assertIsInstance(panel.tabs.widget(1), QScrollArea)
        self.assertIsInstance(panel.tabs.widget(2), QScrollArea)

        # Verify sharpness controls exist
        self.assertTrue(hasattr(panel, "btn_maximize_sharpness"))
        self.assertTrue(hasattr(panel, "spin_sharp_z_range"))
        self.assertEqual(panel.spin_sharp_z_range.value(), 20.0)

        # Set custom Z range
        panel.spin_sharp_z_range.setValue(35.0)
        self.assertEqual(panel.spin_sharp_z_range.value(), 35.0)

        # Clicking the button starts MaximizeImageSharpnessOperation
        with unittest.mock.patch.object(bridge, "start_operation") as mock_start:
            panel.btn_maximize_sharpness.click()
            mock_start.assert_called_once()
            op = mock_start.call_args[0][0]
            self.assertIsInstance(op, MaximizeImageSharpnessOperation)
            self.assertEqual(op.z_range, 35.0)


if __name__ == "__main__":
    unittest.main()


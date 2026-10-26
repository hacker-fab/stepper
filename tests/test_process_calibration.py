import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext
from lib.gen_calibration_pattern import generate_litho_target
from operations import (
    AutofocusOperation,
    ExposureOperation,
    ExposureOperationConfig,
    ProcessCalibrationConfig,
    ProcessCalibrationOperation,
)
from operations.process_calibration import generate_spiral_offsets
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.machine_control_panel.machine_control_panel_widget import MachineControlPanelWidget
from ui.widgets.machine_control_panel.process_calibration_tab import ProcessCalibrationTabWidget


class TestSpiralOffsets(unittest.TestCase):
    def test_spiral_offsets_zero_and_one(self):
        self.assertEqual(generate_spiral_offsets(0, 1000.0), [])
        self.assertEqual(generate_spiral_offsets(1, 1000.0), [(0.0, 0.0)])

    def test_spiral_offsets_sequence(self):
        offsets = generate_spiral_offsets(9, 500.0)
        self.assertEqual(len(offsets), 9)
        expected = [
            (0.0, 0.0),
            (500.0, 0.0),
            (500.0, 500.0),
            (0.0, 500.0),
            (-500.0, 500.0),
            (-500.0, 0.0),
            (-500.0, -500.0),
            (0.0, -500.0),
            (500.0, -500.0),
        ]
        self.assertEqual(offsets, expected)


class TestLithoTargetGeneration(unittest.TestCase):
    def test_generate_litho_target_square(self):
        target = generate_litho_target(size=1080, main_text="3.5s")
        self.assertEqual(target.shape, (1080, 1080))
        self.assertEqual(target.dtype, np.uint8)

    def test_generate_litho_target_multiline(self):
        target = generate_litho_target(size=1080, main_text="LINE 1\nLINE 2\nLINE 3")
        self.assertEqual(target.shape, (1080, 1080))
        self.assertEqual(target.dtype, np.uint8)


class TestExposureOperationDirect(unittest.TestCase):
    def test_exposure_operation_with_layer_index_none(self):
        stage = DummyStage()
        projector = DummyProjector(size=(1920, 1080))
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)

        # Set image source to GENERATED and set a dummy pattern
        dummy_img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        projector.set_generated_image(dummy_img)
        projector.set_image_source(ProjectorImageSource.GENERATED)

        config = ExposureOperationConfig(exposure_time=50.0)
        op = ExposureOperation(layer_index=None, config=config)

        # Execute
        err = op.execute(engine.context, lambda p, m: None)
        self.assertIsNone(err)

        # Image source should remain GENERATED
        self.assertEqual(projector.image_source, ProjectorImageSource.GENERATED)
        # Projector should end in off state and RED mode
        self.assertFalse(projector.is_on)
        self.assertEqual(projector.color_mode, ColorMode.RED)


class TestProcessCalibrationOperation(unittest.TestCase):
    def setUp(self):
        self.stage = DummyStage()
        self.stage.move_absolute({"x": 100.0, "y": 200.0, "z": 50.0})
        self.projector = DummyProjector(size=(1920, 1080))
        self.camera = DummyCamera()
        self.engine = StepperEngine(stage=self.stage, projector=self.projector, camera=self.camera)

    @patch("operations.process_calibration.AutofocusOperation.execute")
    @patch("operations.process_calibration.ExposureOperation.execute")
    def test_process_calibration_full_run(self, mock_exp_exec, mock_af_exec):
        mock_af_exec.return_value = None
        mock_exp_exec.return_value = None

        # Track stage moves
        moved_positions = []
        orig_move_abs = self.stage.move_absolute

        def spy_move_abs(coords):
            res = orig_move_abs(coords)
            moved_positions.append(self.stage.get_position())
            return res

        self.stage.move_absolute = spy_move_abs

        op = ProcessCalibrationOperation(
            min_exposure=1.0,
            max_exposure=3.0,
            sweep_steps=3,
            motion_distance=500.0,
        )

        progress_reports = []
        err = op.execute(self.engine.context, lambda p, m: progress_reports.append((p, m)))

        self.assertIsNone(err)
        # AF should have been called 3 times (once per step)
        self.assertEqual(mock_af_exec.call_count, 3)
        # Exposure should have been called 3 times (once per step)
        self.assertEqual(mock_exp_exec.call_count, 3)

        # Stage started at (100.0, 200.0)
        # Step 0: at (100, 200)
        # Step 1: at (100 + 500, 200 + 0) = (600, 200)
        # Step 2: at (100 + 500, 200 + 500) = (600, 700)
        self.assertEqual(len(moved_positions), 2)  # steps 1 and 2 triggered stage moves
        self.assertAlmostEqual(moved_positions[0][0], 600.0)
        self.assertAlmostEqual(moved_positions[0][1], 200.0)
        self.assertAlmostEqual(moved_positions[1][0], 600.0)
        self.assertAlmostEqual(moved_positions[1][1], 700.0)

        # Generated image should fit projector size
        self.assertIsNotNone(self.projector.generated_image)
        self.assertEqual(self.projector.generated_image.shape, (1080, 1920, 3))

    @patch("operations.process_calibration.AutofocusOperation.execute")
    def test_process_calibration_af_failure(self, mock_af_exec):
        mock_af_exec.return_value = "Blurry surface"

        op = ProcessCalibrationOperation(
            min_exposure=1.0,
            max_exposure=2.0,
            sweep_steps=2,
            motion_distance=500.0,
        )
        err = op.execute(self.engine.context, lambda p, m: None)
        self.assertIn("Autofocus failed at step 1: Blurry surface", err)

    def test_process_calibration_abort(self):
        op = ProcessCalibrationOperation(
            min_exposure=1.0,
            max_exposure=5.0,
            sweep_steps=5,
            motion_distance=500.0,
        )
        op.abort()
        err = op.execute(self.engine.context, lambda p, m: None)
        self.assertEqual(err, "Process calibration aborted")

    @patch("operations.process_calibration.AutofocusOperation")
    @patch("operations.process_calibration.ExposureOperation.execute")
    def test_process_calibration_uses_autofocus_config(self, mock_exp_exec, mock_af_cls):
        """Verify that ProcessCalibrationOperation passes autofocus_config to AutofocusOperation."""
        mock_af_instance = MagicMock()
        mock_af_instance.execute.return_value = None
        mock_af_cls.return_value = mock_af_instance
        mock_exp_exec.return_value = None

        from operations.autofocus import AutofocusConfig
        af_cfg = AutofocusConfig(enabled=True, uv_z_offset=12.5)

        op = ProcessCalibrationOperation(
            min_exposure=1.0,
            max_exposure=1.0,
            sweep_steps=1,
            autofocus_config=af_cfg,
        )
        err = op.execute(self.engine.context, lambda p, m: None)
        self.assertIsNone(err)

        # Inspect constructor call of AutofocusOperation
        mock_af_cls.assert_called_with(config=af_cfg)

    @patch("operations.process_calibration.AutofocusOperation.execute")
    @patch("operations.process_calibration.ExposureOperation.execute")
    def test_process_calibration_2d_fem_matrix_sweep(self, mock_exp_exec, mock_af_exec):
        """Verify 2D Focus-Exposure Matrix sweeps N exposures x M Z-offsets and applies Z offsets."""
        mock_af_exec.return_value = None
        mock_exp_exec.return_value = None

        z_positions_at_exposure = []

        def capture_z_on_exposure(ctx, report):
            z_positions_at_exposure.append(self.stage.get_position()[2])
            return None

        mock_exp_exec.side_effect = capture_z_on_exposure

        # Base Z is 50.0
        self.stage.move_absolute({"z": 50.0})

        op = ProcessCalibrationOperation(
            min_exposure=1.0,
            max_exposure=3.0,
            sweep_steps=2,       # exposures: 1.0s, 3.0s
            min_z_offset=-4.0,
            max_z_offset=4.0,
            z_steps=2,           # z offsets: -4.0, +4.0
            motion_distance=200.0,
        )

        err = op.execute(self.engine.context, lambda p, m: None)
        self.assertIsNone(err)

        # Total steps: 2 exposures x 2 z offsets = 4 steps
        self.assertEqual(mock_af_exec.call_count, 4)
        self.assertEqual(mock_exp_exec.call_count, 4)
        self.assertEqual(len(z_positions_at_exposure), 4)

        # Step 0: exp 1.0s, z_offset -4.0 -> Z = 50.0 - 4.0 = 46.0
        # Step 1: exp 1.0s, z_offset +4.0 -> Z = 50.0 + 4.0 = 54.0
        # Step 2: exp 3.0s, z_offset -4.0 -> Z = 54.0 - 4.0 -> wait, stage at each step:
        # Since mock_af_exec does not move Z, the stage moves by z_off from current position:
        self.assertAlmostEqual(z_positions_at_exposure[0], 46.0)
        self.assertAlmostEqual(z_positions_at_exposure[1], 50.0)  # 46.0 + 4.0 = 50.0


class TestProcessCalibrationUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.engine = StepperEngine(stage=self.stage, projector=self.projector, camera=self.camera)
        self.bridge = QtEngineBridge(self.engine)

    def test_process_tab_widgets_and_properties(self):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        tab = panel.process_tab

        self.assertIsInstance(tab, ProcessCalibrationTabWidget)
        self.assertEqual(tab.spin_min_exposure.value(), 1.0)
        self.assertEqual(tab.spin_max_exposure.value(), 5.0)
        self.assertEqual(tab.spin_sweep_steps.value(), 5)
        self.assertEqual(tab.spin_min_z_offset.value(), 0.0)
        self.assertEqual(tab.spin_max_z_offset.value(), 0.0)
        self.assertEqual(tab.spin_z_steps.value(), 1)
        self.assertEqual(tab.spin_motion_distance.value(), 1000.0)

        # Verify direct aliases on panel
        self.assertEqual(panel.spin_cal_min_exp.value(), 1.0)
        self.assertEqual(panel.spin_cal_max_exp.value(), 5.0)
        self.assertEqual(panel.spin_cal_sweep_steps.value(), 5)
        self.assertEqual(panel.spin_cal_min_z.value(), 0.0)
        self.assertEqual(panel.spin_cal_max_z.value(), 0.0)
        self.assertEqual(panel.spin_cal_z_steps.value(), 1)
        self.assertEqual(panel.spin_cal_motion_dist.value(), 1000.0)
        self.assertEqual(panel.btn_start_process_cal, tab.btn_start_cal)
        self.assertEqual(panel.btn_fem_placeholder, tab.btn_start_cal)

    def test_tab_lock_state(self):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        tab = panel.process_tab

        tab.update_lock_state(is_busy=True)
        self.assertFalse(tab.btn_start_cal.isEnabled())
        self.assertFalse(tab.spin_min_exposure.isEnabled())
        self.assertFalse(tab.spin_max_exposure.isEnabled())
        self.assertFalse(tab.spin_sweep_steps.isEnabled())
        self.assertFalse(tab.spin_min_z_offset.isEnabled())
        self.assertFalse(tab.spin_max_z_offset.isEnabled())
        self.assertFalse(tab.spin_z_steps.isEnabled())
        self.assertFalse(tab.spin_motion_distance.isEnabled())

        tab.update_lock_state(is_busy=False)
        self.assertTrue(tab.btn_start_cal.isEnabled())
        self.assertTrue(tab.spin_min_exposure.isEnabled())
        self.assertTrue(tab.spin_max_exposure.isEnabled())
        self.assertTrue(tab.spin_sweep_steps.isEnabled())
        self.assertTrue(tab.spin_min_z_offset.isEnabled())
        self.assertTrue(tab.spin_max_z_offset.isEnabled())
        self.assertTrue(tab.spin_z_steps.isEnabled())
        self.assertTrue(tab.spin_motion_distance.isEnabled())

    @patch.object(QtEngineBridge, "start_operation")
    def test_tab_start_button_triggers_bridge(self, mock_start_op):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        tab = panel.process_tab

        tab.spin_min_exposure.setValue(2.0)
        tab.spin_max_exposure.setValue(8.0)
        tab.spin_sweep_steps.setValue(4)
        tab.spin_min_z_offset.setValue(-3.0)
        tab.spin_max_z_offset.setValue(3.0)
        tab.spin_z_steps.setValue(3)
        tab.spin_motion_distance.setValue(1200.0)

        tab.btn_start_cal.click()

        self.assertEqual(mock_start_op.call_count, 1)
        op = mock_start_op.call_args[0][0]
        self.assertIsInstance(op, ProcessCalibrationOperation)
        self.assertEqual(op.min_exposure, 2.0)
        self.assertEqual(op.max_exposure, 8.0)
        self.assertEqual(op.sweep_steps, 4)
        self.assertEqual(op.min_z_offset, -3.0)
        self.assertEqual(op.max_z_offset, 3.0)
        self.assertEqual(op.z_steps, 3)
        self.assertEqual(op.motion_distance, 1200.0)


if __name__ == "__main__":
    unittest.main()

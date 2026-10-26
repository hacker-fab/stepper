import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.chip_project import ChipProject
from core.engine import StepperEngine
from core.events import ColorMode, Event, EventBus, ProjectorImageSource
from core.operation import ExecutionContext, OperationManager
from lib.ar_tag import (
    compute_focus_score,
    detect_ar_tags,
    generate_ar_tag_grid,
    get_aruco_dict,
)
from operations import (
    AutofocusConfig,
    AutofocusOperation,
    OpticsCalibrationOperation,
)
from projector import DummyProjector, ProjectorController
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.projector import QtProjector
from ui.widgets.machine_control_panel import MachineControlPanelWidget


class DummyCamWithFrame:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def get_latest_frame(self):
        return self.frame


class TestArUcoLib(unittest.TestCase):
    def test_get_aruco_dict(self):
        d = get_aruco_dict(cv2.aruco.DICT_4X4_50)
        self.assertIsNotNone(d)

    def test_generate_ar_tag_grid(self):
        grid = generate_ar_tag_grid(2, canvas_size=(640, 480))
        self.assertEqual(grid.shape, (480, 640, 3))
        self.assertEqual(grid.dtype, np.uint8)
        self.assertTrue(np.any(grid[:, :, 0] > 0))
        self.assertTrue(np.any(grid[:, :, 1] > 0))
        self.assertTrue(np.any(grid[:, :, 2] > 0))

    def test_detect_ar_tags(self):
        # Generate clean BGR grid and detect
        grid = generate_ar_tag_grid(2, canvas_size=(800, 600))
        count, corners, ids = detect_ar_tags(grid)
        self.assertEqual(count, 4)
        self.assertIsNotNone(ids)
        self.assertEqual(len(ids), 4)

        # Empty/None detection
        count0, corners0, ids0 = detect_ar_tags(None)
        self.assertEqual(count0, 0)
        self.assertIsNone(ids0)

    def test_compute_focus_score(self):
        # None image returns 0
        self.assertEqual(compute_focus_score(None), 0.0)

        # Blank image has 0 variance
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(compute_focus_score(blank), 0.0)

        # Sharp edge has high focus score
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[:50, :, 0] = 255
        score = compute_focus_score(sharp)
        self.assertGreater(score, 100.0)


class TestProjectorGeneratedImageSource(unittest.TestCase):
    def test_generated_image_source_and_display(self):
        proj = DummyProjector()
        self.assertIsNone(proj.generated_image)

        # Set generated image and switch source
        dummy_pattern = np.full((100, 100, 3), 128, dtype=np.uint8)
        proj.set_generated_image(dummy_pattern)
        self.assertIs(proj.generated_image, dummy_pattern)

        proj.set_color_mode(ColorMode.RED)
        proj.set_image_source(ProjectorImageSource.GENERATED)
        self.assertIsNotNone(proj._displayed_image_cache)
        self.assertFalse(proj.is_on)

        # Turning on shows image
        proj.set_on(True)
        self.assertTrue(proj.is_on)
        self.assertIsNotNone(proj._displayed_image_cache)

        # Turning off keeps is_on False
        proj.set_on(False)
        self.assertFalse(proj.is_on)

    def test_projector_on_off_event_and_dlpc_sync(self):
        dlpc_mock = MagicMock()
        proj = DummyProjector(dlpc=dlpc_mock)
        events = EventBus()
        proj.event_bus = events

        on_off_events = []
        events.add_listener(Event.PROJECTOR_ON_OFF_CHANGED, lambda on, *_: on_off_events.append(on))

        # Default is off
        self.assertFalse(proj.is_on)

        # Turning on in Red mode
        proj.set_color_mode(ColorMode.RED)
        proj.set_on(True)
        self.assertTrue(proj.is_on)
        self.assertEqual(on_off_events, [True])
        dlpc_mock.set_illumination_enable.assert_called_with(0b001)

        # Turning on in UV mode
        proj.set_color_mode(ColorMode.UV)
        proj.set_on(True)
        dlpc_mock.set_illumination_enable.assert_called_with(0b100)

        # Turning off
        proj.set_on(False)
        self.assertFalse(proj.is_on)
        self.assertEqual(on_off_events, [True, True, False])
        dlpc_mock.set_illumination_enable.assert_called_with(0)

    def test_projector_on_off_does_not_recompute(self):
        proj = DummyProjector()
        dummy_pattern = np.full((100, 100, 3), 42, dtype=np.uint8)
        proj.set_generated_image(dummy_pattern)
        proj.set_image_source(ProjectorImageSource.GENERATED)
        self.assertIsNotNone(proj._displayed_image_cache)

        # Spy on _recompute_image
        with patch.object(proj, "_recompute_image", wraps=proj._recompute_image) as spy_recompute:
            proj.set_on(True)
            self.assertTrue(proj.is_on)
            # Recompute must NOT have been called
            spy_recompute.assert_not_called()

            proj.set_on(False)
            self.assertFalse(proj.is_on)
            spy_recompute.assert_not_called()

    def test_wait_for_display_handshake(self):
        proj = DummyProjector()
        # By default, base controller is ready
        self.assertTrue(proj.wait_for_display(timeout=0.1))

        # Clear event to simulate pending render
        proj.display_ready_event.clear()
        self.assertFalse(proj.display_ready_event.is_set())

        proj.display_ready_event.set()
        self.assertTrue(proj.wait_for_display(timeout=0.1))

    def test_projector_canvas_widget_paint_event(self):
        from PySide6.QtGui import QImage
        from ui.projector import ProjectorCanvasWidget
        app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

        canvas = ProjectorCanvasWidget()
        self.assertIsNone(canvas._qimage)

        # Rendering null image
        canvas.set_qimage(None)
        canvas.repaint()

        # Rendering valid QImage
        img_arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        qimg = QImage(img_arr.data, 100, 100, 300, QImage.Format_BGR888)
        canvas.set_qimage(qimg)
        self.assertIsNotNone(canvas._qimage)
        canvas.repaint()

    def test_qt_projector_canvas_integration(self):
        app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
        qp = QtProjector()
        self.assertIsNotNone(qp.canvas)

        # Set generated image and update
        dummy = np.full((50, 50, 3), 255, dtype=np.uint8)
        qp.set_generated_image(dummy)
        qp.set_color_mode(ColorMode.RED)
        qp.set_image_source(ProjectorImageSource.GENERATED)
        qp.set_on(True)

        # Process pending Qt signals/events
        app.processEvents()
        self.assertTrue(qp.display_ready_event.is_set())
        self.assertIsNotNone(qp.canvas._qimage)
        qp.close()


class TestIterativeAutofocus(unittest.TestCase):
    def test_autofocus_config(self):
        cfg = AutofocusConfig.from_dict({
            "enabled": True,
            "uv_z_offset": -12.5,
            "min_detection_rate": 0.80,
        })
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.uv_z_offset, -12.5)
        self.assertEqual(cfg.min_detection_rate, 0.80)

    def test_autofocus_applies_uv_z_offset(self):
        stage = DummyStage(initial_position=(0.0, 0.0, 50.0))
        proj = DummyProjector()

        # Dynamic frame that peaks at Z=50.0
        base_grid = generate_ar_tag_grid(2, canvas_size=(800, 600))

        def dynamic_frame():
            curr_z = stage.get_position()[2]
            dist = abs(curr_z - 50.0)
            blur_k = min(51, max(1, int(dist) * 2 + 1))
            val = max(0.05, 1.0 - dist / 550.0)
            blurred = cv2.GaussianBlur(base_grid, (blur_k, blur_k), 0)
            return (blurred * val).astype(np.uint8)

        cam = MagicMock()
        cam.get_latest_frame.side_effect = dynamic_frame

        ctx = ExecutionContext(
            stage=stage,
            projector=proj,
            camera=cam,
            delay_func=lambda _: None,
        )

        cfg = AutofocusConfig(enabled=True, uv_z_offset=15.0, min_detection_rate=0.85)
        op = AutofocusOperation(config=cfg)
        progress_calls = []
        err = op.execute(ctx, lambda p, m: progress_calls.append((p, m)))

        self.assertIsNone(err)
        # Optimal red was 50.0, so final Z after applying +15.0 offset must be 65.0
        final_z = stage.get_position()[2]
        self.assertAlmostEqual(final_z, 65.0, delta=0.5)

        # Projector must be restored to off and ACTIVE_LAYER
        self.assertFalse(proj.is_on)
        self.assertEqual(proj.image_source, ProjectorImageSource.ACTIVE_LAYER)
        self.assertIsNone(proj.generated_image)


class TestOpticsCalibration(unittest.TestCase):
    def test_optics_calibration_aborts_when_no_tags_detected(self):
        stage = DummyStage(initial_position=(0.0, 0.0, 100.0))
        proj = DummyProjector()
        cam = DummyCamWithFrame(np.zeros((600, 800, 3), dtype=np.uint8))  # Blank frame = 0 tags
        warnings = []
        ctx = ExecutionContext(
            stage=stage,
            projector=proj,
            camera=cam,
            warning_callback=lambda msg: warnings.append(msg),
            delay_func=lambda _: None,
        )

        op = OpticsCalibrationOperation(grid_sizes=[2])
        err = op.execute(ctx, lambda p, m: None)

        self.assertIsNotNone(err)
        self.assertIn("No ArUco tags detected", err)
        self.assertEqual(len(warnings), 1)
        self.assertIn("No ArUco tags detected", warnings[0])

        # Projector must be restored to off and ACTIVE_LAYER
        self.assertFalse(proj.is_on)
        self.assertEqual(proj.image_source, ProjectorImageSource.ACTIVE_LAYER)
        self.assertIsNone(proj.generated_image)

    def test_optics_calibration_success_computes_offset(self):
        stage = DummyStage(initial_position=(0.0, 0.0, 100.0))
        proj = DummyProjector()

        # Synthesize camera images based on Z and mode
        # Let optimal Red Z be 102.0 and optimal UV Z be 110.0 -> Offset = +8.0
        def dynamic_frame():
            curr_z = stage.get_position()[2]
            mode = proj.color_mode
            # Create ArUco tags image
            frame = generate_ar_tag_grid(2, canvas_size=(800, 600))
            # Add synthetic sharpness that peaks at 102 for RED and 110 for UV
            target_z = 102.0 if mode == ColorMode.RED else 110.0
            dist = abs(curr_z - target_z)
            blur_k = max(1, int(dist * 2) * 2 + 1)
            blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
            return blurred

        cam = MagicMock()
        cam.get_latest_frame.side_effect = dynamic_frame

        ctx = ExecutionContext(
            stage=stage,
            projector=proj,
            camera=cam,
            delay_func=lambda _: None,
        )

        op = OpticsCalibrationOperation(search_range=10.0, target_accuracy=2.0, grid_sizes=[2])
        progress_msgs = []
        err = op.execute(ctx, lambda p, m: progress_msgs.append(m))

        self.assertIsNone(err)
        self.assertIsNotNone(op.red_best_z)
        self.assertIsNotNone(op.uv_best_z)
        self.assertIsNotNone(op.uv_z_offset)
        self.assertAlmostEqual(op.red_best_z, 102.0, delta=2.1)
        self.assertAlmostEqual(op.uv_best_z, 110.0, delta=2.1)
        self.assertAlmostEqual(op.uv_z_offset, 8.0, delta=2.1)

        # Projector must be restored to off and ACTIVE_LAYER
        self.assertFalse(proj.is_on)
        self.assertEqual(proj.image_source, ProjectorImageSource.ACTIVE_LAYER)
        self.assertIsNone(proj.generated_image)


class TestMachineControlPanelTabs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def test_panel_tabs_and_groupboxes(self):
        stage = DummyStage()
        projector = DummyProjector()
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        engine.autofocus_config = AutofocusConfig(enabled=True, uv_z_offset=7.5)
        bridge = QtEngineBridge(engine)

        panel = MachineControlPanelWidget(engine, bridge)

        # Check tab widget
        self.assertEqual(panel.tabs.count(), 4)
        self.assertEqual(panel.tabs.tabText(0), "Manual Control")
        self.assertEqual(panel.tabs.tabText(1), "Optics Calibration")
        self.assertEqual(panel.tabs.tabText(2), "Process Calibration")
        self.assertEqual(panel.tabs.tabText(3), "ML Data Collection")

        # Check separate Autofocus and Alignment buttons and controls
        self.assertTrue(hasattr(panel, "btn_autofocus"))
        self.assertTrue(hasattr(panel, "btn_align"))
        self.assertTrue(hasattr(panel, "spin_uv_offset"))
        self.assertEqual(panel.spin_uv_offset.value(), 7.5)

        # Changing spin_uv_offset updates engine autofocus_config
        panel.spin_uv_offset.setValue(12.0)
        self.assertEqual(engine.autofocus_config.uv_z_offset, 12.0)

        # Check optics calibration controls
        self.assertTrue(hasattr(panel, "btn_start_optics_cal"))
        self.assertTrue(hasattr(panel, "spin_cal_search_range"))
        self.assertTrue(hasattr(panel, "spin_cal_target_accuracy"))
        self.assertEqual(panel.spin_cal_search_range.value(), 500.0)
        self.assertEqual(panel.spin_cal_target_accuracy.value(), 0.5)
        self.assertTrue(hasattr(panel, "btn_apply_cal_offset"))
        self.assertFalse(panel.btn_apply_cal_offset.isEnabled())

        # Simulate completion of OpticsCalibrationOperation
        cal_op = OpticsCalibrationOperation()
        cal_op.red_best_z = 100.0
        cal_op.uv_best_z = 115.5
        cal_op.uv_z_offset = 15.5

        panel._on_operation_finished(cal_op, None)
        self.assertEqual(panel.lbl_cal_red_z.text(), "100.00 µm")
        self.assertEqual(panel.lbl_cal_uv_z.text(), "115.50 µm")
        self.assertEqual(panel.lbl_cal_offset_z.text(), "+15.50 µm")
        self.assertTrue(panel.btn_apply_cal_offset.isEnabled())
        # Automatically set in GUI panel and engine config after calibration finishes
        self.assertEqual(panel.spin_uv_offset.value(), 15.5)
        self.assertEqual(engine.autofocus_config.uv_z_offset, 15.5)

        # Click apply offset button again to verify idempotency
        panel.btn_apply_cal_offset.click()
        self.assertEqual(panel.spin_uv_offset.value(), 15.5)
        self.assertEqual(engine.autofocus_config.uv_z_offset, 15.5)


if __name__ == "__main__":
    unittest.main()

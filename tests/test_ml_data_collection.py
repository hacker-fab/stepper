import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext
from lib.ar_tag import detect_ar_tags
from lib.cross_marker import generate_cross_pattern, get_base_cross_vertices
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_marker_to_camera_space,
    transform_points_projector_to_camera,
    warp_projector_image_to_camera,
)
from operations.ml_data_collection import (
    MLDataCollectionConfig,
    MLDataCollectionOperation,
    PatternSource,
)
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.machine_control_panel.machine_control_panel_widget import MachineControlPanelWidget
from ui.widgets.machine_control_panel.ml_data_collection_tab import MLDataCollectionTabWidget


app = QApplication.instance() or QApplication([])


class TestCrossMarkerGeneration(unittest.TestCase):
    def test_base_cross_vertices(self):
        b = 10.0
        # Default thickness = 1.0
        verts = get_base_cross_vertices(b)
        self.assertEqual(verts.shape, (12, 2))
        # Top-most point is at y = -1.5b
        self.assertAlmostEqual(np.min(verts[:, 1]), -15.0)
        # Bottom-most point is at y = +1.5b
        self.assertAlmostEqual(np.max(verts[:, 1]), 15.0)
        # Left-most point is at x = -1.5b
        self.assertAlmostEqual(np.min(verts[:, 0]), -15.0)
        # Right-most (longer arm) is at x = +2.5b
        self.assertAlmostEqual(np.max(verts[:, 0]), 25.0)

        # Test with thickness = 0.2
        verts_thin = get_base_cross_vertices(b, thickness=0.2)
        # Bounding box must be exactly the same
        self.assertAlmostEqual(np.min(verts_thin[:, 1]), -15.0)
        self.assertAlmostEqual(np.max(verts_thin[:, 1]), 15.0)
        self.assertAlmostEqual(np.min(verts_thin[:, 0]), -15.0)
        self.assertAlmostEqual(np.max(verts_thin[:, 0]), 25.0)
        # Bar half-width h = 0.5 * 10 * 0.2 = 1.0
        self.assertAlmostEqual(verts_thin[0, 0], -1.0)
        self.assertAlmostEqual(verts_thin[1, 0], 1.0)
        self.assertAlmostEqual(verts_thin[3, 1], -1.0)
        self.assertAlmostEqual(verts_thin[4, 1], 1.0)

    def test_generate_cross_pattern(self):
        canvas_w, canvas_h = 800, 600
        canvas, markers = generate_cross_pattern(
            canvas_size=(canvas_w, canvas_h),
            target_scale_pct=8.0,
            scale_jitter_pct=2.0,
            marker_count=10,
            random_seed=42,
        )
        self.assertEqual(canvas.shape, (canvas_h, canvas_w, 3))
        self.assertEqual(len(markers), 10)

        longer_edge = max(canvas_w, canvas_h)
        for m in markers:
            self.assertGreaterEqual(m.scale_pct, 6.0 - 1e-5)
            self.assertLessEqual(m.scale_pct, 10.0 + 1e-5)
            self.assertGreaterEqual(m.thickness, 0.2 - 1e-5)
            self.assertLessEqual(m.thickness, 1.0 + 1e-5)
            self.assertIn("thickness", m.to_dict())
            # Center inside canvas
            self.assertGreater(m.center_proj[0], 0)
            self.assertLess(m.center_proj[0], canvas_w)
            self.assertGreater(m.center_proj[1], 0)
            self.assertLess(m.center_proj[1], canvas_h)
            # All vertices inside canvas
            for vx, vy in m.vertices_proj:
                self.assertGreaterEqual(vx, 0)
                self.assertLess(vx, canvas_w)
                self.assertGreaterEqual(vy, 0)
                self.assertLess(vy, canvas_h)

        # Verify no markers overlap (enclosing circles check)
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                m1 = markers[i]
                m2 = markers[j]
                b1 = m1.scale_px / 4.0
                b2 = m2.scale_px / 4.0
                r1 = 2.6 * b1
                r2 = 2.6 * b2
                dist = np.hypot(m1.center_proj[0] - m2.center_proj[0], m1.center_proj[1] - m2.center_proj[1])
                self.assertGreaterEqual(dist, r1 + r2)


class TestProjectorCameraTransform(unittest.TestCase):
    def test_calibration_grid_and_homography(self):
        proj_w, proj_h = 640, 480
        grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
            grid_n=2,
            canvas_size=(proj_w, proj_h),
        )
        self.assertEqual(grid_img.shape, (proj_h, proj_w, 3))
        self.assertEqual(len(tag_coords), 4)

        # Detect ArUco tags directly on generated pattern
        count, corners, ids = detect_ar_tags(grid_img)
        self.assertEqual(count, 4)

        # Estimate identity homography
        H, matched = compute_projector_to_camera_homography(corners, ids, tag_coords)
        self.assertIsNotNone(H)
        self.assertEqual(matched, 16)
        # Check that H is close to identity (since camera frame == projector frame in this test)
        norm_H = H / H[2, 2]
        np.testing.assert_allclose(norm_H, np.eye(3), atol=0.1)

    def test_transform_points_and_marker(self):
        # Create a homography representing translation + scaling
        # x_cam = 0.5 * x_proj + 50, y_cam = 0.5 * y_proj + 30
        H = np.array(
            [
                [0.5, 0.0, 50.0],
                [0.0, 0.5, 30.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        pts = np.array([[100.0, 200.0], [300.0, 400.0]])
        transformed = transform_points_projector_to_camera(pts, H)
        expected = np.array([[100.0, 130.0], [200.0, 230.0]])
        np.testing.assert_allclose(transformed, expected, atol=1e-5)

        # Warp image test
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 100:200] = 255
        warped = warp_projector_image_to_camera(img, H, (320, 240))
        self.assertEqual(warped.shape, (240, 320, 3))


class TestMLDataCollectionOperation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.stage = DummyStage()
        self.stage.current_pos = [1000.0, 2000.0, 100.0]
        self.projector = DummyProjector()
        self.camera = DummyCamera()

        # Generate a synthetic camera frame with 2x2 ArUco tags for calibration
        proj_w, proj_h = self.projector.projector_size()
        cal_img, _ = generate_calibration_tag_grid_with_coords(grid_n=2, canvas_size=(proj_w, proj_h))
        self.camera._active = True
        self.camera._latest_frame = cal_img.copy()

        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            delay_func=lambda s: None,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_ml_data_collection_run(self):
        config = MLDataCollectionConfig(
            total_patterns=2,
            pattern_gap=500.0,
            target_scale_pct=8.0,
            scale_jitter_pct=1.0,
            marker_count=5,
            exposure_time=0.1,
            stabilization_delay=0.0,
            save_directory=self.test_dir,
            grid_n=2,
        )

        op = MLDataCollectionOperation(config=config)

        # Mock sub-operations to avoid slow camera loops
        with patch.object(op, "_run_sub_op", wraps=op._run_sub_op) as mock_run:
            def side_effect_sub_op(sub_op, ctx, cb):
                if sub_op.name == "Autofocus":
                    sub_op.best_red_z = 100.0
                    return None
                elif sub_op.name in ("Exposure", "Layer Exposure"):
                    return None
                elif sub_op.name in ("Jog", "Move"):
                    return sub_op.execute(ctx, cb)
                return sub_op.execute(ctx, cb)

            mock_run.side_effect = side_effect_sub_op

            progress_reports = []
            err = op.execute(self.context, lambda p, m: progress_reports.append((p, m)))
            self.assertIsNone(err)

        # Verify output files
        pattern1_img = os.path.join(self.test_dir, "pattern_0001.png")
        pattern1_gt = os.path.join(self.test_dir, "pattern_0001_projected.png")
        pattern1_json = os.path.join(self.test_dir, "pattern_0001.json")
        pattern2_img = os.path.join(self.test_dir, "pattern_0002.png")
        pattern2_gt = os.path.join(self.test_dir, "pattern_0002_projected.png")
        pattern2_json = os.path.join(self.test_dir, "pattern_0002.json")
        master_json = os.path.join(self.test_dir, "dataset_metadata.json")

        self.assertTrue(os.path.exists(pattern1_img))
        self.assertTrue(os.path.exists(pattern1_gt))
        self.assertTrue(os.path.exists(pattern1_json))
        self.assertTrue(os.path.exists(pattern2_img))
        self.assertTrue(os.path.exists(pattern2_gt))
        self.assertTrue(os.path.exists(pattern2_json))
        self.assertTrue(os.path.exists(master_json))

        # Check JSON contents
        with open(pattern1_json, "r") as f:
            rec1 = json.load(f)
        self.assertEqual(rec1["step_index"], 1)
        self.assertEqual(rec1["image_filename"], "pattern_0001.png")
        self.assertEqual(rec1["projected_gt_filename"], "pattern_0001_projected.png")
        self.assertIn("exposure_time", rec1)
        self.assertAlmostEqual(rec1["exposure_time"], 0.1, places=2)
        self.assertIn("markers", rec1)
        self.assertEqual(len(rec1["markers"]), 5)
        for m in rec1["markers"]:
            self.assertIn("center_img", m)
            self.assertIn("rotation_img_deg", m)
            self.assertIn("scale_img_px", m)
            self.assertIn("thickness", m)
            self.assertGreaterEqual(m["thickness"], 0.2)
            self.assertLessEqual(m["thickness"], 1.0)

        with open(master_json, "r") as f:
            master = json.load(f)
        self.assertEqual(master["total_patterns_collected"], 2)
        self.assertIn("homography_matrix", master)
        self.assertIn("min_exposure", master["config"])
        self.assertIn("max_exposure", master["config"])
        self.assertIn("min_thickness", master["config"])
        self.assertIn("max_thickness", master["config"])

    def test_photo_capture_averages_30_frames(self):
        config = MLDataCollectionConfig(
            total_patterns=1,
            pattern_gap=500.0,
            target_scale_pct=8.0,
            scale_jitter_pct=1.0,
            marker_count=2,
            exposure_time=0.1,
            stabilization_delay=0.0,
            save_directory=self.test_dir,
            grid_n=2,
        )
        op = MLDataCollectionOperation(config=config)

        # First call is ArUco calibration frame (must have valid ArUco tags)
        cal_frame = self.camera._latest_frame.copy()
        h, w = cal_frame.shape[:2]

        frame_counter = [0]

        def dynamic_get_frame():
            idx = frame_counter[0]
            frame_counter[0] += 1
            if idx == 0:
                return cal_frame
            # Subsequent frames return constant value equal to idx (from 1 to 30)
            val = idx
            return np.full((h, w, 3), fill_value=val, dtype=np.uint8)

        self.camera.get_latest_frame = dynamic_get_frame

        with patch.object(op, "_run_sub_op", return_value=None):
            err = op.execute(self.context, lambda p, m: None)
            self.assertIsNone(err)

        # 1 call for calibration + 30 calls for pattern photo = 31 total calls
        self.assertEqual(frame_counter[0], 31)

        # Verify averaged image: mean of integers 1..30 is 15.5, which rounds to 16
        saved_img_path = os.path.join(self.test_dir, "pattern_0001.png")
        saved_img = cv2.imread(saved_img_path)
        self.assertIsNotNone(saved_img)
        expected_val = int(np.round(np.mean(list(range(1, 31)))))
        self.assertTrue(np.all(saved_img == expected_val))

    def test_image_folder_pattern_source(self):
        # Create a folder with two dummy image files of different dimensions
        img_folder = os.path.join(self.test_dir, "source_images")
        os.makedirs(img_folder, exist_ok=True)
        img1 = np.full((60, 80, 3), 100, dtype=np.uint8)
        img2 = np.full((90, 120, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(img_folder, "test_img1.png"), img1)
        cv2.imwrite(os.path.join(img_folder, "test_img2.jpg"), img2)

        config = MLDataCollectionConfig(
            pattern_source=PatternSource.IMAGE_FOLDER.value,
            image_folder=img_folder,
            total_patterns=4,
            pattern_gap=500.0,
            exposure_time=0.1,
            stabilization_delay=0.0,
            save_directory=self.test_dir,
            grid_n=2,
        )

        op = MLDataCollectionOperation(config=config)

        # Mock sub-operations
        with patch.object(op, "_run_sub_op", return_value=None):
            err = op.execute(self.context, lambda p, m: None)
            self.assertIsNone(err)

        # Check all 4 pattern image pairs and JSONs exist
        selected_sources = []
        for i in range(1, 5):
            img_path = os.path.join(self.test_dir, f"pattern_{i:04d}.png")
            gt_path = os.path.join(self.test_dir, f"pattern_{i:04d}_projected.png")
            json_path = os.path.join(self.test_dir, f"pattern_{i:04d}.json")

            self.assertTrue(os.path.exists(img_path))
            self.assertTrue(os.path.exists(gt_path))
            self.assertTrue(os.path.exists(json_path))

            with open(json_path, "r", encoding="utf-8") as f:
                rec = json.load(f)

            self.assertEqual(rec["pattern_source"], PatternSource.IMAGE_FOLDER.value)
            self.assertIn("source_image", rec)
            self.assertIn(rec["source_image"], ["test_img1.png", "test_img2.jpg"])
            selected_sources.append(rec["source_image"])

            # Marker fields must be omitted
            self.assertNotIn("markers", rec)
            self.assertNotIn("marker_count", rec)

        # Verify cycling without replacement:
        # First 2 patterns should contain both images without duplicates
        self.assertEqual(set(selected_sources[:2]), {"test_img1.png", "test_img2.jpg"})
        # Next 2 patterns should also contain both images without duplicates
        self.assertEqual(set(selected_sources[2:]), {"test_img1.png", "test_img2.jpg"})

        # Check master metadata
        master_json = os.path.join(self.test_dir, "dataset_metadata.json")
        with open(master_json, "r", encoding="utf-8") as f:
            master = json.load(f)
        self.assertEqual(master["total_patterns_collected"], 4)
        self.assertEqual(master["config"]["pattern_source"], PatternSource.IMAGE_FOLDER.value)
        self.assertEqual(master["config"]["image_folder"], img_folder)

    def test_image_folder_validation_errors(self):
        # 1. Non-existent directory
        bad_config = MLDataCollectionConfig(
            pattern_source=PatternSource.IMAGE_FOLDER.value,
            image_folder=os.path.join(self.test_dir, "does_not_exist"),
        )
        op = MLDataCollectionOperation(config=bad_config)
        err = op.execute(self.context, lambda p, m: None)
        self.assertIsNotNone(err)
        self.assertIn("does not exist", err)

        # 2. Empty directory
        empty_folder = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_folder, exist_ok=True)
        bad_config2 = MLDataCollectionConfig(
            pattern_source=PatternSource.IMAGE_FOLDER.value,
            image_folder=empty_folder,
        )
        op2 = MLDataCollectionOperation(config=bad_config2)
        err2 = op2.execute(self.context, lambda p, m: None)
        self.assertIsNotNone(err2)
        self.assertIn("No valid image files found", err2)


class TestMLDataCollectionTabWidget(unittest.TestCase):
    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.engine = StepperEngine(stage=self.stage, projector=self.projector, camera=self.camera)
        self.bridge = QtEngineBridge(engine=self.engine)

    def test_tab_widget_controls_and_locking(self):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        ml_tab = panel.ml_tab

        # Test defaults
        self.assertTrue(ml_tab.radio_src_marker.isChecked())
        self.assertFalse(ml_tab.radio_src_folder.isChecked())
        self.assertFalse(ml_tab.txt_image_folder.isEnabled())
        self.assertFalse(ml_tab.btn_browse_image_folder.isEnabled())
        self.assertTrue(ml_tab.marker_box.isEnabled())
        self.assertAlmostEqual(ml_tab.spin_target_scale.value(), 8.0)
        self.assertAlmostEqual(ml_tab.spin_scale_jitter.value(), 2.0)
        self.assertEqual(ml_tab.spin_marker_count.value(), 20)
        self.assertAlmostEqual(ml_tab.spin_min_thickness.value(), 0.2)
        self.assertAlmostEqual(ml_tab.spin_max_thickness.value(), 1.0)
        self.assertEqual(ml_tab.spin_total_patterns.value(), 10)
        self.assertAlmostEqual(ml_tab.spin_pattern_gap.value(), 1000.0)
        self.assertAlmostEqual(ml_tab.spin_min_exposure.value(), 8.0)
        self.assertAlmostEqual(ml_tab.spin_max_exposure.value(), 12.0)

        # Test lock state
        ml_tab.update_lock_state(is_busy=True)
        self.assertFalse(ml_tab.btn_start.isEnabled())
        self.assertFalse(ml_tab.spin_target_scale.isEnabled())
        self.assertFalse(ml_tab.spin_min_thickness.isEnabled())
        self.assertFalse(ml_tab.spin_max_thickness.isEnabled())
        self.assertFalse(ml_tab.spin_total_patterns.isEnabled())
        self.assertFalse(ml_tab.spin_min_exposure.isEnabled())
        self.assertFalse(ml_tab.spin_max_exposure.isEnabled())

        ml_tab.update_lock_state(is_busy=False)
        self.assertTrue(ml_tab.btn_start.isEnabled())
        self.assertTrue(ml_tab.spin_target_scale.isEnabled())
        self.assertTrue(ml_tab.spin_min_thickness.isEnabled())
        self.assertTrue(ml_tab.spin_max_thickness.isEnabled())
        self.assertTrue(ml_tab.spin_total_patterns.isEnabled())
        self.assertTrue(ml_tab.spin_min_exposure.isEnabled())
        self.assertTrue(ml_tab.spin_max_exposure.isEnabled())

        # Test start triggers operation with default marker config
        with patch.object(self.bridge, "start_operation") as mock_start:
            ml_tab.btn_start.click()
            mock_start.assert_called_once()
            op = mock_start.call_args[0][0]
            self.assertIsInstance(op, MLDataCollectionOperation)
            self.assertEqual(op.config.pattern_source, PatternSource.GENERATED_MARKER.value)
            self.assertEqual(op.config.total_patterns, 10)
            self.assertAlmostEqual(op.config.min_exposure, 8.0)
            self.assertAlmostEqual(op.config.max_exposure, 12.0)
            self.assertAlmostEqual(op.config.min_thickness, 0.2)
            self.assertAlmostEqual(op.config.max_thickness, 1.0)

        # Test finish updates status
        ml_tab.on_operation_finished(op, err=None)
        self.assertEqual(ml_tab.lbl_status.text(), "Status: Complete")

        ml_tab.on_operation_finished(op, err="Device timeout")
        self.assertEqual(ml_tab.lbl_status.text(), "Status: Failed - Device timeout")

    def test_tab_widget_image_folder_selection_and_validation(self):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        ml_tab = panel.ml_tab

        # Switch to image folder
        ml_tab.radio_src_folder.setChecked(True)
        self.assertTrue(ml_tab.radio_src_folder.isChecked())
        self.assertTrue(ml_tab.txt_image_folder.isEnabled())
        self.assertTrue(ml_tab.btn_browse_image_folder.isEnabled())
        self.assertFalse(ml_tab.marker_box.isEnabled())
        self.assertFalse(ml_tab.spin_target_scale.isEnabled())

        # Click start without folder -> should show error
        ml_tab.txt_image_folder.setText("")
        with patch.object(self.bridge, "start_operation") as mock_start:
            ml_tab.btn_start.click()
            mock_start.assert_not_called()
            self.assertIn("Please select a valid image folder", ml_tab.lbl_status.text())

        # Test lock state when folder mode is active
        ml_tab.update_lock_state(is_busy=True)
        self.assertFalse(ml_tab.txt_image_folder.isEnabled())
        self.assertFalse(ml_tab.btn_browse_image_folder.isEnabled())
        self.assertFalse(ml_tab.spin_target_scale.isEnabled())

        ml_tab.update_lock_state(is_busy=False)
        self.assertTrue(ml_tab.txt_image_folder.isEnabled())
        self.assertTrue(ml_tab.btn_browse_image_folder.isEnabled())
        self.assertFalse(ml_tab.spin_target_scale.isEnabled())  # still disabled because folder mode

        # Set valid folder and click start
        temp_dir = tempfile.mkdtemp()
        try:
            ml_tab.txt_image_folder.setText(temp_dir)
            with patch.object(self.bridge, "start_operation") as mock_start:
                ml_tab.btn_start.click()
                mock_start.assert_called_once()
                op = mock_start.call_args[0][0]
                self.assertIsInstance(op, MLDataCollectionOperation)
                self.assertEqual(op.config.pattern_source, PatternSource.IMAGE_FOLDER.value)
                self.assertEqual(op.config.image_folder, temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

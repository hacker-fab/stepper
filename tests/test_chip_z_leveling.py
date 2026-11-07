import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.operation import ExecutionContext
from core.chip_project import ChipProject, ChipLayer
from core.events import ColorMode, EventBus, ProjectorImageSource
from operations.chip_z_leveling import (
    ChipZLevelingOperation,
    ExposeMarkerAndRecordOperation,
    fit_plane,
    generate_marker_pattern,
)
from stage_control.calibration import StageCalibration, StageCalibrationPoint
from stage_control.dummy_stage import DummyStage


class TestChipZLeveling(unittest.TestCase):
    def setUp(self):
        # 2x2 base calibration grid
        self.points = [
            StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
            StageCalibrationPoint((0, 1), 1, (10000.0, 0.0), (10000.0, 0.0), 110.0),
            StageCalibrationPoint((1, 0), 2, (0.0, 10000.0), (0.0, 10000.0), 120.0),
            StageCalibrationPoint((1, 1), 3, (10000.0, 10000.0), (10000.0, 10000.0), 130.0),
        ]
        self.cal = StageCalibration(
            grid_shape=(2, 2),
            points=self.points,
            stage_type="dummy",
        )

    def test_fit_plane(self):
        # Test exact plane: Delta_Z = 0.002 * X - 0.001 * Y + 15.0
        coords = [
            (1000.0, 2000.0, 0.002 * 1000.0 - 0.001 * 2000.0 + 15.0),
            (5000.0, 2000.0, 0.002 * 5000.0 - 0.001 * 2000.0 + 15.0),
            (5000.0, 8000.0, 0.002 * 5000.0 - 0.001 * 8000.0 + 15.0),
            (1000.0, 8000.0, 0.002 * 1000.0 - 0.001 * 8000.0 + 15.0),
        ]
        a, b, c = fit_plane(coords)
        self.assertAlmostEqual(a, 0.002, places=6)
        self.assertAlmostEqual(b, -0.001, places=6)
        self.assertAlmostEqual(c, 15.0, places=4)

    def test_calibration_plane_offset(self):
        # Base z offset at (5000, 5000) should be 115.0
        base_z = self.cal.get_base_z_offset(5000.0, 5000.0)
        self.assertAlmostEqual(base_z, 115.0, places=2)
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 115.0, places=2)

        # Set chip Z plane: Delta_Z = 0.001 * X + 0.002 * Y + 5.0
        plane = (0.001, 0.002, 5.0)
        self.cal.set_chip_z_plane(plane)
        self.assertEqual(self.cal.get_chip_z_plane(), plane)

        # Expected offset at (5000, 5000): 0.001*5000 + 0.002*5000 + 5.0 = 5 + 10 + 5 = 20.0
        # Total z offset: 115.0 + 20.0 = 135.0
        self.assertAlmostEqual(self.cal.get_base_z_offset(5000.0, 5000.0), 115.0, places=2)
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 135.0, places=2)

        # Verify corr_to_raw includes this offset
        raw_coords = self.cal.corr_to_raw([5000.0, 5000.0, 10.0])
        self.assertAlmostEqual(raw_coords["z"], 145.0, places=2)

        # Clear chip plane
        self.cal.clear_chip_z_plane()
        self.assertIsNone(self.cal.get_chip_z_plane())
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 115.0, places=2)

    def test_chip_z_plane_not_saved_to_disk(self):
        self.cal.set_chip_z_plane((0.001, 0.002, 5.0))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.cal.save_to_json(temp_path)
            loaded = StageCalibration.load_from_json(temp_path)
            self.assertIsNone(loaded.get_chip_z_plane())
            self.assertAlmostEqual(loaded.get_z_offset(5000.0, 5000.0), 115.0, places=2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chip_z_leveling_requires_calibration(self):
        stage = DummyStage()
        stage.set_calibration(None)

        context = ExecutionContext(
            stage=stage,
            camera=MagicMock(),
            projector=MagicMock(),
            machine_config=None,
            delay_func=lambda d: None,
        )

        op = ChipZLevelingOperation(chip_size_mm=10.0)
        err = op.execute(context, lambda p, m: None)
        self.assertIsNotNone(err)
        self.assertIn("No StageCalibration loaded", err)

    def test_chip_z_leveling_operation_execution(self):
        stage = DummyStage()
        stage.set_calibration(self.cal)
        stage.move_absolute({"x": 5000.0, "y": 5000.0, "z": 0.0})

        mock_projector = MagicMock()
        mock_projector.projector_size.return_value = (100, 100)
        mock_camera = MagicMock()
        mock_camera.get_latest_frame.return_value = None

        context = ExecutionContext(
            stage=stage,
            camera=mock_camera,
            projector=mock_projector,
            machine_config=None,
            delay_func=lambda d: None,
        )

        op = ChipZLevelingOperation(chip_size_mm=4.0, stabilization_delay=0.0)

        # Mock sub-operation execution so autofocus returns fixed best_red_z values per point
        # Point 1: (3000, 3000) -> 5.0
        # Point 2: (7000, 3000) -> 9.0
        # Point 3: (7000, 7000) -> 13.0
        # Point 4: (3000, 7000) -> 9.0
        # This corresponds to: Delta_Z = 0.001 * X + 0.001 * Y - 1.0
        # (3000: 3 + 3 - 1 = 5; 7000, 3000: 7 + 3 - 1 = 9; etc.)
        call_count = 0
        expected_z = [5.0, 9.0, 13.0, 9.0]

        def mock_run_sub_op(sub_op, ctx, reporter):
            nonlocal call_count
            sub_op.best_red_z = expected_z[call_count]
            call_count += 1
            return None

        op._run_sub_op = mock_run_sub_op

        progress_reports = []
        err = op.execute(context, lambda p, m: progress_reports.append((p, m)))

        self.assertIsNone(err)
        self.assertEqual(call_count, 4)

        # Verify returned to original position
        self.assertEqual(list(stage.get_position()), [5000.0, 5000.0, 0.0])

        # Verify plane fitted and set
        plane = self.cal.get_chip_z_plane()
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane[0], 0.001, places=5)
        self.assertAlmostEqual(plane[1], 0.001, places=5)
        self.assertAlmostEqual(plane[2], -1.0, places=3)

    def test_generate_marker_pattern(self):
        img1 = generate_marker_pattern(1, (800, 600))
        self.assertEqual(img1.shape, (600, 800, 3))
        # Center should be white meeting point of triangles
        self.assertTrue(np.all(img1[300, 400] == 255))
        # Top-left and bottom-right quadrants should have white pixels (triangles)
        self.assertTrue(np.any(img1[:300, :400] == 255))
        self.assertTrue(np.any(img1[300:, 400:] == 255))
        # Top-right and bottom-left should have text
        self.assertTrue(np.any(img1[:300, 400:] == 255))
        self.assertTrue(np.any(img1[300:, :400] == 255))

        img2 = generate_marker_pattern(2, (800, 600))
        self.assertEqual(img2.shape, (600, 800, 3))
        self.assertTrue(np.all(img2[300, 400] == 255))

    def test_expose_marker_and_record_aborts_if_already_recorded(self):
        project = ChipProject()
        project.chip_progress.marker1_chip = (100.0, 200.0)

        mock_stage = DummyStage()
        mock_projector = MagicMock()
        mock_camera = MagicMock()
        context = ExecutionContext(
            stage=mock_stage,
            camera=mock_camera,
            projector=mock_projector,
            project=project,
            delay_func=lambda d: None,
        )

        op = ExposeMarkerAndRecordOperation(idx=1)
        err = op.execute(context, lambda p, m: None)
        self.assertIsNotNone(err)
        self.assertIn("already recorded", err.lower())

    def test_expose_marker_and_record_execution(self):
        project = ChipProject()
        stage = DummyStage()
        stage.move_absolute({"x": 1200.0, "y": 3400.0, "z": 50.0})

        mock_projector = MagicMock()
        mock_projector.projector_size.return_value = (1920, 1080)
        mock_camera = MagicMock()
        mock_camera._last_frame_size = (1920, 1080)
        mock_camera.get_primary_crosshair_offset_um.return_value = (0.0, 0.0)

        from core.machine_config import MachineConfig
        cfg = MachineConfig()
        cfg.optics.projector_pixel_size = 1.0
        cfg.stage_calibration.camera_pixel_size = 1.0

        bus = EventBus()
        context = ExecutionContext(
            stage=stage,
            camera=mock_camera,
            projector=mock_projector,
            project=project,
            machine_config=cfg,
            event_bus=bus,
            delay_func=lambda d: None,
        )

        with unittest.mock.patch("operations.chip_z_leveling.AutofocusOperation.execute", return_value=None):
            op = ExposeMarkerAndRecordOperation(idx=1, exposure_time=10.0)
            err = op.execute(context, lambda p, m: None)
            self.assertIsNone(err)

        self.assertIsNotNone(project.chip_progress.marker1_chip)
        self.assertAlmostEqual(project.chip_progress.marker1_chip[0], 1200.0, places=1)
        self.assertAlmostEqual(project.chip_progress.marker1_chip[1], 3400.0, places=1)
        self.assertEqual(len(project.exposure_history), 1)

    def test_project_wide_pattern_size_and_origin_dimension_negative_one(self):
        project = ChipProject()
        l1 = project.active_layer
        l2 = project.add_layer("Layer 2")

        # Changing size on one layer affects all layers and project settings
        l1.set_physical_size(1500.0, 2500.0)
        self.assertEqual(project.settings.width_um, 1500.0)
        self.assertEqual(project.settings.height_um, 2500.0)
        self.assertEqual(l2.width_um, 1500.0)
        self.assertEqual(l2.height_um, 2500.0)

        # Set fullscreen (-1)
        project.set_pattern_size(-1.0, -1.0)
        self.assertEqual(l1.width_um, -1.0)
        self.assertEqual(l2.width_um, -1.0)

        # Test resolve_layer_physical_size with -1 dimensions
        from lib.tiling import resolve_layer_physical_size
        w, h = resolve_layer_physical_size(
            (100, 200),
            -1.0,
            -1.0,
            projector_size=(1920, 1080),
            projector_pixel_size=1.5,
        )
        self.assertEqual(w, 1920 * 1.5)
        self.assertEqual(h, 1080 * 1.5)

        # Test adding pattern origin with size
        idx = project.chip_progress.add_pattern_origin((500.0, 600.0), (w, h))
        self.assertEqual(idx, 0)
        self.assertEqual(project.chip_progress.pattern_origins_chip, [(500.0, 600.0)])
        self.assertEqual(project.chip_progress.pattern_sizes_chip, [(w, h)])

    def test_quick_expose_single_fullscreen_tile_and_origin_creation(self):
        from operations.exposure import ExposureOperation
        from core.machine_config import MachineConfig
        import tempfile
        import cv2

        project = ChipProject()
        cfg = MachineConfig()
        cfg.optics.projector_pixel_size = 1.25
        cfg.stage_calibration.camera_pixel_size = 1.0
        # Simulated rotation between projector and stage
        cfg.optics.proj_to_cam_matrix = [[1.0, 0.08, 0.0], [-0.08, 1.0, 0.0], [0.0, 0.0, 1.0]]
        project.machine_config = cfg

        layer = project.active_layer
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            temp_path = f.name

        try:
            layer.set_pattern_path(temp_path)
            # Regular tiling with dimension = -1 under rotation keeps tiling behavior (> 1 tiles)
            tiles_reg, _ = layer.generate_tiles(force=True, single_fullscreen_tile=False)
            self.assertGreater(len(tiles_reg), 1)

            # Fullscreen single tile mode produces exactly 1 tile
            tiles_single, coords = layer.generate_tiles(force=True, single_fullscreen_tile=True)
            self.assertEqual(len(tiles_single), 1)
            self.assertEqual(coords, [(0.0, 0.0)])

            # Execute quick expose with record_origin=True and single_fullscreen_tile=True
            stage = DummyStage()
            stage.move_absolute({"x": 450.0, "y": 650.0, "z": 0.0})
            proj_mock = MagicMock()
            proj_mock.projector_size.return_value = (1920, 1080)
            cam_mock = MagicMock()
            cam_mock._last_frame_size = (1920, 1080)

            context = ExecutionContext(
                stage=stage,
                projector=proj_mock,
                camera=cam_mock,
                project=project,
                machine_config=cfg,
                delay_func=lambda d: None,
            )

            op = ExposureOperation(
                layer_index=0,
                exposure_time=10.0,
                record_origin=True,
                single_fullscreen_tile=True,
            )
            err = op.execute(context, lambda p, m: None)
            self.assertIsNone(err)

            # Verify new origin and calculated projector size recorded
            self.assertEqual(project.chip_progress.pattern_origins_chip, [(450.0, 650.0)])
            self.assertEqual(project.chip_progress.pattern_sizes_chip, [(1920 * 1.25, 1080 * 1.25)])
            self.assertEqual(len(project.exposure_history), 1)

            # Verify that regular tiling cache is restored after quick expose completes
            self.assertGreater(len(layer._tile_cache), 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_quick_expose_and_layer_expose_interchangeable_under_wafer_rotation(self):
        """Verifies that exposing Layer 1 with Quick Expose Here records pattern_transforms_chip,
        and subsequent Layer 2 exposure via LayerExposureOperation after wafer rotation aligns
        to the exact same physical chip orientation without 180 deg inversion."""
        from core.coordinate_system import (
            apply_homography_2d,
            build_projector_to_chip_matrix,
            compute_fullscreen_pattern_to_chip_matrix,
            compute_two_point_rigid_transform,
        )
        from lib.tiling import generate_projector_aligned_tiles

        # Calibrated H_proj_to_cam with 180 deg optical inversion (negative diagonal)
        H_proj_to_cam = np.array([
            [-1.0, 0.0, 1920.0],
            [0.0, -1.0, 1080.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        pw, ph = 1920, 1080
        cw, ch = 1920, 1080
        proj_px = 0.5
        cam_px = 0.5
        w_um, h_um = pw * proj_px, ph * proj_px

        # Asymmetric F-shaped pattern
        img = np.zeros((ph, pw, 3), dtype=np.uint8)
        img[200:800, 300:400] = 255
        img[200:300, 300:900] = 255
        img[450:550, 300:650] = 255

        # Layer 1: Wafer at 0 deg, exposed with Quick Expose Here (single_fullscreen_tile=True)
        m1_stage_0 = (0.0, 0.0)
        m2_stage_0 = (2000.0, 0.0)
        m1_chip, m2_chip = m1_stage_0, m2_stage_0
        T_corr_to_chip_0 = compute_two_point_rigid_transform(m1_stage_0, m2_stage_0, m1_chip, m2_chip)

        tiles_l1, coords_l1 = generate_projector_aligned_tiles(
            img, -1.0, -1.0, 0, 0,
            projector_size=(pw, ph),
            camera_size=(cw, ch),
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
            proj_to_cam_matrix=H_proj_to_cam,
            corr_to_chip_matrix=T_corr_to_chip_0,
            single_fullscreen_tile=True,
        )
        self.assertEqual(len(tiles_l1), 1)
        self.assertEqual(coords_l1, [(0.0, 0.0)])

        # Recorded pattern transform from Quick Expose Here
        M_pat_to_chip = compute_fullscreen_pattern_to_chip_matrix(
            pattern_size_um=(w_um, h_um),
            projector_size=(pw, ph),
            camera_size=(cw, ch),
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
            proj_to_cam_matrix=H_proj_to_cam,
            corr_to_chip_matrix=T_corr_to_chip_0,
        )

        # Layer 2: Wafer rotated by 90 deg, exposed with regular Expose Layer using M_pat_to_chip
        m1_stage_90 = (0.0, 0.0)
        m2_stage_90 = (0.0, 2000.0)
        T_corr_to_chip_90 = compute_two_point_rigid_transform(m1_stage_90, m2_stage_90, m1_chip, m2_chip)

        tiles_l2, coords_l2 = generate_projector_aligned_tiles(
            img, -1.0, -1.0, 0, 0,
            projector_size=(pw, ph),
            camera_size=(cw, ch),
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
            proj_to_cam_matrix=H_proj_to_cam,
            corr_to_chip_matrix=T_corr_to_chip_90,
            pattern_transform_chip=M_pat_to_chip,
            single_fullscreen_tile=False,
        )

        # Compute physical chip coordinates of illuminated pixels in Layer 1
        M_proj_to_chip_l1 = build_projector_to_chip_matrix(
            stage_xy_corr=(0.0, 0.0),
            projector_size=(pw, ph),
            camera_size=(cw, ch),
            projector_pixel_size=proj_px,
            camera_pixel_size=cam_px,
            proj_to_cam_matrix=H_proj_to_cam,
            corr_to_chip_matrix=T_corr_to_chip_0,
        )
        ys1, xs1 = np.where(tiles_l1[0][..., 0] > 0)
        chip_pts_l1 = apply_homography_2d(np.column_stack([xs1, ys1]), M_proj_to_chip_l1)

        # Compute physical chip coordinates of illuminated pixels across all tiles in Layer 2
        chip_pts_l2_list = []
        T_chip_to_corr_90 = np.linalg.inv(T_corr_to_chip_90)
        for tile_img, (rel_cx, rel_cy) in zip(tiles_l2, coords_l2):
            ys2, xs2 = np.where(tile_img[..., 0] > 0)
            if len(xs2) == 0:
                continue
            stage_corr = apply_homography_2d((rel_cx, rel_cy), T_chip_to_corr_90)
            M_proj_to_chip_tile = build_projector_to_chip_matrix(
                stage_xy_corr=(float(stage_corr[0]), float(stage_corr[1])),
                projector_size=(pw, ph),
                camera_size=(cw, ch),
                projector_pixel_size=proj_px,
                camera_pixel_size=cam_px,
                proj_to_cam_matrix=H_proj_to_cam,
                corr_to_chip_matrix=T_corr_to_chip_90,
            )
            pts = apply_homography_2d(np.column_stack([xs2, ys2]), M_proj_to_chip_tile)
            chip_pts_l2_list.append(pts)

        chip_pts_l2 = np.vstack(chip_pts_l2_list)

        # Center of mass on the physical chip must match between Layer 1 (Quick Expose) and Layer 2 (Expose Layer)
        com1 = np.mean(chip_pts_l1, axis=0)
        com2 = np.mean(chip_pts_l2, axis=0)
        np.testing.assert_allclose(com1, com2, atol=1.0)


if __name__ == "__main__":
    unittest.main()

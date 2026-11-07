import json
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from lib.stage_calibration_reconstruction.pipeline import (
    ReconstructionDiagnostics,
    ReconstructionPipeline,
)
from operations.stage_calibration import DEFAULT_ARUCO_DICT
from tests.test_stage_calibration import generate_etched_ar_tag


class TestStageCalibrationComplementary(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.primary_dir = os.path.join(self.tmp_dir, "primary_images")
        self.comp_dir = os.path.join(self.tmp_dir, "comp_images")
        os.makedirs(self.primary_dir, exist_ok=True)
        os.makedirs(self.comp_dir, exist_ok=True)

        # Create a Stage 1 calibration JSON with 2x2 grid (markers 0, 1, 2, 3)
        self.stage1_path = os.path.join(self.tmp_dir, "stage1.json")
        stage1_data = {
            "version": "1.0",
            "stage_type": "dummy",
            "grid_shape": [2, 2],
            "step_distance": 1000.0,
            "calibration_range": 2000.0,
            "points": [
                {"grid_index": [0, 0], "marker_id": 0, "raw_xy": [0.0, 0.0], "z_focus": 50.0},
                {"grid_index": [0, 1], "marker_id": 1, "raw_xy": [1000.0, 0.0], "z_focus": 50.0},
                {"grid_index": [1, 0], "marker_id": 2, "raw_xy": [0.0, 1000.0], "z_focus": 50.0},
                {"grid_index": [1, 1], "marker_id": 3, "raw_xy": [1000.0, 1000.0], "z_focus": 50.0},
            ],
        }
        with open(self.stage1_path, "w", encoding="utf-8") as f:
            json.dump(stage1_data, f, indent=2)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _create_synthetic_image(self, marker_defs, canvas_size=(600, 800)):
        """marker_defs: list of (marker_id, center_x, center_y, tag_size)"""
        h, w = canvas_size
        img = np.full((h, w, 3), 220, dtype=np.uint8)
        for mid, cx, cy, sz in marker_defs:
            tag = generate_etched_ar_tag(mid, sz, dictionary=DEFAULT_ARUCO_DICT)
            th, tw = tag.shape[:2]
            x0 = int(round(cx - tw / 2.0))
            y0 = int(round(cy - th / 2.0))
            if 0 <= y0 and y0 + th <= h and 0 <= x0 and x0 + tw <= w:
                img[y0 : y0 + th, x0 : x0 + tw] = np.stack([tag] * 3, axis=-1)
        return img

    def test_reconstruction_with_and_without_complementary_dir(self):
        # 1. Create primary scan images (markers 0 and 1 in photo 1; markers 2 and 3 in photo 2)
        # Photo 1: Marker 0 at (250, 300), Marker 1 at (550, 300) -> distance dx = 300 px
        p1 = self._create_synthetic_image([(0, 250, 300, 120), (1, 550, 300, 120)])
        cv2.imwrite(os.path.join(self.primary_dir, "scan_0000_r0_c0_x0.0_y0.0.png"), p1)

        # Photo 2: Marker 2 at (250, 300), Marker 3 at (550, 300) -> distance dx = 300 px
        p2 = self._create_synthetic_image([(2, 250, 300, 120), (3, 550, 300, 120)])
        cv2.imwrite(os.path.join(self.primary_dir, "scan_0001_r1_c0_x0.0_y1000.0.png"), p2)

        # Run pipeline WITHOUT complementary directory
        pipeline_single = ReconstructionPipeline(
            images_dir=self.primary_dir,
            stage1_json_path=self.stage1_path,
            camera_pixel_size=1.0,
        )
        cal1, diag1, err1 = pipeline_single.run()
        self.assertIsNone(err1)
        self.assertIsNotNone(cal1)
        self.assertEqual(diag1.primary_pairwise_constraints, 2)  # (0, 1) and (2, 3)
        self.assertEqual(diag1.complementary_pairwise_constraints, 0)
        self.assertEqual(diag1.total_pairwise_constraints, 2)

        # 2. Create complementary scan images rotated by ~90 degrees or with vertical marker pairs (0 and 2)
        # Comp Photo 1: Marker 0 at (400, 150), Marker 2 at (400, 450) -> distance dy = 300 px
        # Comp Photo 2: Marker 1 at (400, 150), Marker 3 at (400, 450) -> distance dy = 300 px
        c1 = self._create_synthetic_image([(0, 400, 150, 120), (2, 400, 450, 120)])
        cv2.imwrite(os.path.join(self.comp_dir, "comp_0000.png"), c1)

        c2 = self._create_synthetic_image([(1, 400, 150, 120), (3, 400, 450, 120)])
        cv2.imwrite(os.path.join(self.comp_dir, "comp_0001.png"), c2)

        # Run pipeline WITH complementary directory
        pipeline_dual = ReconstructionPipeline(
            images_dir=self.primary_dir,
            stage1_json_path=self.stage1_path,
            camera_pixel_size=1.0,
            complementary_images_dir=self.comp_dir,
        )
        cal2, diag2, err2 = pipeline_dual.run()
        self.assertIsNone(err2)
        self.assertIsNotNone(cal2)
        self.assertEqual(diag2.primary_pairwise_constraints, 2)
        self.assertEqual(diag2.complementary_pairwise_constraints, 2)  # (0, 2) and (1, 3)
        self.assertEqual(diag2.total_pairwise_constraints, 4)

        # Global cam_to_stage matrix should be computed strictly from primary scan
        np.testing.assert_allclose(cal1.cam_to_stage_matrix, cal2.cam_to_stage_matrix, atol=1e-5)

    def test_detect_markers_batch(self):
        from lib.stage_calibration_reconstruction.detector import detect_markers_batch

        img1 = self._create_synthetic_image([(0, 250, 300, 120), (1, 550, 300, 120)])
        img2 = self._create_synthetic_image([(2, 250, 300, 120)])
        img3 = self._create_synthetic_image([(3, 400, 300, 120)])

        prog_calls = []

        def prog_cb(completed, total):
            prog_calls.append((completed, total))

        results = detect_markers_batch([img1, img2, img3], progress_cb=prog_cb)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 2)  # img1 has 2 markers
        self.assertEqual(results[1][0], 1)  # img2 has 1 marker
        self.assertEqual(results[2][0], 1)  # img3 has 1 marker
        self.assertTrue(len(prog_calls) >= 3)
        self.assertEqual(prog_calls[-1], (3, 3))

    def test_undetected_tag_opposite_neighbor_averaging(self):
        from lib.stage_calibration_reconstruction.solver import (
            match_global_transform_and_assign_undetected,
        )

        grid_shape = (3, 3)
        nominal_positions = {
            r * 3 + c: (float(c * 1000.0), float(r * 1000.0))
            for r in range(3)
            for c in range(3)
        }

        # Case A: Center tag 4 is missing, opposite neighbors 3 & 5 (+-x) and 1 & 7 (+-y) are detected
        # Tag 3: (0, 1010), Tag 5: (2000, 1030) -> x midpoint = (1000, 1020)
        # Tag 1: (1005, 0), Tag 7: (1015, 2000) -> y midpoint = (1010, 1000)
        # Both pairs present: average of midpoints = (1005.0, 1010.0)
        detected_a = {
            0: (0.0, 0.0),
            1: (1005.0, 0.0),
            2: (2000.0, 0.0),
            3: (0.0, 1010.0),
            # 4 is undetected
            5: (2000.0, 1030.0),
            6: (0.0, 2000.0),
            7: (1015.0, 2000.0),
            8: (2000.0, 2000.0),
        }

        res_a, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_a,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_a[4][0], 1005.0, places=3)
        self.assertAlmostEqual(res_a[4][1], 1010.0, places=3)

        # Case B: Only horizontal opposite neighbors (3 and 5) detected for 4 (1 and 7 missing)
        detected_b = {
            0: (0.0, 0.0),
            2: (2000.0, 0.0),
            3: (0.0, 1010.0),
            5: (2000.0, 1030.0),
            6: (0.0, 2000.0),
            8: (2000.0, 2000.0),
        }
        res_b, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_b,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_b[4][0], 1000.0, places=3)
        self.assertAlmostEqual(res_b[4][1], 1020.0, places=3)

        # Case C: Only vertical opposite neighbors (1 and 7) detected for 4 (3 and 5 missing)
        detected_c = {
            0: (0.0, 0.0),
            1: (1005.0, 0.0),
            2: (2000.0, 0.0),
            6: (0.0, 2000.0),
            7: (1015.0, 2000.0),
            8: (2000.0, 2000.0),
        }
        res_c, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_c,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_c[4][0], 1010.0, places=3)
        self.assertAlmostEqual(res_c[4][1], 1000.0, places=3)

        # Case D: Corner tag 0 missing (has no opposite horizontal or vertical neighbors)
        # Should fallback to global rigid alignment
        detected_d = {
            m: (pos[0] + 50.0, pos[1] + 100.0)
            for m, pos in nominal_positions.items()
            if m != 0
        }
        res_d, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_d,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_d[0][0], 50.0, places=1)
        self.assertAlmostEqual(res_d[0][1], 100.0, places=1)

    def test_estimate_camera_to_stage_transform_orthogonal_independent_scale(self):
        from lib.stage_calibration_reconstruction.solver import estimate_camera_to_stage_transform

        # Simulate skewed stage observations where motor X and Y are non-orthogonal (e.g. shear of 0.05)
        # and have different scale: X motor is 1.0 um/px, Y motor is 1.1 um/px
        # Move 1: stage moves by (1000, 50) while camera image shifts by (-1000, 0)
        # Move 2: stage moves by (0, 1100) while camera image shifts by (0, -1000)
        obs = [
            (0, (0.0, 0.0), (0.0, 0.0)),
            (0, (1000.0, 50.0), (-1000.0, 0.0)),
            (1, (0.0, 0.0), (0.0, 0.0)),
            (1, (0.0, 1100.0), (0.0, -1000.0)),
        ]

        T = estimate_camera_to_stage_transform(obs, default_rotation=0.0)

        # 1. Rows must be strictly orthogonal (zero skew):
        dot_rows = np.dot(T[0], T[1])
        self.assertAlmostEqual(dot_rows, 0.0, places=5)

        # 2. Independent scales: X scale ~ 1.0, Y scale ~ 1.1
        sx = np.linalg.norm(T[0])
        sy = np.linalg.norm(T[1])
        self.assertAlmostEqual(sx, 1.0, delta=0.05)
        self.assertAlmostEqual(sy, 1.1, delta=0.05)
        self.assertGreater(sy, sx)

    def test_additional_stage_rotation_in_calibration(self):
        from stage_control.calibration import StageCalibration, StageCalibrationPoint
        from core.machine_config import MachineConfig

        # Create a 3x3 grid centered around (5000, 5000)
        coords = [4000.0, 5000.0, 6000.0]
        pts = []
        mid = 0
        for r, y in enumerate(coords):
            for c, x in enumerate(coords):
                pts.append(StageCalibrationPoint((r, c), mid, (x, y), (x, y), 0.0))
                mid += 1

        # Base camera mounted at 90 deg relative to stage: (x_cam -> -y_stage, y_cam -> x_stage)
        # R_90 = [[0, -1], [1, 0]]
        base_T = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        cal = StageCalibration(
            grid_shape=(3, 3),
            points=pts,
            cam_to_stage_matrix=base_T,
            additional_rotation_deg=0.0,
        )

        # Baseline: target optical coordinate c = (5000, -5000)
        # raw_space_target = base_T @ [5000, -5000] = [5000, 5000]
        raw_x0, raw_y0 = cal.inverse_map_xy(5000.0, -5000.0)
        self.assertAlmostEqual(raw_x0, 5000.0, places=3)
        self.assertAlmostEqual(raw_y0, 5000.0, places=3)

        # Apply +5 deg additional rotation
        cal.set_additional_rotation(5.0)
        self.assertEqual(cal.additional_rotation_deg, 5.0)

        # Forward map and inverse map should remain mutually inverse
        target_cam = (5100.0, -4900.0)
        rx, ry = cal.inverse_map_xy(target_cam[0], target_cam[1])
        fwd_x, fwd_y = cal.forward_map_xy(rx, ry)
        self.assertAlmostEqual(fwd_x, target_cam[0], places=3)
        self.assertAlmostEqual(fwd_y, target_cam[1], places=3)

        # Verify MachineConfig serialization round-trip
        cfg = MachineConfig()
        cfg.stage_calibration.additional_stage_rotation_deg = 2.5
        cfg.stage_calibration.additional_stage_skew = 0.005
        cfg.stage_calibration.additional_stage_scale_x = 1.02
        cfg.stage_calibration.additional_stage_scale_y = 0.98
        d = cfg.to_dict()
        self.assertEqual(d["stage_calibration"]["additional_stage_rotation_deg"], 2.5)
        self.assertEqual(d["stage_calibration"]["additional_stage_skew"], 0.005)
        self.assertEqual(d["stage_calibration"]["additional_stage_scale_x"], 1.02)
        self.assertEqual(d["stage_calibration"]["additional_stage_scale_y"], 0.98)
        cfg_loaded = MachineConfig.from_dict(d)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_rotation_deg, 2.5)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_skew, 0.005)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_scale_x, 1.02)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_scale_y, 0.98)

    def test_additional_stage_skew_in_calibration(self):
        from stage_control.calibration import StageCalibration, StageCalibrationPoint

        # Create a 3x3 grid centered around (5000, 5000)
        coords = [4000.0, 5000.0, 6000.0]
        pts = []
        mid = 0
        for r, y in enumerate(coords):
            for c, x in enumerate(coords):
                pts.append(StageCalibrationPoint((r, c), mid, (x, y), (x, y), 0.0))
                mid += 1

        # Identity cam to stage matrix for straightforward skew verification:
        # dx_stage = dx_cam, dy_stage = dy_cam
        identity_T = np.eye(2, dtype=np.float64)
        cal = StageCalibration(
            grid_shape=(3, 3),
            points=pts,
            cam_to_stage_matrix=identity_T,
            additional_rotation_deg=0.0,
            additional_skew=0.0,
        )

        # Baseline: with 0 skew, (5000, 5000) -> (5000, 5000)
        rx_base, ry_base = cal.inverse_map_xy(5000.0, 5000.0)
        self.assertAlmostEqual(rx_base, 5000.0, places=3)
        self.assertAlmostEqual(ry_base, 5000.0, places=3)

        # Apply skew of +0.01 (dy/dx = 0.01)
        # S = [[1, 0], [0.01, 1]]
        cal.set_additional_skew(0.01)
        self.assertEqual(cal.additional_skew, 0.01)

        # Under skew=0.01, (5000, 5000) -> ry = 5000 + 0.01*5000 = 5050
        rx0, ry0 = cal.inverse_map_xy(5000.0, 5000.0)
        self.assertAlmostEqual(rx0, 5000.0, places=3)
        self.assertAlmostEqual(ry0, 5050.0, places=3)

        # Moving +100 um along X should produce +1 um shift along Y (100 * 0.01 = 1)
        rx1, ry1 = cal.inverse_map_xy(5100.0, 5000.0)
        # Expected: rx = 5100.0, ry = 5000.0 + 0.01 * 5100.0 = 5051.0
        self.assertAlmostEqual(rx1, 5100.0, places=3)
        self.assertAlmostEqual(ry1, 5051.0, places=3)

        # Verify dy/dx ratio: delta_y / delta_x == 0.01
        dy = ry1 - ry0
        dx = rx1 - rx0
        self.assertAlmostEqual(dy / dx, 0.01, places=5)

        # Forward map and inverse map should remain mutually inverse
        fwd_x, fwd_y = cal.forward_map_xy(rx1, ry1)
        self.assertAlmostEqual(fwd_x, 5100.0, places=3)
        self.assertAlmostEqual(fwd_y, 5000.0, places=3)

    def test_stage_motion_scale_in_calibration(self):
        from stage_control.calibration import StageCalibration, StageCalibrationPoint
        import tempfile
        import os

        # Create a 3x3 grid centered around (5000, 5000)
        coords = [4000.0, 5000.0, 6000.0]
        pts = []
        mid = 0
        for r, y in enumerate(coords):
            for c, x in enumerate(coords):
                pts.append(StageCalibrationPoint((r, c), mid, (x, y), (x, y), 0.0))
                mid += 1

        identity_T = np.eye(2, dtype=np.float64)
        cal = StageCalibration(
            grid_shape=(3, 3),
            points=pts,
            cam_to_stage_matrix=identity_T,
            additional_scale_x=1.0,
            additional_scale_y=1.0,
        )

        # Baseline: default scale is 1.0 (both X and Y)
        rx_base0, ry_base0 = cal.inverse_map_xy(5000.0, 5000.0)
        rx_base_x, ry_base_x = cal.inverse_map_xy(5100.0, 5000.0)
        rx_base_y, ry_base_y = cal.inverse_map_xy(5000.0, 5100.0)
        self.assertAlmostEqual(rx_base_x - rx_base0, 100.0, places=3)
        self.assertAlmostEqual(ry_base_y - ry_base0, 100.0, places=3)

        # Apply higher motion scale: scale_x = 1.05 (moves 5% more), scale_y = 1.10 (moves 10% more)
        cal.set_additional_scale(1.05, 1.10)
        self.assertEqual(cal.additional_scale_x, 1.05)
        self.assertEqual(cal.additional_scale_y, 1.10)

        # Move 100 um along X in commanded coords -> stage raw moves 105 um
        rx0, ry0 = cal.inverse_map_xy(5000.0, 5000.0)
        rx_x, ry_x = cal.inverse_map_xy(5100.0, 5000.0)
        dx_raw = rx_x - rx0
        dy_raw = ry_x - ry0
        self.assertAlmostEqual(dx_raw, 105.0, places=3)
        self.assertAlmostEqual(dy_raw, 0.0, places=3)

        # Move 100 um along Y in commanded coords -> stage raw moves 110 um
        rx_y, ry_y = cal.inverse_map_xy(5000.0, 5100.0)
        dx_raw_y = rx_y - rx0
        dy_raw_y = ry_y - ry0
        self.assertAlmostEqual(dx_raw_y, 0.0, places=3)
        self.assertAlmostEqual(dy_raw_y, 110.0, places=3)

        # Forward map and inverse map should remain mutually inverse
        fwd_x, fwd_y = cal.forward_map_xy(rx_x, ry_x)
        self.assertAlmostEqual(fwd_x, 5100.0, places=3)
        self.assertAlmostEqual(fwd_y, 5000.0, places=3)

        # Test JSON persistence roundtrip
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            cal.save_to_json(temp_path)
            loaded_cal = StageCalibration.load_from_json(temp_path)
            self.assertAlmostEqual(loaded_cal.additional_scale_x, 1.05, places=5)
            self.assertAlmostEqual(loaded_cal.additional_scale_y, 1.10, places=5)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chip_rotation_removed_while_preserving_camera_rotation(self):
        """Tests that a wafer mounting rotation (e.g. 2.6 deg) is completely un-rotated
        from the calibration grid output, so manual additional_stage_rotation can remain 0,
        while the physical ~90 deg camera-to-stage rotation is strictly preserved.
        """
        from lib.stage_calibration_reconstruction.solver import solve_global_positions
        from stage_control.calibration import StageCalibration, StageCalibrationPoint

        # 1. 3x3 nominal grid, 1000 um spacing
        nominal = {
            r * 3 + c: (float(c * 1000.0), float(r * 1000.0))
            for r in range(3)
            for c in range(3)
        }

        # 2. Wafer is taped on the stage with a 2.6 degree rotation
        theta_chip = np.radians(2.6)
        c_th, s_th = np.cos(theta_chip), np.sin(theta_chip)
        R_chip = np.array([[c_th, -s_th], [s_th, c_th]], dtype=np.float64)
        t_chip = np.array([500.0, -300.0], dtype=np.float64)

        # Observed stage positions of detected markers have this 2.6 deg tilt
        obs_stage_pos = {}
        for mid, p_nom in nominal.items():
            p_s = R_chip @ np.array(p_nom, dtype=np.float64) + t_chip
            obs_stage_pos[mid] = (float(p_s[0]), float(p_s[1]))

        # 3. Solve global calibrated positions
        cal_positions, R_solved = solve_global_positions(
            grid_shape=(3, 3),
            measurements=[],
            anchor_id=0,
            anchor_pos=(0.0, 0.0),
            nominal_positions=nominal,
            observed_positions=obs_stage_pos,
            step_distance=1000.0,
            default_rotation=theta_chip,
        )

        # 4. Verify that the 2.6 deg rotation is COMPLETELY REMOVED from cal_positions:
        # Every horizontal neighbor vector must be [1000.0, 0.0] (0 deg tilt)
        for r in range(3):
            for c in range(2):
                p_left = np.array(cal_positions[r * 3 + c])
                p_right = np.array(cal_positions[r * 3 + (c + 1)])
                dp = p_right - p_left
                self.assertAlmostEqual(dp[0], 1000.0, places=3)
                self.assertAlmostEqual(dp[1], 0.0, places=3)

        # Every vertical neighbor vector must be [0.0, 1000.0] (90 deg to horizontal)
        for r in range(2):
            for c in range(3):
                p_top = np.array(cal_positions[r * 3 + c])
                p_bottom = np.array(cal_positions[(r + 1) * 3 + c])
                dp = p_bottom - p_top
                self.assertAlmostEqual(dp[0], 0.0, places=3)
                self.assertAlmostEqual(dp[1], 1000.0, places=3)

        # 5. Build StageCalibration with cal_positions as corr_xy and a 90 deg camera-to-stage matrix
        cal_points = []
        for r in range(3):
            for c in range(3):
                mid = r * 3 + c
                cal_points.append(
                    StageCalibrationPoint(
                        grid_index=(r, c),
                        marker_id=mid,
                        raw_xy=nominal[mid],
                        corr_xy=cal_positions[mid],
                        z_focus=0.0,
                    )
                )

        # Camera is mounted at ~90 deg relative to stage: [x_s, y_s] = [-y_c, x_c]
        T_cam_to_stage = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)

        cal = StageCalibration(
            grid_shape=(3, 3),
            points=cal_points,
            cam_to_stage_matrix=T_cam_to_stage,
            additional_rotation_deg=0.0,  # Zero manual rotation!
        )

        # 6. Verify cam_to_stage_matrix retains the 90 deg rotation
        np.testing.assert_allclose(cal.cam_to_stage_matrix, T_cam_to_stage, atol=1e-6)

        # 7. Commanded motion along stage X axis (1000 um) requires raw stage movement exactly along X
        # with NO 2.6 deg deviation!
        # In optical space, an X stage movement corresponds to [0, 1000] camera shift:
        # T_cam_to_stage @ [0, 1000] = [-1000, 0]
        # At target (1000, 1000) vs (2000, 1000) directly in stage corr space:
        raw_x0, raw_y0 = cal.inverse_map_xy(0.0, -1000.0)
        raw_x1, raw_y1 = cal.inverse_map_xy(0.0, -2000.0)
        # Delta raw:
        d_raw_x = raw_x1 - raw_x0
        d_raw_y = raw_y1 - raw_y0
        self.assertAlmostEqual(d_raw_x, 1000.0, places=3)
        self.assertAlmostEqual(d_raw_y, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

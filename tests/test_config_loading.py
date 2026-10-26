import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import toml
from camera import DummyCamera, get_camera
from operations import (
    AlignmentConfig,
    AlignmentOperation,
    AutofocusConfig,
    AutofocusOperation,
)
from stage_control import DummyStage, get_stage_controller


class TestConfigLoading(unittest.TestCase):
    def test_camera_subsections(self):
        # 1. Dummy camera with [camera.dummy]
        cfg_dummy = {
            "type": "dummy",
            "dummy": {"width": 800, "height": 600},
        }
        cam_dummy = get_camera(cfg_dummy)
        self.assertIsInstance(cam_dummy, DummyCamera)
        self.assertEqual(cam_dummy.width, 800)
        self.assertEqual(cam_dummy.height, 600)

        # 2. Backward compatible flat dummy camera
        cfg_legacy = {"type": "dummy", "width": 1024, "height": 768}
        cam_legacy = get_camera(cfg_legacy)
        self.assertEqual(cam_legacy.width, 1024)
        self.assertEqual(cam_legacy.height, 768)

        # 3. Webcam index in [camera.webcam]
        with patch("camera.webcam.Webcam") as mock_webcam:
            cfg_webcam = {
                "type": "webcam",
                "webcam": {"index": 2},
            }
            get_camera(cfg_webcam)
            mock_webcam.assert_called_once_with(2)

    def test_stage_subsections(self):
        # 1. Dummy stage with [stage.dummy]
        cfg_dummy = {
            "type": "dummy",
            "dummy": {"delay": 0.05, "speed": 500.0},
        }
        stage_dummy = get_stage_controller(cfg_dummy)
        self.assertIsInstance(stage_dummy, DummyStage)
        self.assertEqual(stage_dummy.delay, 0.05)
        self.assertEqual(stage_dummy.speed, 500.0)

        # 2. Backward compatible flat dummy stage
        cfg_legacy = {"type": "dummy", "delay": 0.03}
        stage_legacy = get_stage_controller(cfg_legacy)
        self.assertEqual(stage_legacy.delay, 0.03)

        # 3. GRBL stage port, baud-rate, homing from [stage.grbl]
        with patch("serial.Serial") as mock_serial, patch("stage_control.grbl_stage.GrblStage") as mock_grbl:
            mock_serial.return_value.name = "COM6"
            cfg_grbl = {
                "type": "grbl",
                "grbl": {
                    "port": "COM6",
                    "baud-rate": 115200,
                    "homing": True,
                },
            }
            get_stage_controller(cfg_grbl, tiling=True)
            mock_serial.assert_called_once_with("COM6", 115200)
            mock_grbl.assert_called_once_with(mock_serial.return_value, True, True)

        # 4. OMM stage port and baud-rate from [stage.omm] or [stage.oom]
        with patch("stage_control.omm_stage.OMMStage") as mock_omm:
            cfg_omm = {
                "type": "omm",
                "omm": {
                    "port": "COM3",
                    "baud-rate": 921600,
                    "z-max": 50.0,
                },
            }
            get_stage_controller(cfg_omm)
            mock_omm.assert_called_once_with(50.0)
            mock_omm.return_value.connect.assert_called_once_with("COM3", 921600)

        # 5. OMM fallback if spelled 'oom'
        with patch("stage_control.omm_stage.OMMStage") as mock_omm:
            cfg_oom = {
                "type": "omm",
                "oom": {
                    "port": "COM4",
                    "baud-rate": 921600,
                },
            }
            get_stage_controller(cfg_oom)
            mock_omm.return_value.connect.assert_called_once_with("COM4", 921600)

    def test_alignment_config_from_dict(self):
        d = {
            "enabled": True,
            "model_path": "custom/path.pt",
            "right_marker_x": 1900.0,
            "left_marker_x": 200.0,
            "top_marker_y": 300.0,
            "bottom_marker_y": 1000.0,
            "x_scale_factor": -1200.0,
            "y_scale_factor": 850.0,
        }
        cfg = AlignmentConfig.from_dict(d)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model_path, "custom/path.pt")
        self.assertEqual(cfg.right_marker_x, 1900.0)
        self.assertEqual(cfg.left_marker_x, 200.0)
        self.assertEqual(cfg.top_marker_y, 300.0)
        self.assertEqual(cfg.bottom_marker_y, 1000.0)
        self.assertEqual(cfg.x_scale_factor, -1200.0)
        self.assertEqual(cfg.y_scale_factor, 850.0)

    def test_autofocus_config_from_dict_and_operation(self):
        af_cfg = AutofocusConfig.from_dict({"enabled": False})
        self.assertFalse(af_cfg.enabled)

        op = AutofocusOperation(config=af_cfg)
        mock_progress = MagicMock()
        mock_ctx = MagicMock()
        res = op.execute(mock_ctx, mock_progress)
        self.assertEqual(res, "Autofocus disabled in config")
        mock_progress.assert_called_with(1.0, "Autofocus disabled in config")

    def test_default_toml_parsing(self):
        default_toml_path = Path(__file__).resolve().parent.parent / "default.toml"
        with open(default_toml_path, "r") as f:
            cfg = toml.load(f)

        self.assertIn("camera", cfg)
        self.assertIn("stage", cfg)
        self.assertIn("projector", cfg)
        self.assertIn("tiling", cfg)
        self.assertIn("autofocus", cfg)
        self.assertIn("alignment", cfg)

        # Check camera section structure
        self.assertEqual(cfg["camera"]["type"], "webcam")
        self.assertEqual(cfg["camera"]["dummy"]["width"], 640)
        self.assertEqual(cfg["camera"]["dummy"]["height"], 480)
        self.assertEqual(cfg["camera"]["webcam"]["index"], 0)
        self.assertEqual(cfg["camera"]["pylon"]["index"], 0)

        # Check dummy camera from default.toml
        cam = get_camera({**cfg["camera"], "type": "dummy"})
        self.assertIsInstance(cam, DummyCamera)
        self.assertEqual(cam.width, 640)
        self.assertEqual(cam.height, 480)

        # Check stage section structure
        self.assertEqual(cfg["stage"]["type"], "omm")
        self.assertIn("grbl", cfg["stage"])
        self.assertEqual(cfg["stage"]["grbl"]["baud-rate"], 115200)

        # Check alignment section structure
        align_cfg = AlignmentConfig.from_dict(cfg["alignment"])
        self.assertFalse(align_cfg.enabled)
        self.assertEqual(align_cfg.model_path, "ckpts/best.pt")
        self.assertEqual(align_cfg.right_marker_x, 1820.0)

        # Check autofocus section
        af_cfg = AutofocusConfig.from_dict(cfg["autofocus"])
        self.assertTrue(af_cfg.enabled)

        # Check tiling section
        self.assertFalse(cfg["tiling"]["enabled"])

    def test_execution_context_does_not_contain_configs(self):
        from core.operation import ExecutionContext
        mock_stage = MagicMock()
        mock_proj = MagicMock()
        mock_cam = MagicMock()
        ctx = ExecutionContext(stage=mock_stage, projector=mock_proj, camera=mock_cam)
        self.assertFalse(hasattr(ctx, "autofocus_config"))
        self.assertFalse(hasattr(ctx, "alignment_config"))

    def test_operations_receive_configs_via_constructor(self):
        af_cfg = AutofocusConfig(enabled=False)
        af_op = AutofocusOperation(config=af_cfg)
        self.assertFalse(af_op.config.enabled)

        align_cfg = AlignmentConfig(enabled=False, model_path="custom.pt")
        align_op = AlignmentOperation(config=align_cfg)
        self.assertFalse(align_op.config.enabled)
        self.assertEqual(align_op.config.model_path, "custom.pt")


if __name__ == "__main__":
    unittest.main()

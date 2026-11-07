import os
import tempfile
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import Event
from core.machine_config import (
    AutofocusMachineConfig,
    AutofocusStageConfig,
    ExposureMachineConfig,
    MachineConfig,
    OpticsMachineConfig,
    StageCalibrationMachineConfig,
)
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.machine_control_panel.machine_config_tab import MachineConfigTabWidget


class TestMachineConfigDataclass(unittest.TestCase):
    def test_default_values(self):
        cfg = MachineConfig()
        self.assertEqual(cfg.optics.uv_z_offset, 0.0)
        self.assertEqual(cfg.optics.settle_delay, 0.1)
        self.assertEqual(len(cfg.autofocus.stages), 2)
        self.assertEqual(cfg.autofocus.stages[0].grid_n, 2)
        self.assertEqual(cfg.autofocus.stages[1].grid_n, 4)
        self.assertEqual(cfg.exposure.default_exposure_time_ms, 8000.0)
        self.assertEqual(cfg.stage_calibration.calibration_file_path, "stage_calibration.json")
        self.assertEqual(cfg.stage_calibration.camera_pixel_size, 1.0)
        self.assertEqual(cfg.stage_calibration.additional_stage_scale_x, 1.0)
        self.assertEqual(cfg.stage_calibration.additional_stage_scale_y, 1.0)

    def test_serialization_and_deserialization(self):
        cfg = MachineConfig(
            optics=OpticsMachineConfig(
                uv_z_offset=12.5,
                settle_delay=0.2,
            ),
            autofocus=AutofocusMachineConfig(
                stages=[
                    AutofocusStageConfig(grid_n=2, sweep_range=300.0, threshold=4.0),
                    AutofocusStageConfig(grid_n=4, sweep_range=50.0, threshold=1.0),
                    AutofocusStageConfig(grid_n=8, sweep_range=5.0, threshold=0.1),
                ]
            ),
            exposure=ExposureMachineConfig(default_exposure_time_ms=6500.0),
            stage_calibration=StageCalibrationMachineConfig(
                calibration_file_path="custom_cal.json",
                camera_pixel_size=0.85,
            ),
        )

        d = cfg.to_dict()
        loaded = MachineConfig.from_dict(d)

        self.assertEqual(loaded.optics.uv_z_offset, 12.5)
        self.assertEqual(loaded.optics.settle_delay, 0.2)
        self.assertEqual(len(loaded.autofocus.stages), 3)
        self.assertEqual(loaded.autofocus.stages[2].grid_n, 8)
        self.assertEqual(loaded.exposure.default_exposure_time_ms, 6500.0)
        self.assertEqual(loaded.stage_calibration.camera_pixel_size, 0.85)

    def test_save_and_load_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_machine_config.toml")
            cfg = MachineConfig(
                optics=OpticsMachineConfig(uv_z_offset=15.0, settle_delay=0.15),
                autofocus=AutofocusMachineConfig(
                    stages=[
                        AutofocusStageConfig(grid_n=3, sweep_range=100.0, threshold=2.0),
                    ]
                ),
                exposure=ExposureMachineConfig(default_exposure_time_ms=4500.0),
            )
            cfg.save(path)

            loaded = MachineConfig.load(path)
            self.assertEqual(loaded.optics.uv_z_offset, 15.0)
            self.assertEqual(loaded.optics.settle_delay, 0.15)
            self.assertEqual(len(loaded.autofocus.stages), 1)
            self.assertEqual(loaded.autofocus.stages[0].grid_n, 3)
            self.assertEqual(loaded.exposure.default_exposure_time_ms, 4500.0)

    def test_load_or_default_fallback(self):
        cfg = MachineConfig.load_or_default("non_existent_file_path.toml")
        self.assertEqual(cfg.exposure.default_exposure_time_ms, 8000.0)


class TestMachineConfigUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.config = MachineConfig(
            optics=OpticsMachineConfig(uv_z_offset=8.5, settle_delay=0.25),
            autofocus=AutofocusMachineConfig(
                stages=[
                    AutofocusStageConfig(grid_n=2, sweep_range=250.0, threshold=5.0),
                    AutofocusStageConfig(grid_n=4, sweep_range=10.0, threshold=0.5),
                ]
            ),
            exposure=ExposureMachineConfig(default_exposure_time_ms=7500.0),
        )
        self.engine = StepperEngine(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            machine_config=self.config,
        )
        self.bridge = QtEngineBridge(self.engine)
        self.tab = MachineConfigTabWidget(self.engine, self.bridge)

    def test_ui_initialization_from_config(self):
        self.assertEqual(self.tab.spin_uv_offset.value(), 8.5)
        self.assertEqual(self.tab.spin_settle_delay.value(), 0.25)
        self.assertEqual(self.tab.table_af_stages.rowCount(), 2)
        self.assertEqual(self.tab.spin_default_exposure.value(), 7500.0)
        self.assertEqual(self.tab.spin_camera_pixel_size.value(), 1.0)
        self.assertEqual(self.tab.spin_stage_scale_x.value(), 1.0)
        self.assertEqual(self.tab.spin_stage_scale_y.value(), 1.0)

    def test_ui_change_updates_engine_config(self):
        self.tab.spin_uv_offset.setValue(14.2)
        self.assertEqual(self.engine.machine_config.optics.uv_z_offset, 14.2)

        self.tab.spin_settle_delay.setValue(0.5)
        self.assertEqual(self.engine.machine_config.optics.settle_delay, 0.5)

        self.tab.spin_default_exposure.setValue(9200.0)
        self.assertEqual(self.engine.machine_config.exposure.default_exposure_time_ms, 9200.0)

        self.tab.spin_camera_pixel_size.setValue(1.25)
        self.assertEqual(self.engine.machine_config.stage_calibration.camera_pixel_size, 1.25)

        self.tab.spin_stage_scale_x.setValue(1.045)
        self.tab.spin_stage_scale_y.setValue(0.965)
        self.assertEqual(self.engine.machine_config.stage_calibration.additional_stage_scale_x, 1.045)
        self.assertEqual(self.engine.machine_config.stage_calibration.additional_stage_scale_y, 0.965)

    def test_add_and_remove_af_stages(self):
        init_count = self.tab.table_af_stages.rowCount()
        self.tab.btn_add_stage.click()
        self.assertEqual(self.tab.table_af_stages.rowCount(), init_count + 1)
        self.assertEqual(len(self.engine.machine_config.autofocus.stages), init_count + 1)

        self.tab.btn_remove_stage.click()
        self.assertEqual(self.tab.table_af_stages.rowCount(), init_count)
        self.assertEqual(len(self.engine.machine_config.autofocus.stages), init_count)

    def test_lock_state(self):
        self.tab.update_lock_state(is_busy=True)
        self.assertFalse(self.tab.spin_uv_offset.isEnabled())
        self.assertFalse(self.tab.spin_settle_delay.isEnabled())
        self.assertFalse(self.tab.table_af_stages.isEnabled())
        self.assertFalse(self.tab.btn_add_stage.isEnabled())
        self.assertFalse(self.tab.btn_remove_stage.isEnabled())
        self.assertFalse(self.tab.spin_default_exposure.isEnabled())
        self.assertFalse(self.tab.btn_save_config.isEnabled())

        self.tab.update_lock_state(is_busy=False)
        self.assertTrue(self.tab.spin_uv_offset.isEnabled())
        self.assertTrue(self.tab.spin_settle_delay.isEnabled())
        self.assertTrue(self.tab.table_af_stages.isEnabled())
        self.assertTrue(self.tab.btn_add_stage.isEnabled())
        self.assertTrue(self.tab.btn_remove_stage.isEnabled())
        self.assertTrue(self.tab.spin_default_exposure.isEnabled())
        self.assertTrue(self.tab.btn_save_config.isEnabled())


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.chip_project import (
    ChipLayer,
    ChipProject,
    ExposureLog,
    LayerSettingsOverride,
    PatterningSettings,
)
from core.events import Event, EventBus, ShownImage
from core.operation import (
    ExecutionContext,
    Operation,
    OperationManager,
)
from operations import (
    AlignmentConfig,
    AlignmentOperation,
    ExposureOperation,
    JogOperation,
    TiledExposureOperation,
)
from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from projector import ProjectorController
from stage_control import StageController
from ui.bridge import QtEngineBridge


class TestChipProject(unittest.TestCase):
    def test_default_project_has_one_layer(self):
        project = ChipProject()
        self.assertEqual(len(project.layers), 1)
        self.assertEqual(project.active_layer_index, 0)
        self.assertEqual(project.active_layer.name, "Layer 1")
        self.assertEqual(project.settings.exposure_time, 8000.0)

    def test_cannot_remove_last_layer(self):
        project = ChipProject()
        self.assertEqual(len(project.layers), 1)
        with self.assertRaises(ValueError):
            project.remove_layer(0)
        self.assertEqual(len(project.layers), 1)

    def test_add_and_select_layers(self):
        events = EventBus()
        emitted = []
        events.add_listener(Event.ACTIVE_LAYER_CHANGED, lambda idx, *args: emitted.append(("layer", idx)))
        events.add_listener(Event.PROJECT_CHANGED, lambda *args: emitted.append(("project", None)))

        project = ChipProject(events=events)
        l2 = project.add_layer("Layer 2 - Gate")
        self.assertEqual(len(project.layers), 2)
        self.assertEqual(project.active_layer_index, 1)
        self.assertEqual(project.active_layer.name, "Layer 2 - Gate")
        self.assertIn(("layer", 1), emitted)
        self.assertIn(("project", None), emitted)

        emitted.clear()
        # Select first layer
        ok = project.select_layer(0)
        self.assertTrue(ok)
        self.assertEqual(project.active_layer.name, "Layer 1")
        self.assertIn(("layer", 0), emitted)

        # Remove layer 2
        ok = project.remove_layer(1)
        self.assertTrue(ok)
        self.assertEqual(len(project.layers), 1)
        self.assertEqual(project.active_layer_index, 0)

    def test_settings_overrides_resolution(self):
        project = ChipProject()
        project.settings.exposure_time = 12000.0
        project.settings.tiling_enabled = False

        # Layer 1 has no overrides -> inherits project defaults
        eff1 = project.active_layer.get_effective_settings(project.settings)
        self.assertEqual(eff1.exposure_time, 12000.0)
        self.assertFalse(eff1.tiling_enabled)

        # Add Layer 2 with overrides
        l2 = project.add_layer("Layer 2")
        l2.overrides.exposure_time = 4500.0
        l2.overrides.tiling_enabled = True

        eff2 = l2.get_effective_settings(project.settings)
        self.assertEqual(eff2.exposure_time, 4500.0)
        self.assertTrue(eff2.tiling_enabled)
        # Inherits other non-overridden settings
        self.assertEqual(eff2.pitch_x, project.settings.pitch_x)

    def test_serialization_round_trip(self):
        project = ChipProject(name="Test Chip")
        project.settings.exposure_time = 9500.0
        l1 = project.active_layer
        l1.pattern_path = "/path/to/mask1.png"
        l1.exposures.append(
            ExposureLog(
                time=datetime(2026, 9, 10, 12, 0, 0),
                path="/path/to/mask1.png",
                coords=(1.0, 2.0, 3.0),
                duration=9500.0,
                aborted=False,
            )
        )

        l2 = project.add_layer("Layer 2")
        l2.overrides.exposure_time = 3000.0

        disk_data = project.to_disk()
        restored = ChipProject.from_disk(disk_data)

        self.assertEqual(restored.name, "Test Chip")
        self.assertEqual(restored.settings.exposure_time, 9500.0)
        self.assertEqual(len(restored.layers), 2)
        self.assertEqual(restored.layers[0].pattern_path, "/path/to/mask1.png")
        self.assertEqual(len(restored.layers[0].exposures), 1)
        self.assertEqual(restored.layers[0].exposures[0].duration, 9500.0)
        self.assertEqual(restored.layers[1].overrides.exposure_time, 3000.0)

    def test_file_save_and_load(self):
        import tempfile
        project = ChipProject(name="Persistent Chip")
        project.settings.exposure_time = 5000.0
        project.add_layer("Layer 2")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            project.save(temp_path)
            loaded = ChipProject.load(temp_path)
            self.assertEqual(loaded.name, "Persistent Chip")
            self.assertEqual(loaded.settings.exposure_time, 5000.0)
            self.assertEqual(len(loaded.layers), 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_render_pattern_caching(self):
        from PIL import Image
        project = ChipProject()
        layer = project.active_layer
        img = Image.new("RGB", (50, 50), "blue")
        layer.set_pattern_image(img)

        rendered1 = layer.render_pattern(project.settings, (100, 100))
        self.assertIsNotNone(rendered1)
        self.assertEqual(rendered1.size, (100, 100))

        # Second call returns cached instance
        rendered2 = layer.render_pattern(project.settings, (100, 100))
        self.assertIs(rendered1, rendered2)

        # Modifying adjust invalidates cache
        layer.image_adjust = (5.0, 5.0, 0.0)
        layer.invalidate_render_cache()
        rendered3 = layer.render_pattern(project.settings, (100, 100))
        self.assertIsNot(rendered1, rendered3)


class MockStage(StageController):
    def __init__(self):
        super().__init__()
        self.pos = (0.0, 0.0, 0.0)
        self.moves = []

    def get_position(self):
        return self.pos

    def move_relative(self, microns):
        self.moves.append(("rel", microns))
        x = self.pos[0] + microns.get("x", 0)
        y = self.pos[1] + microns.get("y", 0)
        z = self.pos[2] + microns.get("z", 0)
        self.pos = (x, y, z)
        return True

    def move_absolute(self, microns):
        self.moves.append(("abs", microns))
        x = microns.get("x", self.pos[0])
        y = microns.get("y", self.pos[1])
        z = microns.get("z", self.pos[2])
        self.pos = (x, y, z)
        return True

    def home(self):
        self.moves.append(("home", {}))
        self.pos = (0.0, 0.0, 0.0)
        return True

    def has_homing(self):
        return True

    def get_bounds(self):
        return None


class MockProjector(ProjectorController):
    def __init__(self):
        super().__init__()
        self.shown = []
        self.cleared = False

    def size(self):
        return (1920, 1080)

    def show(self, img):
        self.shown.append(img)
        self.cleared = False

    def clear(self):
        self.cleared = True


class DummyOperation(Operation):
    def __init__(self, name: str = "Dummy"):
        super().__init__(name)

    def execute(self, context: ExecutionContext, report_progress):
        report_progress(0.5, "Running...")


class TestOperationManager(unittest.TestCase):
    def setUp(self):
        self.stage = MockStage()
        self.projector = MockProjector()
        self.camera = DummyCamera()
        self.events = EventBus()
        self.project = ChipProject(events=self.events)
        self.warnings = []
        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            project=self.project,
            event_bus=self.events,
            warning_callback=lambda msg: self.warnings.append(msg),
            delay_func=lambda _: None,
        )
        self.manager = OperationManager(self.context, self.events)

    def test_single_active_operation(self):
        op1 = DummyOperation("Op1")
        op2 = DummyOperation("Op2")

        called_workers = []

        def mock_run_async(worker):
            called_workers.append(worker)

        # Start op1
        ok = self.manager.start_operation(op1, run_async_callback=mock_run_async)
        self.assertTrue(ok)
        self.assertFalse(self.manager.can_start_operation())
        self.assertEqual(self.manager.current_operation, op1)

        # Attempt to start op2 while op1 is active
        ok2 = self.manager.start_operation(op2, run_async_callback=mock_run_async)
        self.assertFalse(ok2)
        self.assertTrue(any("another operation ('Op1') is currently running" in w for w in self.warnings))

        # Run op1's worker to completion
        called_workers[0]()

        # Now op1 has completed
        self.assertTrue(self.manager.can_start_operation())
        self.assertIsNone(self.manager.current_operation)

    def test_operation_abort(self):
        aborted_events = []
        self.events.add_listener(Event.OPERATION_ABORTED, lambda name: aborted_events.append(name))

        op = DummyOperation("AbortableOp")

        def mock_run_async(worker):
            self.manager.abort_current()
            worker()

        self.manager.start_operation(op, run_async_callback=mock_run_async)
        self.assertTrue(op.is_aborted)
        self.assertIn("AbortableOp", aborted_events)

    def test_alignment_operation(self):
        cfg = AlignmentConfig(
            enabled=True,
            model_path="",
            right_marker_x=100,
            left_marker_x=20,
            top_marker_y=20,
            bottom_marker_y=100,
            x_scale_factor=1.0,
            y_scale_factor=1.0,
        )
        op = AlignmentOperation(cfg)
        self.assertTrue(op.config.enabled)
        self.assertEqual(op.config.right_marker_x, 100)


class TestHierarchicalOperations(unittest.TestCase):
    def test_tiled_exposure_calls_sub_operations(self):
        stage = MockStage()
        projector = MockProjector()
        events = EventBus()
        project = ChipProject()
        layer = project.active_layer
        settings = PatterningSettings(
            exposure_time=10.0,
            tiling_enabled=True,
            tile_width=2000,
            tile_height=1000,
            pitch_x=100.0,
            pitch_y=100.0,
        )

        context = ExecutionContext(
            stage=stage,
            projector=projector,
            camera=DummyCamera(),
            project=project,
            event_bus=events,
            delay_func=lambda _: None,
        )
        manager = OperationManager(context, events)

        tiled_op = TiledExposureOperation(layer_index=0, settings=settings)
        manager.run(tiled_op)

        # Verify stage moves were recorded
        self.assertTrue(len(stage.moves) > 0)
        # Verify layer exposures recorded
        self.assertTrue(len(layer.exposures) > 0)
        # Verify projector mode ended in CLEAR
        self.assertEqual(projector.mode, ShownImage.CLEAR)


class TestProjectorController(unittest.TestCase):
    def test_mode_transitions(self):
        proj = MockProjector()
        self.assertEqual(proj.mode, ShownImage.CLEAR)

        proj.set_mode(ShownImage.RED_FOCUS)
        self.assertEqual(proj.mode, ShownImage.RED_FOCUS)

        proj.set_mode(ShownImage.CLEAR)
        self.assertEqual(proj.mode, ShownImage.CLEAR)
        self.assertIsNone(proj.current_image)


class TestUIInstantiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        if not QApplication.instance():
            cls.app = QApplication(["test", "-platform", "offscreen"])
        else:
            cls.app = QApplication.instance()

    def test_main_window_instantiation(self):
        from camera import get_camera
        from core.engine import StepperEngine
        from stage_control import get_stage_controller
        from ui.bridge import QtEngineBridge
        from ui.main_window import MainWindow

        stage = get_stage_controller({"enabled": False})
        camera = get_camera({"type": "none"})

        class MockProjector(ProjectorController):
            def size(self):
                return (1920, 1080)

            def clear(self):
                pass

            def show(self, img):
                pass

            def close(self):
                pass

        engine = StepperEngine(stage=stage, projector=MockProjector(), camera=camera)
        bridge = QtEngineBridge(engine)
        win = MainWindow(engine, bridge)
        self.assertIsNotNone(win)
        self.assertIsNotNone(win.workflow_panel_widget)
        self.assertIsNotNone(win.machine_control_widget)
        win.close()


class TestConsolidatedEvents(unittest.TestCase):
    def test_all_consolidated_events_exist_and_documented(self):
        expected_events = {
            # Project related
            "PROJECT_CHANGED",
            "ACTIVE_LAYER_CHANGED",
            "EXPOSURE_CONFIG_CHANGED",
            # Stage
            "STAGE_POSITION_CHANGED",
            # Projector
            "PROJECTOR_IMAGE_CHANGED",
            # Camera
            "CAMERA_FRAME_READY",
            # Operations
            "OPERATION_STARTED",
            "OPERATION_PROGRESS",
            "OPERATION_FINISHED",
            "OPERATION_ABORTED",
            # Warning
            "WARNING_MESSAGE",
        }
        actual_events = {e.name for e in Event}
        self.assertEqual(actual_events, expected_events)

    def test_bridge_signals_dispatch(self):
        from PySide6.QtWidgets import QApplication
        if not QApplication.instance():
            _ = QApplication(["test", "-platform", "offscreen"])

        stage = MockStage()
        projector = MockProjector()
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        signals_received = []
        bridge.project_changed.connect(lambda p: signals_received.append("project"))
        bridge.active_layer_changed.connect(lambda idx: signals_received.append(f"layer_{idx}"))
        bridge.exposure_config_changed.connect(lambda: signals_received.append("exposure_cfg"))
        bridge.stage_position_changed.connect(lambda pos: signals_received.append("stage"))
        bridge.projector_image_changed.connect(lambda mode: signals_received.append("projector"))
        bridge.camera_frame_ready.connect(lambda f: signals_received.append("camera"))
        bridge.warning_emitted.connect(lambda msg: signals_received.append(f"warn_{msg}"))

        # Emit events
        engine.event_bus.emit(Event.PROJECT_CHANGED)
        engine.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, 0)
        engine.event_bus.emit(Event.EXPOSURE_CONFIG_CHANGED)
        engine.event_bus.emit(Event.STAGE_POSITION_CHANGED)
        engine.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED, ShownImage.PATTERN)
        engine.event_bus.emit(Event.CAMERA_FRAME_READY, None)
        engine.event_bus.emit(Event.WARNING_MESSAGE, "Test warning")

        self.assertIn("project", signals_received)
        self.assertIn("layer_0", signals_received)
        self.assertIn("exposure_cfg", signals_received)
        self.assertIn("stage", signals_received)
        self.assertIn("projector", signals_received)
        self.assertIn("camera", signals_received)
        self.assertIn("warn_Test warning", signals_received)


if __name__ == "__main__":
    unittest.main()

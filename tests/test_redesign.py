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
    LayerSettingsOverride,
    PatterningSettings,
)
from core.events import ColorMode, Event, EventBus, ProjectorImageSource
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

        l2 = project.add_layer("Layer 2")
        l2.overrides.exposure_time = 3000.0

        disk_data = project.to_disk()
        restored = ChipProject.from_disk(disk_data)

        self.assertEqual(restored.name, "Test Chip")
        self.assertEqual(restored.settings.exposure_time, 9500.0)
        self.assertEqual(len(restored.layers), 2)
        self.assertEqual(restored.layers[0].pattern_path, "/path/to/mask1.png")
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

    def test_layer_tile_caching(self):
        import tempfile
        from PIL import Image
        project = ChipProject()
        layer = project.active_layer

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        img = Image.new("RGB", (200, 200), "white")
        img.save(temp_path)

        try:
            layer.set_pattern_path(temp_path)
            self.assertTrue(layer._pattern_cache_dirty)
            self.assertTrue(layer._tile_cache_dirty)

            tiles1 = layer.get_tiles(project.settings, (100, 100))
            self.assertFalse(layer._pattern_cache_dirty)
            self.assertFalse(layer._tile_cache_dirty)
            self.assertEqual(len(tiles1), 1)

            # Second call returns cached instance
            tiles2 = layer.get_tiles(project.settings, (100, 100))
            self.assertIs(tiles1[0], tiles2[0])

            # Modifying adjust invalidates tile cache
            layer.set_image_adjust((5.0, 5.0, 0.0))
            self.assertTrue(layer._tile_cache_dirty)
            tiles3 = layer.get_tiles(project.settings, (100, 100))
            self.assertFalse(layer._tile_cache_dirty)
            self.assertIsNot(tiles1[0], tiles3[0])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chip_project_active_tile(self):
        events = EventBus()
        emitted_tiles = []
        events.add_listener(Event.ACTIVE_TILE_CHANGED, lambda idx: emitted_tiles.append(idx))

        project = ChipProject(events=events)
        self.assertEqual(project.active_tile_index, 0)

        project.select_tile(3)
        self.assertEqual(project.active_tile_index, 3)
        self.assertIn(3, emitted_tiles)

        # Switching layer resets active tile to 0
        project.add_layer("Layer 2")
        self.assertEqual(project.active_tile_index, 0)
        self.assertEqual(emitted_tiles[-1], 0)


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
        # Verify projector color mode ended in DISABLE
        self.assertEqual(projector.color_mode, ColorMode.DISABLE)


class TestProjectorController(unittest.TestCase):
    def test_mode_transitions(self):
        proj = MockProjector()
        self.assertEqual(proj.color_mode, ColorMode.DISABLE)

        proj.set_color_mode(ColorMode.RED)
        self.assertEqual(proj.color_mode, ColorMode.RED)

        proj.set_color_mode(ColorMode.DISABLE)
        self.assertEqual(proj.color_mode, ColorMode.DISABLE)
        self.assertIsNone(proj.current_image)

    def test_projector_subscribes_and_updates_from_project(self):
        import tempfile
        from PIL import Image

        events = EventBus()
        proj = MockProjector()
        proj.event_bus = events

        project = ChipProject(events=events)
        proj.set_project(project)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        img = Image.new("RGB", (200, 200), "white")
        img.save(temp_path)

        try:
            project.active_layer.set_pattern_path(temp_path)

            # Set color mode to Red
            proj.set_color_mode(ColorMode.RED)
            self.assertTrue(len(proj.shown) > 0)
            self.assertIsNotNone(proj.current_image)

            # Switching active tile triggers update_display
            proj.shown.clear()
            project.select_tile(0)
            self.assertTrue(len(proj.shown) > 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


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
            "ACTIVE_TILE_CHANGED",
            "EXPOSURE_CONFIG_CHANGED",
            # Stage
            "STAGE_POSITION_CHANGED",
            # Projector
            "PROJECTOR_COLOR_MODE_CHANGED",
            "PROJECTOR_IMAGE_SOURCE_CHANGED",
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
        bridge.active_tile_changed.connect(lambda idx: signals_received.append(f"tile_{idx}"))
        bridge.exposure_config_changed.connect(lambda: signals_received.append("exposure_cfg"))
        bridge.stage_position_changed.connect(lambda pos: signals_received.append("stage"))
        bridge.projector_color_mode_changed.connect(lambda mode: signals_received.append("proj_color"))
        bridge.projector_image_source_changed.connect(lambda src: signals_received.append("proj_src"))
        bridge.projector_image_changed.connect(lambda img: signals_received.append("projector_img"))
        bridge.camera_frame_ready.connect(lambda f: signals_received.append("camera"))
        bridge.warning_emitted.connect(lambda msg: signals_received.append(f"warn_{msg}"))

        # Emit events
        engine.event_bus.emit(Event.PROJECT_CHANGED)
        engine.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, 0)
        engine.event_bus.emit(Event.ACTIVE_TILE_CHANGED, 0)
        engine.event_bus.emit(Event.EXPOSURE_CONFIG_CHANGED)
        engine.event_bus.emit(Event.STAGE_POSITION_CHANGED)
        engine.projector.set_color_mode(ColorMode.RED)
        engine.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
        engine.event_bus.emit(Event.CAMERA_FRAME_READY, None)
        engine.event_bus.emit(Event.WARNING_MESSAGE, "Test warning")

        self.assertIn("project", signals_received)
        self.assertIn("layer_0", signals_received)
        self.assertIn("tile_0", signals_received)
        self.assertIn("exposure_cfg", signals_received)
        self.assertIn("stage", signals_received)
        self.assertIn("proj_color", signals_received)
        self.assertIn("proj_src", signals_received)
        self.assertIn("projector_img", signals_received)
        self.assertIn("camera", signals_received)
        self.assertIn("warn_Test warning", signals_received)


class TestChipProjectAndProjectorRefinement(unittest.TestCase):
    def test_chiplayer_fields_and_caching(self):
        from dataclasses import fields
        import tempfile
        from PIL import Image

        layer = ChipLayer()
        # Verify exact field names
        field_names = {f.name for f in fields(layer)}
        expected_fields = {
            "name",
            "pattern_path",
            "image_adjust",
            "overrides",
            "_pattern_cache",
            "_pattern_cache_dirty",
            "_tile_cache",
            "_tile_cache_dirty",
            "events",
        }
        self.assertEqual(field_names, expected_fields)

        # Verify initial states
        self.assertEqual(layer.name, "Layer 1")
        self.assertIsNone(layer.pattern_path)
        self.assertEqual(layer.image_adjust, (0.0, 0.0, 0.0))
        self.assertIsNone(layer._pattern_cache)
        self.assertFalse(layer._pattern_cache_dirty)
        self.assertEqual(layer._tile_cache, [])
        self.assertFalse(layer._tile_cache_dirty)
        self.assertIsNone(layer.events)

        # Test loading and caching with tiling enabled
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        img = Image.new("RGB", (4000, 3000), "white")
        img.save(temp_path)

        try:
            settings = PatterningSettings(
                tiling_enabled=True,
                tile_width=2000,
                tile_height=1500,
                overlap_x=200,
                overlap_y=200,
            )
            layer.set_pattern_path(temp_path)
            self.assertTrue(layer._pattern_cache_dirty)
            self.assertTrue(layer._tile_cache_dirty)

            tiles = layer.get_tiles(settings, (1920, 1080))
            self.assertGreater(len(tiles), 1)
            self.assertFalse(layer._tile_cache_dirty)

            # Slicing disabled -> 1 tile
            settings.tiling_enabled = False
            layer.mark_dirty()
            single_tile = layer.get_tiles(settings, (1920, 1080))
            self.assertEqual(len(single_tile), 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chipproject_render_for_projector(self):
        import tempfile
        from PIL import Image

        project = ChipProject()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        img = Image.new("RGB", (200, 200), "white")
        img.save(temp_path)

        try:
            project.active_layer.set_pattern_path(temp_path)

            # DISABLE mode -> returns None
            disabled = project.render_for_projector(
                ColorMode.DISABLE, ProjectorImageSource.ACTIVE_LAYER, projector_size=(100, 100)
            )
            self.assertIsNone(disabled)

            # RED mode -> non-empty image with red channel active
            red_img = project.render_for_projector(
                ColorMode.RED, ProjectorImageSource.ACTIVE_LAYER, projector_size=(100, 100)
            )
            self.assertIsNotNone(red_img)
            self.assertEqual(red_img.size, (100, 100))
            r, g, b = red_img.split()[:3]
            self.assertGreater(r.getextrema()[1], 0)
            self.assertEqual(g.getextrema()[1], 0)
            self.assertEqual(b.getextrema()[1], 0)

            # UV mode -> non-empty image with blue channel active
            uv_img = project.render_for_projector(
                ColorMode.UV, ProjectorImageSource.ACTIVE_LAYER, projector_size=(100, 100)
            )
            self.assertIsNotNone(uv_img)
            r, g, b = uv_img.split()[:3]
            self.assertEqual(r.getextrema()[1], 0)
            self.assertEqual(g.getextrema()[1], 0)
            self.assertGreater(b.getextrema()[1], 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestExposureColorModeAndTilingUpdates(unittest.TestCase):
    def _create_engine(self):
        stage = MockStage()
        camera = DummyCamera()
        projector = MockProjector()
        return StepperEngine(stage=stage, projector=projector, camera=camera)

    def test_exposure_operation_restores_color_mode(self):
        engine = self._create_engine()
        engine.projector.set_color_mode(ColorMode.RED)
        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

        engine.project.settings.exposure_time = 10.0
        op = ExposureOperation(layer_index=0, settings=engine.project.settings)
        op.execute(engine.context, lambda p, m: None)

        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

    def test_tiled_exposure_operation_restores_color_mode(self):
        engine = self._create_engine()
        engine.projector.set_color_mode(ColorMode.RED)
        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

        engine.project.settings.exposure_time = 10.0
        engine.project.settings.tile_width = 1000
        engine.project.settings.tile_height = 1000
        op = TiledExposureOperation(layer_index=0, settings=engine.project.settings)
        op.execute(engine.context, lambda p, m: None)

        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

    def test_chip_project_update_settings_invalidates_caches_and_emits_event(self):
        bus = EventBus()
        events_received = []
        bus.add_listener(Event.EXPOSURE_CONFIG_CHANGED, lambda *args: events_received.append(True))

        project = ChipProject(events=bus)
        layer = project.active_layer
        layer._tile_cache = ["dummy_tile"]
        layer._tile_cache_dirty = False

        project.update_settings(exposure_time=1234.0, tiling_enabled=True, tile_width=800)
        self.assertEqual(project.settings.exposure_time, 1234.0)
        self.assertTrue(project.settings.tiling_enabled)
        self.assertEqual(project.settings.tile_width, 800)
        self.assertTrue(layer._tile_cache_dirty)
        self.assertEqual(len(events_received), 1)

    def test_workflow_panel_widgets_sync_and_regenerate(self):
        from PySide6.QtWidgets import QApplication
        from ui.bridge import QtEngineBridge
        from ui.widgets.workflow_panel import (
            ProjectSubpanelWidget,
            LayerSubpanelWidget,
            ActionSubpanelWidget,
        )

        if not QApplication.instance():
            _ = QApplication(["test", "-platform", "offscreen"])
        engine = self._create_engine()
        bridge = QtEngineBridge(engine)

        proj_panel = ProjectSubpanelWidget(engine, bridge)
        layer_panel = LayerSubpanelWidget(engine, bridge)
        action_panel = ActionSubpanelWidget(engine, bridge)

        # Verify btn_regenerate_tiles exists
        self.assertTrue(hasattr(layer_panel, "btn_regenerate_tiles"))

        # Changing project settings via spinbox or update_settings updates other panels
        proj_panel.spin_default_exp.setValue(6543)
        self.assertIn("6543", action_panel.lbl_active_exp.text())

        proj_panel.chk_default_tiling.setChecked(True)
        self.assertIn("Enabled", layer_panel.lbl_tiling_status.text())
        self.assertIn("Tiling", action_panel.lbl_active_mode.text())

        # With no pattern loaded, regenerate button is disabled
        self.assertFalse(layer_panel.btn_regenerate_tiles.isEnabled())

        # Load a temporary pattern to enable regeneration
        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
            img = Image.new("RGB", (2000, 2000), (255, 255, 255))
            img.save(temp_path)

        try:
            engine.project.active_layer.set_pattern_path(temp_path)
            engine.event_bus.emit(Event.PROJECT_CHANGED, engine.project)
            self.assertTrue(layer_panel.btn_regenerate_tiles.isEnabled())

            dummy_tile = "old_dummy_tile"
            engine.project.active_layer._tile_cache = [dummy_tile]
            engine.project.active_layer._tile_cache_dirty = False
            layer_panel.btn_regenerate_tiles.click()
            self.assertNotIn(dummy_tile, engine.project.active_layer._tile_cache)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chiplayer_emits_exposure_config_changed(self):
        bus = EventBus()
        events_received = []
        bus.add_listener(Event.EXPOSURE_CONFIG_CHANGED, lambda *args: events_received.append(True))

        layer = ChipLayer(name="Test Layer", events=bus)

        # 1. set_image_adjust emits
        layer.set_image_adjust((10.0, 20.0, 5.0))
        self.assertEqual(len(events_received), 1)

        # 2. set_exposure_override emits
        layer.set_exposure_override(3000.0)
        self.assertEqual(len(events_received), 2)

        # 3. set_tiling_override emits
        layer.set_tiling_override(True)
        self.assertEqual(len(events_received), 3)

        # 4. update_overrides emits
        layer.update_overrides(exposure_time=4000.0)
        self.assertEqual(len(events_received), 4)

        # 5. regenerate_tiles emits
        layer.regenerate_tiles()
        self.assertEqual(len(events_received), 5)


if __name__ == "__main__":
    unittest.main()

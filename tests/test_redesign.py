import io
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
    ExposureOperationConfig,
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
        eff1 = project.settings.with_overrides(project.active_layer.overrides)
        self.assertEqual(eff1.exposure_time, 12000.0)
        self.assertFalse(eff1.tiling_enabled)

        # Add Layer 2 with overrides
        l2 = project.add_layer("Layer 2")
        l2.overrides.exposure_time = 4500.0
        l2.overrides.tiling_enabled = True

        eff2 = project.settings.with_overrides(l2.overrides)
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
        import cv2
        import numpy as np
        project = ChipProject()
        layer = project.active_layer

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, np.full((200, 200, 3), 255, dtype=np.uint8))

        try:
            layer.set_pattern_path(temp_path)
            self.assertEqual(layer._tile_cache, [])
            self.assertEqual(layer._tile_coords, [])

            tiles1, coords1 = layer.generate_tiles()
            self.assertEqual(len(tiles1), 1)
            self.assertEqual(len(coords1), 1)

            # Second call returns cached instance
            tiles2, coords2 = layer.generate_tiles()
            self.assertIs(tiles1[0], tiles2[0])

            # Modifying overrides invalidates tile cache
            layer.update_overrides(exposure_time=5000.0)
            self.assertEqual(layer._tile_cache, [])
            tiles3, coords3 = layer.generate_tiles()
            self.assertEqual(len(tiles3), 1)
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

    def projector_size(self):
        return (1920, 1080)

    def update_display(self):
        if self._displayed_image_cache is not None:
            self.shown.append(self._displayed_image_cache)
            self.cleared = False
        else:
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

    def test_operation_failure_dispatches_error_and_prints(self):
        failed_events = []
        self.events.add_listener(
            Event.OPERATION_FAILED,
            lambda name, err: failed_events.append((name, err)),
        )

        class FailingOp(Operation):
            def __init__(self):
                super().__init__("FailingOp")

            def execute(self, context, report_progress):
                return "Something broke"

        op = FailingOp()
        error_callbacks = []
        finished_callbacks = []

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.manager.start_operation(
                op,
                run_async_callback=lambda w: w(),
                on_finished=lambda: finished_callbacks.append(True),
                on_error=lambda err: error_callbacks.append(err),
            )
            stdout_val = mock_stdout.getvalue()

        self.assertEqual(len(failed_events), 1)
        self.assertEqual(failed_events[0], ("FailingOp", "Something broke"))
        self.assertEqual(error_callbacks, ["Something broke"])
        self.assertEqual(finished_callbacks, [True])
        self.assertIn("[Operation Error]", stdout_val)
        self.assertIn("Something broke", stdout_val)

    def test_operation_abort_prints(self):
        op = DummyOperation("AbortToPrint")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            def mock_run_async(worker):
                self.manager.abort_current()
                worker()

            self.manager.start_operation(op, run_async_callback=mock_run_async)
            stdout_val = mock_stdout.getvalue()

        self.assertIn("[Operation Aborted]", stdout_val)
        self.assertIn("AbortToPrint", stdout_val)

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
        layer._tile_cache = [np.zeros((1000, 2000, 3), dtype=np.uint8), np.zeros((1000, 2000, 3), dtype=np.uint8)]
        layer._tile_coords = [(0.0, 0.0), (100.0, 0.0)]
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
        # Verify projector ended turned off
        self.assertFalse(projector.is_on)


class TestProjectorController(unittest.TestCase):
    def test_mode_transitions(self):
        proj = MockProjector()
        self.assertEqual(proj.color_mode, ColorMode.RED)
        self.assertFalse(proj.is_on)

        proj.set_color_mode(ColorMode.UV)
        self.assertEqual(proj.color_mode, ColorMode.UV)

        proj.set_color_mode(ColorMode.RED)
        self.assertEqual(proj.color_mode, ColorMode.RED)

        # Turning on shows image, turning off blanks it
        proj.set_on(True)
        self.assertTrue(proj.is_on)

        proj.set_on(False)
        self.assertFalse(proj.is_on)

    def test_projector_subscribes_and_updates_from_project(self):
        import tempfile
        import cv2
        import numpy as np

        events = EventBus()
        proj = MockProjector()
        proj.event_bus = events

        project = ChipProject(events=events)
        proj.set_project(project)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, np.full((200, 200, 3), 255, dtype=np.uint8))

        try:
            project.active_layer.set_pattern_path(temp_path)
            project.active_layer.generate_tiles()

            # Turn on projector and set color mode to Red
            proj.set_on(True)
            proj.set_color_mode(ColorMode.RED)
            self.assertTrue(len(proj.shown) > 0)
            self.assertIsNotNone(proj._displayed_image_cache)

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
            def projector_size(self):
                return (1920, 1080)

            def update_display(self):
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
            "LAYER_CACHE_RECOMPUTED",
            "EXPOSURE_HISTORY_CHANGED",
            # Stage
            "STAGE_POSITION_CHANGED",
            # Projector
            "PROJECTOR_ON_OFF_CHANGED",
            "PROJECTOR_COLOR_MODE_CHANGED",
            "PROJECTOR_IMAGE_SOURCE_CHANGED",
            "PROJECTOR_BRIGHTNESS_CHANGED",
            "PROJECTOR_IMAGE_CHANGED",
            # Camera
            "CAMERA_FRAME_READY",
            # Operations
            "OPERATION_STARTED",
            "OPERATION_PROGRESS",
            "OPERATION_FINISHED",
            "OPERATION_ABORTED",
            "OPERATION_FAILED",
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
        bridge.layer_cache_recomputed.connect(lambda l: signals_received.append("layer_recomputed"))
        bridge.stage_position_changed.connect(lambda pos: signals_received.append("stage"))
        bridge.projector_color_mode_changed.connect(lambda mode: signals_received.append("proj_color"))
        bridge.projector_image_source_changed.connect(lambda src: signals_received.append("proj_src"))
        bridge.projector_brightness_changed.connect(lambda b: signals_received.append(f"proj_brightness_{b}"))
        bridge.projector_on_off_changed.connect(lambda on: signals_received.append(f"proj_on_{on}"))
        bridge.projector_image_changed.connect(lambda img: signals_received.append("projector_img"))
        bridge.camera_frame_ready.connect(lambda f: signals_received.append("camera"))
        bridge.warning_emitted.connect(lambda msg: signals_received.append(f"warn_{msg}"))

        # Emit events
        engine.event_bus.emit(Event.PROJECT_CHANGED)
        engine.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, 0)
        engine.event_bus.emit(Event.ACTIVE_TILE_CHANGED, 0)
        engine.event_bus.emit(Event.EXPOSURE_CONFIG_CHANGED)
        engine.event_bus.emit(Event.LAYER_CACHE_RECOMPUTED, None)
        engine.event_bus.emit(Event.STAGE_POSITION_CHANGED)
        engine.event_bus.emit(Event.PROJECTOR_ON_OFF_CHANGED, True)
        engine.event_bus.emit(Event.PROJECTOR_IMAGE_CHANGED, None)
        engine.projector.set_color_mode(ColorMode.RED)
        engine.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
        engine.projector.set_brightness(0.5)
        engine.event_bus.emit(Event.CAMERA_FRAME_READY, None)
        engine.event_bus.emit(Event.WARNING_MESSAGE, "Test warning")

        self.assertIn("project", signals_received)
        self.assertIn("layer_0", signals_received)
        self.assertIn("tile_0", signals_received)
        self.assertIn("exposure_cfg", signals_received)
        self.assertIn("layer_recomputed", signals_received)
        self.assertIn("stage", signals_received)
        self.assertIn("proj_on_True", signals_received)
        self.assertIn("proj_color", signals_received)
        self.assertIn("proj_src", signals_received)
        self.assertIn("proj_brightness_0.5", signals_received)
        self.assertIn("projector_img", signals_received)
        self.assertIn("camera", signals_received)
        self.assertIn("warn_Test warning", signals_received)


class TestChipProjectAndProjectorRefinement(unittest.TestCase):
    def test_chiplayer_fields_and_caching(self):
        from dataclasses import fields
        import tempfile
        import cv2
        import numpy as np

        layer = ChipLayer()
        # Verify exact field names
        field_names = {f.name for f in fields(layer)}
        expected_fields = {
            "name",
            "pattern_path",
            "scale_w",
            "scale_h",
            "threshold",
            "overrides",
            "_pattern_cache",
            "_tile_cache",
            "_tile_coords",
            "events",
            "_project",
        }
        self.assertEqual(field_names, expected_fields)

        # Verify initial states
        self.assertEqual(layer.name, "Layer 1")
        self.assertIsNone(layer.pattern_path)
        self.assertEqual(layer.scale_w, -1)
        self.assertEqual(layer.scale_h, -1)
        self.assertEqual(layer.threshold, 50)
        self.assertIsNone(layer._pattern_cache)
        self.assertEqual(layer._tile_cache, [])
        self.assertEqual(layer._tile_coords, [])
        self.assertIsNone(layer.events)
        self.assertIsNone(layer._project)

        # Test loading and caching with tiling enabled
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, np.full((3000, 4000, 3), 255, dtype=np.uint8))

        try:
            project = ChipProject()
            layer = project.active_layer
            layer.overrides.tiling_enabled = True
            layer.overrides.tile_width = 2000
            layer.overrides.tile_height = 1500
            layer.overrides.overlap_x = 200
            layer.overrides.overlap_y = 200

            layer.set_pattern_path(temp_path)
            self.assertEqual(layer._tile_cache, [])

            tiles, coords = layer.generate_tiles()
            self.assertGreater(len(tiles), 1)
            self.assertEqual(len(tiles), len(coords))

            # Slicing disabled -> 1 tile
            layer.overrides.tiling_enabled = False
            single_tile, single_coords = layer.generate_tiles(force=True)
            self.assertEqual(len(single_tile), 1)
            self.assertEqual(single_coords, [(0.0, 0.0)])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chipproject_render_for_projector(self):
        import tempfile
        import cv2
        import numpy as np

        project = ChipProject()
        proj = MockProjector()
        proj.set_project(project)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, np.full((200, 200, 3), 255, dtype=np.uint8))

        try:
            project.active_layer.set_pattern_path(temp_path)
            project.active_layer.generate_tiles()
            proj.set_on(True)
            proj.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

            # RED mode -> non-empty image with red channel active (index 2 in BGR)
            proj.set_color_mode(ColorMode.RED)
            red_img = proj._displayed_image_cache
            self.assertIsNotNone(red_img)
            self.assertEqual(red_img.shape[:2], (200, 200))
            self.assertEqual(np.max(red_img[:, :, 0]), 0)
            self.assertEqual(np.max(red_img[:, :, 1]), 0)
            self.assertGreater(np.max(red_img[:, :, 2]), 0)

            # UV mode -> non-empty image with blue channel active (index 0 in BGR)
            proj.set_color_mode(ColorMode.UV)
            uv_img = proj._displayed_image_cache
            self.assertIsNotNone(uv_img)
            self.assertGreater(np.max(uv_img[:, :, 0]), 0)
            self.assertEqual(np.max(uv_img[:, :, 1]), 0)
            self.assertEqual(np.max(uv_img[:, :, 2]), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_exposure_config_change_does_not_trigger_tile_recompute(self):
        import tempfile
        import cv2
        import numpy as np

        stage = MockStage()
        camera = DummyCamera()
        projector = MockProjector()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, np.full((200, 200, 3), 255, dtype=np.uint8))

        try:
            layer = engine.project.active_layer
            layer.set_pattern_path(temp_path)

            # Generate tiles initially with force=True
            tiles, coords = layer.generate_tiles(force=True)
            self.assertGreater(len(tiles), 0)

            # Track update_display calls on projector
            display_updates = []
            orig_update_display = projector.update_display
            projector.update_display = lambda: display_updates.append(True)

            # Changing exposure settings clears tiling cache
            engine.project.update_settings(pitch_x=50.0)
            self.assertEqual(layer._tile_cache, [])
            self.assertEqual(layer._tile_coords, [])
            # Projector should NOT have updated because it listens to LAYER_CACHE_RECOMPUTED
            self.assertEqual(len(display_updates), 0)

            # layer.set_exposure_override should also clear cache
            layer.set_exposure_override(9999.0)
            self.assertEqual(layer._tile_cache, [])
            self.assertEqual(len(display_updates), 0)

            # Calling generate_tiles with force=False when cache empty -> recomputes and emits LAYER_CACHE_RECOMPUTED
            layer.generate_tiles(force=False)
            self.assertGreater(len(layer._tile_cache), 0)
            self.assertEqual(len(display_updates), 1)

            # Calling generate_tiles with force=False when cached -> skips recomputing and does not emit
            layer.generate_tiles(force=False)
            self.assertEqual(len(display_updates), 1)

            # Calling generate_tiles with force=True -> recomputes even when cached and emits
            layer.generate_tiles(force=True)
            self.assertEqual(len(display_updates), 2)

            projector.update_display = orig_update_display
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestExposureColorModeAndTilingUpdates(unittest.TestCase):
    def _create_engine(self):
        stage = MockStage()
        camera = DummyCamera()
        projector = MockProjector()
        return StepperEngine(stage=stage, projector=projector, camera=camera)

    def test_exposure_operation_disables_projector_after_exposure(self):
        engine = self._create_engine()
        engine.projector.set_color_mode(ColorMode.RED)
        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

        engine.project.settings.exposure_time = 10.0
        op = ExposureOperation(
            layer_index=0,
            config=ExposureOperationConfig(exposure_time=engine.project.settings.exposure_time),
        )
        op.execute(engine.context, lambda p, m: None)

        self.assertFalse(engine.projector.is_on)

    def test_exposure_operation_disables_projector_even_if_previously_uv(self):
        engine = self._create_engine()
        engine.projector.set_color_mode(ColorMode.UV)
        self.assertEqual(engine.projector.color_mode, ColorMode.UV)

        engine.project.settings.exposure_time = 10.0
        op = ExposureOperation(
            layer_index=0,
            config=ExposureOperationConfig(exposure_time=engine.project.settings.exposure_time),
        )
        op.execute(engine.context, lambda p, m: None)

        self.assertFalse(engine.projector.is_on)

    def test_tiled_exposure_operation_disables_projector_after_exposure(self):
        engine = self._create_engine()
        engine.projector.set_color_mode(ColorMode.RED)
        self.assertEqual(engine.projector.color_mode, ColorMode.RED)

        engine.project.settings.exposure_time = 10.0
        engine.project.settings.tile_width = 1000
        engine.project.settings.tile_height = 1000
        op = TiledExposureOperation(layer_index=0, settings=engine.project.settings)
        op.execute(engine.context, lambda p, m: None)

        self.assertFalse(engine.projector.is_on)

    def test_exposure_operation_preheats_tile_before_uv_mode(self):
        engine = self._create_engine()
        layer = engine.project.active_layer

        call_order = []
        orig_get_tile = layer.get_tile
        def spy_get_tile(*args, **kwargs):
            call_order.append("get_tile")
            return orig_get_tile(*args, **kwargs)
        layer.get_tile = spy_get_tile

        orig_set_on = engine.projector.set_on
        def spy_set_on(on):
            call_order.append(f"set_on_{on}")
            return orig_set_on(on)
        engine.projector.set_on = spy_set_on

        engine.project.settings.exposure_time = 20.0
        op = ExposureOperation(
            layer_index=0,
            config=ExposureOperationConfig(exposure_time=engine.project.settings.exposure_time),
        )
        op.execute(engine.context, lambda p, m: None)

        # Ensure get_tile is called BEFORE projector turns on
        self.assertIn("get_tile", call_order)
        self.assertIn("set_on_True", call_order)
        self.assertIn("set_on_False", call_order)
        get_tile_idx = call_order.index("get_tile")
        on_idx = call_order.index("set_on_True")
        off_idx = [i for i, x in enumerate(call_order) if x == "set_on_False"][-1]
        self.assertLess(get_tile_idx, on_idx, "Tile must be pre-heated before projector turns on")
        self.assertLess(on_idx, off_idx, "Projector must be turned off after exposure finishes")
        self.assertFalse(engine.projector.is_on)

    def test_chip_project_update_settings_invalidates_caches_and_emits_event(self):
        bus = EventBus()
        events_received = []
        bus.add_listener(Event.EXPOSURE_CONFIG_CHANGED, lambda *args: events_received.append(True))

        project = ChipProject(events=bus)
        layer = project.active_layer
        layer._tile_cache = ["dummy_tile"]
        layer._tile_coords = [(0.0, 0.0)]

        project.update_settings(exposure_time=1234.0, tiling_enabled=True, tile_width=800)
        self.assertEqual(project.settings.exposure_time, 1234.0)
        self.assertTrue(project.settings.tiling_enabled)
        self.assertEqual(project.settings.tile_width, 800)
        self.assertEqual(layer._tile_cache, [])
        self.assertEqual(layer._tile_coords, [])
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
        import cv2
        import numpy as np
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_path = tf.name
            cv2.imwrite(temp_path, np.full((2000, 2000, 3), 255, dtype=np.uint8))

        try:
            engine.project.active_layer.set_pattern_path(temp_path)
            engine.event_bus.emit(Event.PROJECT_CHANGED, engine.project)
            self.assertTrue(layer_panel.btn_regenerate_tiles.isEnabled())

            dummy_tile = np.zeros((10, 10, 3), dtype=np.uint8)
            engine.project.active_layer._tile_cache = [dummy_tile]
            layer_panel.btn_regenerate_tiles.click()
            self.assertFalse(any(dummy_tile is t for t in engine.project.active_layer._tile_cache))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chiplayer_emits_exposure_config_changed(self):
        bus = EventBus()
        events_received = []
        recomputed_received = []
        bus.add_listener(Event.EXPOSURE_CONFIG_CHANGED, lambda *args: events_received.append(True))
        bus.add_listener(Event.LAYER_CACHE_RECOMPUTED, lambda *args: recomputed_received.append(args[0]))

        project = ChipProject(events=bus)
        layer = project.active_layer

        # 1. set_exposure_override emits EXPOSURE_CONFIG_CHANGED and clears cache
        layer._tile_cache = ["dummy"]
        layer.set_exposure_override(3000.0)
        self.assertEqual(len(events_received), 1)
        self.assertEqual(layer._tile_cache, [])

        # 2. set_tiling_override emits EXPOSURE_CONFIG_CHANGED and clears cache
        layer._tile_cache = ["dummy"]
        layer.set_tiling_override(True)
        self.assertEqual(len(events_received), 2)
        self.assertEqual(layer._tile_cache, [])

        # 3. update_overrides emits EXPOSURE_CONFIG_CHANGED and clears cache
        layer._tile_cache = ["dummy"]
        layer.update_overrides(exposure_time=4000.0)
        self.assertEqual(len(events_received), 3)
        self.assertEqual(layer._tile_cache, [])

        # 4. generate_tiles emits LAYER_CACHE_RECOMPUTED from chip layer
        layer.generate_tiles(force=True)
        self.assertEqual(len(events_received), 3)
        self.assertEqual(len(recomputed_received), 1)
        self.assertIs(recomputed_received[0], layer)

    def test_tiled_exposure_exact_sequence(self):
        from unittest.mock import patch

        actions = []
        engine = self._create_engine()
        layer = engine.project.active_layer
        layer.generate_tiles = lambda force=False: (
            ["tile0", "tile1"],
            [(10.0, 20.0), (30.0, 40.0)],
        )

        orig_set_on = engine.projector.set_on
        def spy_set_on(on):
            actions.append(("projector_on", on))
            return orig_set_on(on)
        engine.projector.set_on = spy_set_on

        orig_set_color = engine.projector.set_color_mode
        def spy_set_color(mode):
            actions.append(("projector_color", mode))
            return orig_set_color(mode)
        engine.projector.set_color_mode = spy_set_color

        orig_set_src = engine.projector.set_image_source
        def spy_set_src(src):
            actions.append(("projector_src", src))
            return orig_set_src(src)
        engine.projector.set_image_source = spy_set_src

        orig_select_tile = engine.project.select_tile
        def spy_select_tile(idx):
            actions.append(("select_tile", idx))
            return orig_select_tile(idx)
        engine.project.select_tile = spy_select_tile

        orig_move_abs = engine.stage.move_absolute
        def spy_move_abs(coords):
            actions.append(("stage_move", coords))
            return orig_move_abs(coords)
        engine.stage.move_absolute = spy_move_abs

        with patch("operations.tiling.AutofocusOperation.execute") as mock_af, \
             patch("operations.tiling.ExposureOperation.execute") as mock_exp:
            mock_af.side_effect = lambda ctx, prog: actions.append(("autofocus",))
            mock_exp.side_effect = lambda ctx, prog: actions.append(("exposure",))

            engine.projector.set_on(False)
            actions.clear()

            op = TiledExposureOperation(layer_index=0, settings=engine.project.settings)
            err = op.execute(engine.context, lambda p, m: None)
            self.assertIsNone(err)

        expected_sequence = [
            # Tile 0
            ("projector_on", False),
            ("stage_move", {"x": 10.0, "y": 20.0}),
            ("select_tile", 0),
            ("projector_src", ProjectorImageSource.ACTIVE_LAYER),
            ("projector_color", ColorMode.RED),
            ("autofocus",),
            ("exposure",),
            # Tile 1
            ("projector_on", False),
            ("stage_move", {"x": 30.0, "y": 40.0}),
            ("select_tile", 1),
            ("projector_src", ProjectorImageSource.ACTIVE_LAYER),
            ("projector_color", ColorMode.RED),
            ("autofocus",),
            ("exposure",),
            # Final restoration
            ("projector_on", False),
        ]
        self.assertEqual(actions, expected_sequence)


if __name__ == "__main__":
    unittest.main()


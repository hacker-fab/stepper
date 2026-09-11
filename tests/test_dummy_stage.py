import math
import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage_control.stage_controller import StageController
from stage_control.dummy_stage import DummyStage
from stage_control import get_stage_controller


class TestStageControllerAbstract(unittest.TestCase):
    def test_cannot_instantiate_abstract_stage_controller(self):
        with self.assertRaises(TypeError):
            StageController()


class TestDummyStage(unittest.TestCase):
    def test_initial_position_and_bounds(self):
        stage = DummyStage()
        self.assertEqual(stage.get_position(), (0.0, 0.0, 0.0))
        self.assertTrue(stage.has_homing())
        self.assertIsNotNone(stage.get_bounds())
        self.assertIn("x", stage.get_bounds())

    def test_custom_initial_position(self):
        stage = DummyStage(initial_position=(100.0, 200.0, 300.0))
        self.assertEqual(stage.get_position(), (100.0, 200.0, 300.0))

    def test_move_relative_and_absolute(self):
        stage = DummyStage(delay=0.0)
        
        # Relative move
        res = stage.move_relative({"x": 50.0, "y": -20.0, "z": 10.0})
        self.assertTrue(res)
        self.assertEqual(stage.get_position(), (50.0, -20.0, 10.0))

        # Relative move with uppercase keys
        stage.move_relative({"X": 10.0, "Y": 20.0})
        self.assertEqual(stage.get_position(), (60.0, 0.0, 10.0))

        # Absolute move
        res = stage.move_absolute({"x": 100.0, "z": 0.0})
        self.assertTrue(res)
        self.assertEqual(stage.get_position(), (100.0, 0.0, 0.0))

        # Home
        stage.home()
        self.assertEqual(stage.get_position(), (0.0, 0.0, 0.0))

    def test_delays_fixed(self):
        delays_recorded = []
        stage = DummyStage(delay=0.05, delay_func=lambda d: delays_recorded.append(d))
        
        stage.move_relative({"x": 100.0})
        self.assertEqual(len(delays_recorded), 1)
        self.assertAlmostEqual(delays_recorded[0], 0.05)

        # Zero distance move should not trigger delay
        stage.move_relative({"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(len(delays_recorded), 1)

    def test_delays_proportional_speed(self):
        delays_recorded = []
        speed = 1000.0  # um/s
        stage = DummyStage(speed=speed, delay_func=lambda d: delays_recorded.append(d))

        stage.move_absolute({"x": 300.0, "y": 400.0, "z": 0.0})
        expected_distance = math.sqrt(300.0**2 + 400.0**2)  # 500.0
        expected_time = expected_distance / speed  # 0.5s

        self.assertEqual(len(delays_recorded), 1)
        self.assertAlmostEqual(delays_recorded[0], expected_time)

    def test_factory_get_stage_controller(self):
        dummy = get_stage_controller({"type": "dummy", "delay": 0.02, "autofocus": 5.0})
        self.assertIsInstance(dummy, DummyStage)
        self.assertEqual(dummy.delay, 0.02)

        disabled = get_stage_controller({"enabled": False})
        self.assertIsInstance(disabled, DummyStage)

        fallback = get_stage_controller({"type": "non_existent_stage_type"})
        self.assertIsInstance(fallback, DummyStage)


if __name__ == "__main__":
    unittest.main()

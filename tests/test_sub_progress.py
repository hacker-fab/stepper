import unittest
from core.operation import sub_progress


class TestSubProgress(unittest.TestCase):
    def test_sub_progress_linear_mapping(self):
        reports = []
        def reporter(pct: float, msg: str):
            reports.append((round(pct, 4), msg))

        # Map 0.0 - 1.0 to 0.20 - 0.60
        sub = sub_progress(reporter, 0.20, 0.60, prefix="SubOp")

        sub(0.0, "Start")
        sub(0.5, "Halfway")
        sub(1.0, "Done")

        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0], (0.20, "SubOp - Start"))
        self.assertEqual(reports[1], (0.40, "SubOp - Halfway"))
        self.assertEqual(reports[2], (0.60, "SubOp - Done"))

    def test_sub_progress_clamping(self):
        reports = []
        def reporter(pct: float, msg: str):
            reports.append((round(pct, 4), msg))

        sub = sub_progress(reporter, 0.10, 0.90)

        # Values < 0.0 clamp to start_pct
        sub(-0.5, "Negative")
        self.assertEqual(reports[-1], (0.10, "Negative"))

        # Values > 1.0 clamp to end_pct
        sub(1.5, "Overflow")
        self.assertEqual(reports[-1], (0.90, "Overflow"))

    def test_sub_progress_without_prefix(self):
        reports = []
        def reporter(pct: float, msg: str):
            reports.append((round(pct, 4), msg))

        sub = sub_progress(reporter, 0.0, 1.0)
        sub(0.25, "Raw message without prefix")
        self.assertEqual(reports[0], (0.25, "Raw message without prefix"))

    def test_sub_progress_empty_message_with_prefix(self):
        reports = []
        def reporter(pct: float, msg: str):
            reports.append((round(pct, 4), msg))

        sub = sub_progress(reporter, 0.3, 0.7, prefix="Step 1")
        sub(0.0, "")
        self.assertEqual(reports[0], (0.3, "Step 1"))


if __name__ == "__main__":
    unittest.main()


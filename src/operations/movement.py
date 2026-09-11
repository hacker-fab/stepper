
from core.operation import Operation, ExecutionContext
from typing import Callable, Optional

class JogOperation(Operation):
    """Moves stage relative or absolute."""

    def __init__(self, coords: dict[str, float], relative: bool = True):
        mode_str = "relative" if relative else "absolute"
        super().__init__(f"Jog Stage ({mode_str})")
        self.coords = coords
        self.relative = relative

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> str | None:
        report_progress(0.0, f"Moving stage {self.coords}...")
        if self.relative:
            ok = context.stage.move_relative(self.coords)
        else:
            ok = context.stage.move_absolute(self.coords)

        report_progress(1.0, "Move complete")
        return None if ok else "Move failed"


class HomeOperation(Operation):
    """Homes the motion stage."""

    def __init__(self):
        super().__init__("Homing Stage")

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        report_progress(0.1, "Homing stage ($H)...")
        ok = context.stage.home()
        if ok:
            report_progress(1.0, "Stage homed")
            return None
        else:
            report_progress(1.0, "Stage homing failed")
            return "Stage homing failed"


import time
from datetime import datetime
from typing import Callable, Optional

from core.chip_project import ExposureLog, PatterningSettings
from core.events import Event, ShownImage
from core.operation import ExecutionContext, Operation


class ExposureOperation(Operation):
    """Exposes a single layer mask for a defined duration."""

    def __init__(self, layer_index: int, settings: PatterningSettings):
        super().__init__("Layer Exposure")
        self.layer_index = layer_index
        self.settings = settings

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        duration_ms = self.settings.exposure_time
        if context.project is None or self.layer_index >= len(context.project.layers):
            return "Invalid project or layer index"
        layer = context.project.layers[self.layer_index]

        report_progress(0.0, f"Starting exposure ({int(duration_ms)} ms)...")
        rendered = layer.render_pattern(self.settings, context.projector.size())
        context.projector.set_mode(ShownImage.PATTERN, rendered)

        start_t = time.time()
        end_t = start_t + (duration_ms / 1000.0)

        while time.time() < end_t:
            if self.is_aborted:
                break
            elapsed = time.time() - start_t
            pct = min(1.0, max(0.0, elapsed / (duration_ms / 1000.0)))
            report_progress(pct, f"Exposing {layer.name}... ({int(pct * 100)}%)")
            context.delay_func(0.1)

        context.projector.set_mode(ShownImage.CLEAR)

        log = ExposureLog(
            time=datetime.now(),
            path=layer.pattern_path or "",
            coords=context.stage.get_position(),
            duration=duration_ms,
            aborted=self.is_aborted,
        )
        layer.exposures.append(log)
        if context.event_bus:
            context.event_bus.emit(Event.PROJECT_CHANGED, context.project)

        if self.is_aborted:
            report_progress(1.0, "Exposure aborted")
            return "Exposure aborted"
        else:
            report_progress(1.0, "Exposure finished")
            return None

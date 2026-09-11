import time
from datetime import datetime
from typing import Callable, Optional

from core.chip_project import PatterningSettings
from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation


class ExposureOperation(Operation):
    """Exposes a single layer mask for a defined duration."""

    def __init__(self, layer_index: int, settings: PatterningSettings, tile_index: Optional[int] = None):
        super().__init__("Layer Exposure")
        self.layer_index = layer_index
        self.settings = settings
        self.tile_index = tile_index

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        duration_ms = self.settings.exposure_time
        if context.project is None or self.layer_index >= len(context.project.layers):
            return "Invalid project or layer index"
        layer = context.project.layers[self.layer_index]

        prev_color_mode = context.projector.color_mode
        try:
            report_progress(0.0, f"Starting exposure ({int(duration_ms)} ms)...")
            context.project.select_layer(self.layer_index)
            if self.tile_index is not None:
                context.project.select_tile(self.tile_index)
            context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
            context.projector.set_color_mode(ColorMode.UV)

            start_t = time.time()
            end_t = start_t + (duration_ms / 1000.0)

            progress_resolution = min((0.1, duration_ms / 1000.0 / 10))

            while time.time() < end_t:
                if self.is_aborted:
                    break
                elapsed = time.time() - start_t
                pct = min(1.0, max(0.0, elapsed / (duration_ms / 1000.0)))
                report_progress(pct, f"Exposing {layer.name}... ({int(pct * 100)}%)")
                context.delay_func(progress_resolution)
        finally:
            context.projector.set_color_mode(prev_color_mode)

        if self.is_aborted:
            report_progress(1.0, "Exposure aborted")
            return "Exposure aborted"
        else:
            report_progress(1.0, "Exposure finished")
            return None

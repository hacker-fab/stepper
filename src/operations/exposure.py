import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Union

from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation


@dataclass
class ExposureOperationConfig:
    """Configuration for exposure operation."""

    exposure_time: float = 8000.0  # ms

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "ExposureOperationConfig":
        if not d:
            return cls()
        return cls(
            exposure_time=float(d.get("exposure_time", 8000.0)),
        )


class ExposureOperation(Operation):
    """Exposes a single layer mask for a defined duration."""

    def __init__(
        self,
        layer_index: Optional[int],
        config: Optional[Union[ExposureOperationConfig, dict]] = None,
        tile_index: Optional[int] = None,
    ):
        super().__init__("Layer Exposure" if layer_index is not None else "Exposure")
        self.layer_index = layer_index
        if isinstance(config, dict):
            self.config = ExposureOperationConfig.from_dict(config)
        else:
            self.config = config or ExposureOperationConfig()
        self.tile_index = tile_index

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        duration_ms = self.config.exposure_time
        if self.layer_index is not None:
            if context.project is None or self.layer_index >= len(context.project.layers):
                msg = "Invalid project or layer index"
                report_progress(1.0, f"Exposure failed: {msg}")
                return msg
            layer = context.project.layers[self.layer_index]
            layer_name = layer.name
        else:
            layer = None
            layer_name = "Pattern"

        coords = context.stage.get_position() if context.stage else (0.0, 0.0, 0.0)
        report_progress(0.0, f"Preparing exposure for {layer_name}...")

        # Turn off projector
        context.projector.set_on(False)

        if self.layer_index is not None and context.project is not None:
            # Select layer and tile, and set projector image source to ACTIVE_LAYER
            context.project.select_layer(self.layer_index)
            if self.tile_index is not None:
                context.project.select_tile(self.tile_index)
            context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

            # Pre-render the tile into cache and configure UV pattern before turning projector on
            tile_idx = self.tile_index if self.tile_index is not None else context.project.active_tile_index
            layer.generate_tiles()
            layer.get_tile(tile_idx)

        context.projector.set_color_mode(ColorMode.UV)

        # sleep 3 sec to allow other events process so exposure time is accurate
        context.delay_func(3)
        # Turn on projector output and wait for display to render before starting timer
        context.projector.set_on(True)
        context.projector.wait_for_display(timeout=2.0)

        start_datetime = datetime.now()
        start_t = time.time()
        end_t = start_t + (duration_ms / 1000.0)

        try:
            report_progress(0.0, f"Starting exposure ({int(duration_ms)} ms)...")
            progress_resolution = min((0.1, duration_ms / 1000.0 / 10))

            while time.time() < end_t:
                if self.is_aborted:
                    break
                elapsed = time.time() - start_t
                pct = min(1.0, max(0.0, elapsed / (duration_ms / 1000.0)))
                report_progress(pct, f"Exposing {layer_name}... ({int(pct * 100)}%)")
                context.delay_func(progress_resolution)
        finally:
            # Ensure projector output is turned off immediately after exposure
            context.projector.set_on(False)
            context.projector.wait_for_display(timeout=1.0)
            elapsed_ms = (time.time() - start_t) * 1000.0
            context.projector.set_color_mode(ColorMode.RED)
            if context.project is not None:
                from core.chip_project import ExposureRecord
                record = ExposureRecord(
                    coords=coords,
                    time=start_datetime,
                    duration=elapsed_ms,
                    aborted=self.is_aborted,
                    layer_index=self.layer_index,
                    tile_index=self.tile_index,
                )
                context.project.add_exposure_record(record)

        if self.is_aborted:
            report_progress(1.0, "Exposure aborted")
            return "Exposure aborted"
        else:
            report_progress(1.0, "Exposure finished")
            return None

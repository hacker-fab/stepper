import os
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple
from PIL import Image

from core.chip_project import PatterningSettings
from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import Operation, ExecutionContext
from operations.movement import JogOperation
from operations.autofocus import AutofocusOperation
from lib.tiling import (
    calculate_tile_position,
    generate_snake_sequence,
    split_image_into_tiles,
    split_image_with_overlap,
)


class TiledExposureOperation(Operation):
    """Runs a multi-tile step-and-repeat exposure sequence composing lower-level operations directly."""

    def __init__(self, layer_index: int, settings: PatterningSettings):
        super().__init__("Tiled Exposure")
        self.layer_index = layer_index
        self.settings = settings

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.project is None or self.layer_index >= len(context.project.layers):
            return "Invalid project or layer index"
        layer = context.project.layers[self.layer_index]
        context.project.select_layer(self.layer_index)

        s = self.settings
        x_count = max(1, int(s.tile_width // 1000)) if s.tile_width else 3
        y_count = max(1, int(s.tile_height // 1000)) if s.tile_height else 3
        x_pitch = s.pitch_x
        y_pitch = s.pitch_y
        duration_ms = s.exposure_time

        seq = generate_snake_sequence(x_count, y_count)
        total_tiles = len(seq)
        pos = context.stage.get_position()
        x_start, y_start = pos[0], pos[1]

        report_progress(0.0, f"Beginning tiled exposure ({total_tiles} tiles)...")

        prev_color_mode = context.projector.color_mode
        try:
            for i, (x_idx, y_idx) in enumerate(seq):
                if self.is_aborted:
                    break

                pct_base = i / total_tiles
                report_progress(
                    pct_base,
                    f"Tile {i + 1}/{total_tiles} ({x_idx}, {y_idx}) - Moving stage...",
                )

                # Move stage if not initial tile via direct JogOperation execution
                if not (x_idx == 0 and y_idx == 0):
                    tx, ty = calculate_tile_position(
                        x_start, y_start, 1, 1, x_idx, y_idx, x_pitch, y_pitch
                    )
                    jog_op = JogOperation({"x": tx, "y": ty}, relative=False)
                    jog_op.execute(context, lambda p, m: None)

                if self.is_aborted:
                    break

                # Red autofocus via direct AutofocusOperation execution
                report_progress(pct_base, f"Tile {i + 1}/{total_tiles} - Autofocusing...")
                af_op = AutofocusOperation(blue_only=False)
                af_op.execute(context, lambda p, m: None)

                if self.is_aborted:
                    break

                # Expose tile
                report_progress(pct_base, f"Tile {i + 1}/{total_tiles} - Exposing...")
                context.project.select_tile(i)
                context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
                context.projector.set_color_mode(ColorMode.UV)

                end_t = time.time() + (duration_ms / 1000.0)
                while time.time() < end_t:
                    if self.is_aborted:
                        break
                    context.delay_func(0.02)

                context.projector.set_color_mode(ColorMode.DISABLE)
        finally:
            context.projector.set_color_mode(prev_color_mode)

        if context.event_bus:
            context.event_bus.emit(Event.PROJECT_CHANGED, context.project)
        if self.is_aborted:
            report_progress(1.0, "Tiled exposure aborted")
            return "Tiled exposure aborted"
        else:
            report_progress(1.0, "Tiled exposure complete")
            return None

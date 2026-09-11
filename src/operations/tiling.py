from typing import Callable, Optional

from core.chip_project import PatterningSettings
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation


class TiledExposureOperation(Operation):
    """Runs a multi-tile step-and-repeat exposure sequence composing lower-level operations directly."""

    def __init__(self, layer_index: int, settings: PatterningSettings):
        super().__init__("Tiled Exposure")
        self.layer_index = layer_index
        self.settings = settings
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.project is None or self.layer_index >= len(context.project.layers):
            return "Invalid project or layer index"

        context.project.select_layer(self.layer_index)
        layer = context.project.layers[self.layer_index]

        # 1. Get the tiling path for the active layer
        pos = context.stage.get_position()
        start_pos = (pos[0], pos[1])
        tiling_path = layer.get_tiling_path(self.settings, start_pos=start_pos)
        total_tiles = len(tiling_path)

        if total_tiles == 0:
            report_progress(1.0, "No tiles to expose")
            return None

        report_progress(0.0, f"Beginning tiled exposure ({total_tiles} tiles)...")

        prev_color_mode = context.projector.color_mode
        try:
            for tile_idx, (tx, ty) in enumerate(tiling_path):
                if self.is_aborted:
                    break

                pct_base = tile_idx / total_tiles
                pct_step = 1.0 / total_tiles

                # Step 1: Set the projector to black
                context.projector.set_color_mode(ColorMode.DISABLE)

                # Step 2: Move to the position
                report_progress(
                    pct_base,
                    f"Tile {tile_idx + 1}/{total_tiles} - Moving stage to ({tx:.1f}, {ty:.1f})...",
                )
                jog_op = JogOperation({"x": tx, "y": ty}, relative=False)
                err = self._run_sub_op(jog_op, context, lambda p, m: None)
                if err or self.is_aborted:
                    break

                # Step 3: Set the active tile
                context.project.select_tile(tile_idx)

                # Step 4: Source to active layer
                context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

                # Step 5: Set projector to red and run auto focus
                context.projector.set_color_mode(ColorMode.RED)
                report_progress(
                    pct_base + 0.3 * pct_step,
                    f"Tile {tile_idx + 1}/{total_tiles} - Autofocusing...",
                )
                af_op = AutofocusOperation(blue_only=False)
                err = self._run_sub_op(af_op, context, lambda p, m: None)
                if err or self.is_aborted:
                    break

                # Step 6: Then, do an exposure
                report_progress(
                    pct_base + 0.6 * pct_step,
                    f"Tile {tile_idx + 1}/{total_tiles} - Exposing...",
                )
                exp_op = ExposureOperation(
                    layer_index=self.layer_index,
                    settings=self.settings,
                    tile_index=tile_idx,
                )
                err = self._run_sub_op(
                    exp_op,
                    context,
                    lambda p, m: report_progress(
                        pct_base + (0.6 + 0.4 * p) * pct_step,
                        f"Tile {tile_idx + 1}/{total_tiles} - {m}",
                    ),
                )
                if err or self.is_aborted:
                    break
        finally:
            context.projector.set_color_mode(prev_color_mode)

        if self.is_aborted:
            report_progress(1.0, "Tiled exposure aborted")
            return "Tiled exposure aborted"
        else:
            report_progress(1.0, "Tiled exposure complete")
            return None

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        progress_cb: Callable[[float, str], None],
    ) -> Optional[str]:
        if self.is_aborted:
            return "Aborted"
        self._current_sub_op = op
        try:
            return op.execute(context, progress_cb)
        finally:
            self._current_sub_op = None

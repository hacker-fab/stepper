from typing import Callable, Optional, Any

from core.chip_project import PatterningSettings
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation, ExposureOperationConfig
from operations.movement import JogOperation


class TiledExposureOperation(Operation):
    """Runs a multi-tile step-and-repeat exposure sequence composing lower-level operations directly."""

    def __init__(
        self,
        layer_index: int,
        settings: PatterningSettings,
        autofocus_config: Optional[Any] = None,
    ):
        super().__init__("Tiled Exposure")
        self.layer_index = layer_index
        self.settings = settings
        self.autofocus_config = autofocus_config
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
        tiles, tile_coords = layer.generate_tiles()
        total_tiles = len(tiles)

        if total_tiles == 0:
            report_progress(1.0, "No tiles to expose")
            return None

        report_progress(0.0, f"Beginning tiled exposure ({total_tiles} tiles)...")

        last_error: Optional[str] = None
        try:
            for tile_idx, (offset_x, offset_y) in enumerate(tile_coords):
                if self.is_aborted:
                    break

                pct_base = tile_idx / total_tiles
                pct_step = 1.0 / total_tiles

                # Step 1: Set the projector to black
                context.projector.set_on(False)

                # Step 2: Move to the position
                tx = start_pos[0] + offset_x
                ty = start_pos[1] + offset_y
                report_progress(
                    pct_base,
                    f"Tile {tile_idx + 1}/{total_tiles} - Moving stage to ({tx:.1f}, {ty:.1f})...",
                )
                jog_op = JogOperation({"x": tx, "y": ty}, relative=False)
                err = self._run_sub_op(jog_op, context, lambda p, m: None)
                if err or self.is_aborted:
                    last_error = err
                    break

                # Step 3: Set the active tile
                context.project.select_tile(tile_idx)

                # Step 4: Source to active layer
                context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

                # Step 5: Set projector to red and run auto focus
                af_enabled = True
                if self.autofocus_config is not None:
                    if isinstance(self.autofocus_config, dict):
                        af_enabled = self.autofocus_config.get("enabled", True)
                    else:
                        af_enabled = getattr(self.autofocus_config, "enabled", True)

                if af_enabled:
                    context.projector.set_color_mode(ColorMode.RED)
                    report_progress(
                        pct_base + 0.3 * pct_step,
                        f"Tile {tile_idx + 1}/{total_tiles} - Autofocusing...",
                    )
                    af_op = AutofocusOperation(config=self.autofocus_config)
                    err = self._run_sub_op(af_op, context, lambda p, m: None)
                    if err or self.is_aborted:
                        last_error = err
                        break

                # Step 6: Then, do an exposure
                report_progress(
                    pct_base + 0.6 * pct_step,
                    f"Tile {tile_idx + 1}/{total_tiles} - Exposing...",
                )
                exp_op = ExposureOperation(
                    layer_index=self.layer_index,
                    config=ExposureOperationConfig(exposure_time=self.settings.exposure_time),
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
                    last_error = err
                    break
        finally:
            context.projector.set_on(False)

        if self.is_aborted:
            report_progress(1.0, "Tiled exposure aborted")
            return "Tiled exposure aborted"
        elif last_error is not None:
            msg = f"Tiled exposure failed: {last_error}"
            report_progress(1.0, msg)
            return msg
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

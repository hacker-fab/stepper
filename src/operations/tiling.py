from typing import Callable, List, Optional, Tuple

from core.chip_project import PatterningSettings
from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation


class LayerExposureOperation(Operation):
    """Unified layer exposure operation that computes the required projector tiles in chip-space
    and executes a step-and-repeat exposure sequence across one or more pattern origins.
    """

    def __init__(
        self,
        layer_index: int,
        settings: Optional[PatterningSettings] = None,
        uv_z_offset: Optional[float] = None,
        origin_mode: str = "auto",
        origin_index: Optional[int] = None,
        autofocus: bool = True,
    ):
        super().__init__("Layer Exposure")
        self.layer_index = int(layer_index)
        self.settings = settings
        self.uv_z_offset = uv_z_offset
        self.origin_mode = origin_mode
        self.origin_index = origin_index
        self.autofocus = bool(autofocus)
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.project is None or self.layer_index < 0 or self.layer_index >= len(context.project.layers):
            msg = "Invalid project or layer index"
            report_progress(1.0, msg)
            return msg

        project = context.project
        chip_progress = project.chip_progress

        # Check multi-layer marker requirement
        if len(project.layers) > 1 and not chip_progress.has_reference_markers():
            msg = (
                "Chip project has multiple layers, but the two chip markers have not been recorded yet. "
                "Please record Marker 1 and Marker 2 using 'Locate Chip Marker' before exposing."
            )
            if context.warning_callback:
                context.warning_callback(msg)
            report_progress(1.0, msg)
            return msg

        if context.camera is not None and getattr(context.camera, "_last_frame_size", None):
            project.camera_size = context.camera._last_frame_size

        project.select_layer(self.layer_index)
        layer = project.layers[self.layer_index]
        effective_settings = self.settings if self.settings is not None else layer.get_effective_settings()

        pos = context.stage.get_position() if context.stage is not None else (0.0, 0.0, 0.0)
        cur_chip_xy = chip_progress.corr_to_chip((float(pos[0]), float(pos[1])))

        # Determine effective physical pattern size (using calculated projector size if -1)
        pw, ph = project.projector_size
        proj_px = 1.0
        if context.machine_config is not None:
            proj_px = float(context.machine_config.optics.projector_pixel_size)
        img = layer.get_pattern_image()
        img_shape = img.shape if img is not None else (ph, pw)
        from lib.tiling import resolve_layer_physical_size
        eff_w_um, eff_h_um = resolve_layer_physical_size(
            img_shape,
            layer.width_um,
            layer.height_um,
            projector_size=(pw, ph),
            projector_pixel_size=proj_px,
        )
        effective_pattern_size = (float(eff_w_um), float(eff_h_um))

        # Resolve target origin(s) and their pattern transforms in chip-space
        origins_with_indices: List[Tuple[int, Tuple[float, float]]] = []
        if self.origin_mode == "current" or (self.layer_index == 0 and self.origin_mode in ("auto", "current")):
            new_idx = chip_progress.add_pattern_origin(cur_chip_xy, size_um=effective_pattern_size)
            origins_with_indices = [(new_idx, cur_chip_xy)]
            if context.event_bus is not None:
                context.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, chip_progress)
        elif (
            self.origin_mode == "index"
            and self.origin_index is not None
            and 0 <= self.origin_index < len(chip_progress.pattern_origins_chip)
        ):
            origins_with_indices = [(self.origin_index, chip_progress.pattern_origins_chip[self.origin_index])]
        elif self.origin_mode in ("all", "auto") and chip_progress.pattern_origins_chip:
            origins_with_indices = list(enumerate(chip_progress.pattern_origins_chip))
        else:
            new_idx = chip_progress.add_pattern_origin(cur_chip_xy, size_um=effective_pattern_size)
            origins_with_indices = [(new_idx, cur_chip_xy)]
            if context.event_bus is not None:
                context.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, chip_progress)

        first_transform = chip_progress.get_pattern_transform_chip(origins_with_indices[0][0])
        tiles, tile_rel_coords = layer.generate_tiles(pattern_transform_chip=first_transform)
        total_tiles_per_origin = len(tiles)
        if total_tiles_per_origin == 0:
            report_progress(1.0, "No tiles to expose")
            return None

        total_steps = len(origins_with_indices) * total_tiles_per_origin
        report_progress(
            0.0,
            f"Beginning layer exposure ({len(origins_with_indices)} origin(s) × {total_tiles_per_origin} tile(s))...",
        )

        last_error: Optional[str] = None
        step_counter = 0
        try:
            for orig_idx, (prog_orig_idx, origin_chip) in enumerate(origins_with_indices):
                if self.is_aborted:
                    break
                orig_transform = chip_progress.get_pattern_transform_chip(prog_orig_idx)
                tiles, tile_rel_coords = layer.generate_tiles(pattern_transform_chip=orig_transform)
                total_tiles_per_origin = len(tiles)
                for tile_idx, (rel_cx, rel_cy) in enumerate(tile_rel_coords):
                    if self.is_aborted:
                        break

                    pct_base = step_counter / max(1, total_steps)
                    pct_step = 1.0 / max(1, total_steps)
                    step_counter += 1

                    prefix_label = (
                        f"Origin {orig_idx + 1}/{len(origins_with_indices)} Tile {tile_idx + 1}/{total_tiles_per_origin}"
                        if len(origins_with_indices) > 1
                        else f"Tile {tile_idx + 1}/{total_tiles_per_origin}"
                    )

                    # Step 1: Set projector output off
                    context.projector.set_on(False)

                    # Step 2: Compute target stage coordinate in corr-stage-space and move
                    target_chip = (origin_chip[0] + rel_cx, origin_chip[1] + rel_cy)
                    tx_corr, ty_corr = chip_progress.chip_to_corr(target_chip)

                    jog_reporter = sub_progress(
                        report_progress,
                        pct_base,
                        pct_base + 0.15 * pct_step,
                        prefix=prefix_label,
                    )
                    report_progress(
                        pct_base,
                        f"{prefix_label} - Moving stage to ({tx_corr:.1f}, {ty_corr:.1f}) µm...",
                    )
                    jog_op = JogOperation({"x": tx_corr, "y": ty_corr}, relative=False)
                    err = self._run_sub_op(jog_op, context, jog_reporter)
                    if err or self.is_aborted:
                        last_error = err
                        break

                    # Step 3: Select tile and set projector image source to active layer
                    project.select_tile(tile_idx)
                    context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

                    # Step 4: Autofocus (if enabled)
                    if self.autofocus:
                        context.projector.set_color_mode(ColorMode.RED)
                        af_reporter = sub_progress(
                            report_progress,
                            pct_base + 0.15 * pct_step,
                            pct_base + 0.60 * pct_step,
                            prefix=prefix_label,
                        )
                        af_op = AutofocusOperation(uv_z_offset=self.uv_z_offset)
                        err = self._run_sub_op(af_op, context, af_reporter)
                        if err or self.is_aborted:
                            last_error = err
                            break

                    # Step 5: Expose tile using ExposureOperation sub-op
                    exp_start_frac = 0.60 if self.autofocus else 0.15
                    exp_reporter = sub_progress(
                        report_progress,
                        pct_base + exp_start_frac * pct_step,
                        pct_base + 1.0 * pct_step,
                        prefix=prefix_label,
                    )
                    exp_time = effective_settings.exposure_time
                    exp_op = ExposureOperation(
                        layer_index=self.layer_index,
                        exposure_time=exp_time,
                        tile_index=tile_idx,
                        record_origin=False,
                        single_fullscreen_tile=False,
                    )
                    err = self._run_sub_op(exp_op, context, exp_reporter)
                    if err or self.is_aborted:
                        last_error = err
                        break

                if last_error is not None or self.is_aborted:
                    break
        finally:
            context.projector.set_on(False)

        if self.is_aborted:
            report_progress(1.0, "Layer exposure aborted")
            return "Layer exposure aborted"
        elif last_error is not None:
            msg = f"Layer exposure failed: {last_error}"
            report_progress(1.0, msg)
            return msg
        else:
            report_progress(1.0, "Layer exposure complete")
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


# Alias for backward compatibility
TiledExposureOperation = LayerExposureOperation


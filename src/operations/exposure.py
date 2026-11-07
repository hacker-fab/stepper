import time
from datetime import datetime
from typing import Callable, Optional

from core.events import ColorMode, Event, ProjectorImageSource
from core.operation import ExecutionContext, Operation


class ExposureOperation(Operation):
    """Exposes a single layer mask for a defined duration."""

    def __init__(
        self,
        layer_index: Optional[int] = None,
        exposure_time: Optional[float] = None,
        tile_index: Optional[int] = None,
        record_origin: bool = False,
        single_fullscreen_tile: bool = False,
    ):
        super().__init__("Layer Exposure" if layer_index is not None else "Exposure")
        self.layer_index = layer_index
        self.exposure_time = exposure_time
        self.tile_index = tile_index
        self.record_origin = bool(record_origin)
        self.single_fullscreen_tile = bool(single_fullscreen_tile)

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.exposure_time is not None:
            duration_ms = float(self.exposure_time)
        elif context.machine_config is not None:
            duration_ms = float(context.machine_config.exposure.default_exposure_time_ms)
        else:
            duration_ms = 8000.0

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
            # Pre-render the tile into cache and configure UV pattern before turning projector on
            if self.single_fullscreen_tile:
                layer.generate_tiles(single_fullscreen_tile=True)
            else:
                layer.generate_tiles()
            tile_idx = self.tile_index if self.tile_index is not None else context.project.active_tile_index
            if layer._tile_cache and (tile_idx < 0 or tile_idx >= len(layer._tile_cache)):
                tile_idx = 0
            context.project.select_tile(tile_idx)
            context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
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
                from core.coordinate_system import (
                    compute_fullscreen_pattern_to_chip_matrix,
                    compute_projector_footprint_in_chip_space,
                )

                sx, sy, sz = float(coords[0]), float(coords[1]), float(coords[2])
                chip_xy = context.project.chip_progress.corr_to_chip((sx, sy))
                chip_coords = (float(chip_xy[0]), float(chip_xy[1]), sz)

                proj_size = (
                    context.projector.projector_size()
                    if context.projector is not None and hasattr(context.projector, "projector_size")
                    else context.project.projector_size
                )
                cam_size = (
                    context.camera._last_frame_size
                    if context.camera is not None and getattr(context.camera, "_last_frame_size", None)
                    else context.project.camera_size
                )
                proj_px = 1.0
                cam_px = 1.0
                proj_to_cam = None
                if context.machine_config is not None:
                    proj_px = float(context.machine_config.optics.projector_pixel_size)
                    cam_px = float(context.machine_config.stage_calibration.camera_pixel_size)
                    proj_to_cam = context.machine_config.optics.proj_to_cam_matrix

                corr_to_chip_mat = context.project.chip_progress.get_corr_to_chip_matrix()
                footprint_chip = compute_projector_footprint_in_chip_space(
                    stage_xy_corr=(sx, sy),
                    projector_size=proj_size,
                    camera_size=cam_size,
                    projector_pixel_size=proj_px,
                    camera_pixel_size=cam_px,
                    proj_to_cam_matrix=proj_to_cam,
                    corr_to_chip_matrix=corr_to_chip_mat,
                )
                record = ExposureRecord(
                    coords=chip_coords,
                    time=start_datetime,
                    duration=elapsed_ms,
                    aborted=self.is_aborted,
                    layer_index=self.layer_index,
                    tile_index=self.tile_index,
                    footprint_chip_corners=footprint_chip,
                )
                context.project.add_exposure_record(record)

                if self.record_origin and not self.is_aborted and self.layer_index is not None and layer is not None:
                    from lib.tiling import resolve_layer_physical_size
                    img = layer.get_pattern_image()
                    img_shape = img.shape if img is not None else (proj_size[1], proj_size[0])
                    eff_w, eff_h = resolve_layer_physical_size(
                        img_shape,
                        layer.width_um,
                        layer.height_um,
                        projector_size=proj_size,
                        projector_pixel_size=proj_px,
                    )
                    transform_chip = None
                    if self.single_fullscreen_tile:
                        transform_chip = compute_fullscreen_pattern_to_chip_matrix(
                            pattern_size_um=(float(eff_w), float(eff_h)),
                            projector_size=proj_size,
                            camera_size=cam_size,
                            projector_pixel_size=proj_px,
                            camera_pixel_size=cam_px,
                            proj_to_cam_matrix=proj_to_cam,
                            corr_to_chip_matrix=corr_to_chip_mat,
                        )
                    context.project.chip_progress.add_pattern_origin(
                        chip_xy,
                        size_um=(float(eff_w), float(eff_h)),
                        transform_chip=transform_chip,
                    )

                if self.single_fullscreen_tile and layer is not None:
                    # Restore standard layer tiling cache so UI previews and multi-tile expose operations remain consistent
                    layer.generate_tiles()

                if context.event_bus is not None:
                    context.event_bus.emit(Event.CHIP_PROGRESS_CHANGED, context.project.chip_progress)

        if self.is_aborted:
            report_progress(1.0, f"Exposure of {layer_name} aborted")
            return f"Exposure of {layer_name} aborted"

        report_progress(1.0, f"Exposure of {layer_name} completed successfully")
        return None

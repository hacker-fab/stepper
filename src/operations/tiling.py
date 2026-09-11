import os
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple
from PIL import Image

from core.chip_project import ExposureLog, PatterningSettings
from core.events import Event, ShownImage
from core.operation import Operation, ExecutionContext
from operations.movement import JogOperation
from operations.autofocus import AutofocusOperation


def split_image_with_overlap(
    image_path: str,
    tile_width: int = 3840,
    tile_height: int = 2160,
    overlap_x: int = 200,
    overlap_y: int = 200,
    output_dir: str = "tiles",
) -> Tuple[int, int, int, Tuple[int, int]]:
    """Takes an arbitrary sized image and splits it into overlapping tiles.

    Returns:
        (x_tiles_count, y_tiles_count, total_tiles_count, (img_w, img_h))
    """
    img = Image.open(image_path)
    img_w, img_h = img.size
    os.makedirs(output_dir, exist_ok=True)

    stride_x = tile_width - overlap_x
    stride_y = tile_height - overlap_y

    # Compute all top-left coordinates
    x_positions: List[int] = []
    y_positions: List[int] = []

    # Horizontal positions
    x = 0
    while True:
        if x + tile_width >= img_w:
            x = max(0, img_w - tile_width)
            x_positions.append(x)
            break
        x_positions.append(x)
        x += stride_x

    # Vertical positions
    y = 0
    while True:
        if y + tile_height >= img_h:
            y = max(0, img_h - tile_height)
            y_positions.append(y)
            break
        y_positions.append(y)
        y += stride_y

    tile_count = 0
    for tile_id_y, top in enumerate(y_positions):
        for tile_id_x, left in enumerate(x_positions):
            right = left + tile_width
            bottom = top + tile_height

            box = (left, top, right, bottom)
            tile = img.crop(box)
            tile.save(os.path.join(output_dir, f"tile_{tile_id_y},{tile_id_x}.png"))
            tile_count += 1

    print(f"X amount = {len(x_positions)}")
    print(f"Y amount = {len(y_positions)}")
    print(f"Saved {tile_count} tiles to {output_dir}")

    return len(x_positions), len(y_positions), tile_count, (img_w, img_h)


def generate_snake_sequence(x_amount: int, y_amount: int) -> List[Tuple[int, int]]:
    """Generates (x_idx, y_idx) tile coordinate list in a boustrophedon (snake) pattern.

    Left to right on even rows, right to left on odd rows.
    """
    coords: List[Tuple[int, int]] = []
    for y_idx in range(y_amount):
        if y_idx % 2 == 0:
            for x_idx in range(x_amount):
                coords.append((x_idx, y_idx))
        else:
            for x_idx in range(x_amount - 1, -1, -1):
                coords.append((x_idx, y_idx))
    return coords


def calculate_tile_position(
    x_start: float,
    y_start: float,
    x_dir: int,
    y_dir: int,
    x_idx: int,
    y_idx: int,
    x_offset: float,
    y_offset: float,
) -> Tuple[float, float]:
    """Calculates target absolute stage coordinates for a given tile index."""
    target_x = x_start + x_dir * x_idx * x_offset
    target_y = y_start + y_dir * y_idx * y_offset
    return target_x, target_y


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

            # Load sliced tile if present, or render layer pattern
            tile_path = f"tiles/tile_{y_idx},{x_idx}.png"
            if os.path.exists(tile_path):
                tile_img = Image.open(tile_path)
            else:
                tile_img = layer.render_pattern(s, context.projector.size())

            # Expose
            report_progress(pct_base, f"Tile {i + 1}/{total_tiles} - Exposing...")
            context.projector.set_mode(ShownImage.PATTERN, tile_img)

            end_t = time.time() + (duration_ms / 1000.0)
            while time.time() < end_t:
                if self.is_aborted:
                    break
                context.delay_func(0.02)

            context.projector.set_mode(ShownImage.CLEAR)

            log = ExposureLog(
                time=datetime.now(),
                path=tile_path if os.path.exists(tile_path) else (layer.pattern_path or ""),
                coords=context.stage.get_position(),
                duration=duration_ms,
                aborted=self.is_aborted,
            )
            layer.exposures.append(log)

        if context.event_bus:
            context.event_bus.emit(Event.PROJECT_CHANGED, context.project)
        if self.is_aborted:
            report_progress(1.0, "Tiled exposure aborted")
            return "Tiled exposure aborted"
        else:
            report_progress(1.0, "Tiled exposure complete")
            return None

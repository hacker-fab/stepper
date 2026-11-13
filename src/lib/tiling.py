"""Placement of tiled patterns: alignment-cross geometry, where each tile goes, seam checks, time budget.

Projector/pattern pixels ("frame px") are the pixels of the 3840x2160 tile images.  The placement puts a
rows x cols pattern in the middle of a slightly bigger area so that the crosses neighbouring tiles share
overlap exactly.  No GUI code, so it is tested headless (see tiling_selftest.py).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .img import exposure_mask

# --------------------------------------------------------------------------------------
# Projector (from the old UI; NOT measured on this machine unless noted)
# --------------------------------------------------------------------------------------


PROJECTOR_SIZE_PX = (3840, 2160)  # tile size used by TilingFrame / the sample patterns
# DLP471TP + 10x objective: 1037 x 583 um per tile (comments in gui.py TilingFrame/MapFrame)
PROJ_UM_PER_PX = (1037.0 / 3840.0, 583.0 / 2160.0)


# sample patterns (assets/tiling_samples), reading order TL, TR, BL, BR.  Patterns 3, 2, 1 are the TR, BL and BR
# tiles (their crosses only line up in that arrangement); the top-left tile was not supplied, so it is crosses-only.
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "tiling_samples")
SAMPLE_SET = ["pattern_4_crosses_only.png", "pattern_3.png", "pattern_2.png", "pattern_1.png"]


# --------------------------------------------------------------------------------------
# Cross layout inputs
# --------------------------------------------------------------------------------------


# PLEASE KEEP THE CROSSES CONSISTENT.  Every pattern tile must carry its four alignment crosses
# with the SAME size and at the SAME distances from the frame edges, because the placement map
# overlaps neighbouring tiles by exactly these distances (a 2 px difference between tiles is a
# 2 px stitching error on the chip).  Please design every pattern with identical:
#   * cross width and height,
#   * distance from the left / right frame edge, and from the top / bottom frame edge.
# Enter the values below (frame px, 3840x2160 tiles).  Distances are measured from the frame edge
# to the nearest edge of the cross box.  The current values are the ones measured from the
# sample patterns pattern_1/2/3.png.  The picker and approval window warn when a chosen pattern
# disagrees with them, and they are used as the layout when a pattern's crosses cannot be found
# (for example a blank first-layer pattern).
CROSS_DIST_LEFT_PX = 17  # frame left edge  -> left cross box
CROSS_WIDTH_PX = 314  # width of every cross box
CROSS_DIST_TOP_PX = 17  # frame top edge   -> top cross box
CROSS_DIST_RIGHT_PX = 17  # frame right edge -> right cross box
CROSS_HEIGHT_PX = 198  # height of every cross box
CROSS_DIST_BOTTOM_PX = 17  # frame bottom edge -> bottom cross box
CROSS_SPEC_TOL_PX = 2  # how far a measured pattern may differ before it is flagged


# --------------------------------------------------------------------------------------
# Cross geometry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossGeometry:
    """Measured position of the four alignment crosses in one pattern frame (frame px)."""

    frame_w: int
    frame_h: int
    cross_w: int
    cross_h: int
    tl: Tuple[int, int]  # top-left pixel of each cross's bounding box
    tr: Tuple[int, int]
    bl: Tuple[int, int]
    br: Tuple[int, int]
    center_in_cross: Tuple[float, float]  # where the two bars meet, relative to the bbox corner

    @property
    def pitch_x(self) -> int:
        """Left cross -> right cross distance (bbox left edge to bbox left edge)."""
        return self.tr[0] - self.tl[0]

    @property
    def pitch_y(self) -> int:
        return self.bl[1] - self.tl[1]

    @property
    def overlap_x(self) -> int:
        """How much neighbouring frames overlap horizontally so that their crosses coincide."""
        return self.frame_w - self.pitch_x

    @property
    def overlap_y(self) -> int:
        return self.frame_h - self.pitch_y

    def corner(self, name: str) -> Tuple[int, int]:
        return getattr(self, name.lower())

    def footprint(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Part of a frame (x0, y0, x1, y1, half-open) that this tile is responsible for.

        Left/top-most tiles run from the outer edge of their first cross; a tile with a
        neighbour to its left (above) drops its own left (top) cross because the neighbour's
        right (bottom) cross is already there.  Right/bottom crosses are always included.
        """
        x0 = self.tl[0] + (self.cross_w if col > 0 else 0)
        y0 = self.tl[1] + (self.cross_h if row > 0 else 0)
        x1 = self.tr[0] + self.cross_w
        y1 = self.bl[1] + self.cross_h
        return (x0, y0, x1, y1)

    def summary(self) -> List[str]:
        return [
            f"frame {self.frame_w}x{self.frame_h} px",
            f"cross box {self.cross_w}x{self.cross_h} px, {self.tl[0]} px from left, {self.tl[1]} px from top",
            f"cross-to-cross pitch: x {self.pitch_x} px, y {self.pitch_y} px (bbox edge to same edge)",
            f"gap between crosses: x {self.tr[0] - (self.tl[0] + self.cross_w)} px, "
            f"y {self.bl[1] - (self.tl[1] + self.cross_h)} px",
            f"neighbour overlap: x {self.overlap_x} px, y {self.overlap_y} px",
        ]


def measure_cross_geometry(img: np.ndarray) -> CrossGeometry:
    """Find the four corner crosses (small shapes that do not touch the frame border)."""
    mask = exposure_mask(img).astype(np.uint8)
    h, w = mask.shape
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    corners: Dict[str, List[Tuple[int, int, int, int, int]]] = {}
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if x > 0 and y > 0 and x + bw < w and y + bh < h and area < 0.01 * w * h:
            key = ("T" if y + bh / 2 < h / 2 else "B") + ("L" if x + bw / 2 < w / 2 else "R")
            corners.setdefault(key, []).append((x, y, bw, bh, area))
    if set(corners) != {"TL", "TR", "BL", "BR"} or any(len(v) != 1 for v in corners.values()):
        found = {k: len(v) for k, v in corners.items()}
        raise ValueError(f"Expected exactly one cross in each corner, found {found or 'none'}")
    sizes = {(v[0][2], v[0][3]) for v in corners.values()}
    if len(sizes) != 1:
        raise ValueError(f"The four crosses differ in size: {sorted(sizes)}")
    cw, ch = sizes.pop()
    pos = {k: (v[0][0], v[0][1]) for k, v in corners.items()}
    # where the bars meet: columns/rows that span the full cross
    x, y = pos["TL"]
    patch = mask[y : y + ch, x : x + cw]
    full_cols = np.where(patch.sum(axis=0) == patch.sum(axis=0).max())[0]
    full_rows = np.where(patch.sum(axis=1) == patch.sum(axis=1).max())[0]
    center = (float(full_cols.mean()), float(full_rows.mean()))
    return CrossGeometry(w, h, cw, ch, pos["TL"], pos["TR"], pos["BL"], pos["BR"], center)


def spec_geometry(frame_size: Tuple[int, int] = PROJECTOR_SIZE_PX) -> CrossGeometry:
    """The cross layout typed in at the top of this file (CROSS_* variables)."""
    fw, fh = frame_size
    left, top = CROSS_DIST_LEFT_PX, CROSS_DIST_TOP_PX
    right_x = fw - CROSS_DIST_RIGHT_PX - CROSS_WIDTH_PX
    bottom_y = fh - CROSS_DIST_BOTTOM_PX - CROSS_HEIGHT_PX
    return CrossGeometry(
        fw, fh, CROSS_WIDTH_PX, CROSS_HEIGHT_PX,
        (left, top), (right_x, top), (left, bottom_y), (right_x, bottom_y),
        ((CROSS_WIDTH_PX - 1) / 2.0, (CROSS_HEIGHT_PX - 1) / 2.0),
    )


def spec_mismatches(geo: CrossGeometry, tol_px: int = CROSS_SPEC_TOL_PX) -> List[str]:
    """Human-readable differences between a measured pattern and the CROSS_* inputs (empty = ok)."""
    spec = spec_geometry((geo.frame_w, geo.frame_h))
    pairs = [
        ("cross width", geo.cross_w, spec.cross_w),
        ("cross height", geo.cross_h, spec.cross_h),
        ("distance from left edge", geo.tl[0], spec.tl[0]),
        ("distance from top edge", geo.tl[1], spec.tl[1]),
        ("distance from right edge", geo.frame_w - geo.tr[0] - geo.cross_w, CROSS_DIST_RIGHT_PX),
        ("distance from bottom edge", geo.frame_h - geo.bl[1] - geo.cross_h, CROSS_DIST_BOTTOM_PX),
    ]
    return [f"{n}: pattern has {got} px, CROSS_* inputs say {want} px" for n, got, want in pairs if abs(got - want) > tol_px]


def cross_geometry_for(img: np.ndarray) -> CrossGeometry:
    """Measured layout of a pattern, or the typed-in layout when no crosses can be found in it."""
    try:
        return measure_cross_geometry(img)
    except ValueError:
        return spec_geometry((img.shape[1], img.shape[0]))


def same_layout(a: CrossGeometry, b: CrossGeometry) -> bool:
    """Same frame size, cross size and cross positions (the sub-pixel bar centre may differ)."""
    key = lambda g: (g.frame_w, g.frame_h, g.cross_w, g.cross_h, g.tl, g.tr, g.bl, g.br)
    return key(a) == key(b)


def cross_only_image(img: np.ndarray, geo: CrossGeometry) -> np.ndarray:
    """Copy of a pattern with everything except the four corner crosses blanked.  Used as a
    stand-in for a missing tile: it keeps the seams consistent without implying any content."""
    out = np.zeros_like(img)
    for name in ("tl", "tr", "bl", "br"):
        x, y = geo.corner(name)
        out[y : y + geo.cross_h, x : x + geo.cross_w] = img[y : y + geo.cross_h, x : x + geo.cross_w]
    return out


def cross_patch(img: np.ndarray, geo: CrossGeometry, corner: str) -> np.ndarray:
    x, y = geo.corner(corner)
    return exposure_mask(img)[y : y + geo.cross_h, x : x + geo.cross_w]


def seam_mismatch(a: np.ndarray, b: np.ndarray, geo: CrossGeometry, direction: str) -> int:
    """Differing pixels between the crosses that must coincide when tile ``a`` is placed left of
    (direction 'h') or above (direction 'v') tile ``b``.  0 means they overlap perfectly."""
    if direction == "h":
        pairs = [("tr", "tl"), ("br", "bl")]
    elif direction == "v":
        pairs = [("bl", "tl"), ("br", "tr")]
    else:
        raise ValueError("direction must be 'h' or 'v'")
    return int(
        sum((cross_patch(a, geo, ca) != cross_patch(b, geo, cb)).sum() for ca, cb in pairs)
    )


def check_seams(images: Sequence[np.ndarray], rows: int, cols: int, geo: CrossGeometry) -> List[dict]:
    """Seam report for a rows x cols grid of images given in reading order."""
    out = []
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            if c + 1 < cols:
                m = seam_mismatch(images[s], images[s + 1], geo, "h")
                out.append({"a": s, "b": s + 1, "dir": "h", "mismatch_px": m, "ok": m == 0})
            if r + 1 < rows:
                m = seam_mismatch(images[s], images[s + cols], geo, "v")
                out.append({"a": s, "b": s + cols, "dir": "v", "mismatch_px": m, "ok": m == 0})
    return out


# --------------------------------------------------------------------------------------
# Placement map
# --------------------------------------------------------------------------------------


@dataclass
class PlacedTile:
    slot: int  # 0-based, reading order
    row: int
    col: int
    crop: Tuple[int, int, int, int]  # region of the frame that is exposed (frame px, half-open)
    frame_origin: Tuple[int, int]  # area px of the frame's pixel (0, 0)
    dest: Tuple[int, int, int, int]  # area px box the crop lands on (half-open)
    center_um: Tuple[float, float]  # frame centre relative to the area centre (x right, y down)


@dataclass
class PlacementMap:
    geo: CrossGeometry
    rows: int
    cols: int
    area_rows: int
    area_cols: int
    mode: str  # 'trimmed' (split at the cross footprints) or 'whole' (full frames, crosses overlap)
    um_per_px: Tuple[float, float]
    nudge_um: Tuple[float, float]
    area_box: Tuple[int, int, int, int]  # the white area, area px, origin at its top-left
    tiles: List[PlacedTile]

    @property
    def area_size_px(self) -> Tuple[int, int]:
        return (self.area_box[2] - self.area_box[0], self.area_box[3] - self.area_box[1])

    @property
    def area_size_um(self) -> Tuple[float, float]:
        w, h = self.area_size_px
        return (w * self.um_per_px[0], h * self.um_per_px[1])

    @property
    def step_um(self) -> Tuple[float, float]:
        """Stage step between neighbouring tiles."""
        return (self.geo.pitch_x * self.um_per_px[0], self.geo.pitch_y * self.um_per_px[1])

    def stage_targets(
        self, area_center_um: Tuple[float, float], sign: Tuple[int, int] = (1, -1)
    ) -> List[Tuple[float, float]]:
        """Stage (x, y) in um at which each tile's frame centre must sit, reading order."""
        cx, cy = area_center_um
        return [(cx + sign[0] * t.center_um[0], cy + sign[1] * t.center_um[1]) for t in self.tiles]

    def report(self) -> List[str]:
        w, h = self.area_size_px
        wu, hu = self.area_size_um
        lines = [
            f"mode: {self.mode}   pattern {self.rows}x{self.cols} in area {self.area_rows}x{self.area_cols}",
            f"area: {w}x{h} px = {wu:.0f}x{hu:.0f} um   step between tiles: "
            f"{self.step_um[0]:.1f} um (x), {self.step_um[1]:.1f} um (y)",
        ]
        for t in self.tiles:
            x0, y0, x1, y1 = t.crop
            lines.append(
                f"tile {t.slot + 1} (row {t.row + 1}, col {t.col + 1}): crop x {x0}-{x1} y {y0}-{y1} "
                f"= {x1 - x0}x{y1 - y0} px, centre {t.center_um[0]:+.1f}, {t.center_um[1]:+.1f} um from area centre"
            )
        return lines


def tile_pattern(img: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    """The pattern a tile actually exposes: the frame with everything outside its footprint ``crop`` black."""
    x0, y0, x1, y1 = crop
    out = np.zeros_like(img)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out


def build_placement_map(
    geo: CrossGeometry,
    pattern_rows: int = 2,
    pattern_cols: int = 2,
    area_rows: int = 3,
    area_cols: int = 3,
    mode: str = "trimmed",
    um_per_px: Tuple[float, float] = PROJ_UM_PER_PX,
    nudge_um: Tuple[float, float] = (0.0, 0.0),
) -> PlacementMap:
    """Place a pattern_rows x pattern_cols pattern in the middle of an area_rows x area_cols area.

    Tiles are laid out at the measured cross pitch so neighbouring crosses coincide.  With a 2x2
    pattern in a 3x3 area every tile sits half a pitch in from the area edge.
    """
    if mode not in ("trimmed", "whole"):
        raise ValueError("mode must be 'trimmed' or 'whole'")
    if pattern_rows > area_rows or pattern_cols > area_cols:
        raise ValueError("pattern does not fit in the area")
    px, py = geo.pitch_x, geo.pitch_y
    fx0, fy0, fx1, fy1 = geo.footprint(0, 0)
    area_w = (fx1 - fx0) + (area_cols - 1) * px
    area_h = (fy1 - fy0) + (area_rows - 1) * py
    shift_c = (area_cols - pattern_cols) / 2.0
    shift_r = (area_rows - pattern_rows) / 2.0
    nx, ny = nudge_um[0] / um_per_px[0], nudge_um[1] / um_per_px[1]
    tiles: List[PlacedTile] = []
    for i in range(pattern_rows):
        for j in range(pattern_cols):
            ox = int(round((j + shift_c) * px - geo.tl[0] + nx))
            oy = int(round((i + shift_r) * py - geo.tl[1] + ny))
            crop = geo.footprint(i, j) if mode == "trimmed" else (0, 0, geo.frame_w, geo.frame_h)
            dest = (ox + crop[0], oy + crop[1], ox + crop[2], oy + crop[3])
            cx = (ox + geo.frame_w / 2.0 - area_w / 2.0) * um_per_px[0]
            cy = (oy + geo.frame_h / 2.0 - area_h / 2.0) * um_per_px[1]
            tiles.append(PlacedTile(i * pattern_cols + j, i, j, crop, (ox, oy), dest, (cx, cy)))
    return PlacementMap(
        geo, pattern_rows, pattern_cols, area_rows, area_cols, mode, um_per_px, nudge_um,
        (0, 0, area_w, area_h), tiles,
    )


# colours (BGR)
_BLUE = np.array([215, 70, 0], np.float32)
TILE_COLORS = [(40, 140, 255), (60, 170, 60), (200, 90, 200), (190, 150, 30), (80, 80, 220), (30, 190, 190)]


def render_placement(
    pm: PlacementMap,
    images: Sequence[np.ndarray],
    scale: float = 0.125,
    background: Optional[np.ndarray] = None,
    show_labels: bool = True,
    show_crosses: bool = True,
    show_outlines: bool = True,
    show_grid: bool = True,
    pad_px: int = 600,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Draw the placement on the (white) area.  Returns (BGR image, canvas origin in area px).

    ``background``: optional BGR/gray image already at preview scale covering exactly the area.
    """
    ax0, ay0, ax1, ay1 = pm.area_box
    bx0 = min([ax0] + [t.dest[0] for t in pm.tiles]) - pad_px
    by0 = min([ay0] + [t.dest[1] for t in pm.tiles]) - pad_px
    bx1 = max([ax1] + [t.dest[2] for t in pm.tiles]) + pad_px
    by1 = max([ay1] + [t.dest[3] for t in pm.tiles]) + pad_px
    W = int(math.ceil((bx1 - bx0) * scale))
    H = int(math.ceil((by1 - by0) * scale))

    def sx(v: float) -> int:
        return int(round((v - bx0) * scale))

    def sy(v: float) -> int:
        return int(round((v - by0) * scale))

    canvas = np.full((H, W, 3), 232, np.uint8)  # outside the area
    x0, y0, x1, y1 = sx(ax0), sy(ay0), sx(ax1), sy(ay1)
    canvas[y0:y1, x0:x1] = 255  # the white area
    if background is not None:
        bg = background if background.ndim == 3 else cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        bg = cv2.resize(bg, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        canvas[y0:y1, x0:x1] = bg

    # exposure coverage of all tiles (union, so 'whole' mode overlaps merge)
    cover = np.zeros((H, W), np.float32)
    for t in pm.tiles:
        mask = exposure_mask(images[t.slot])
        c0, r0, c1, r1 = t.crop
        part = mask[r0:r1, c0:c1].astype(np.float32)
        dx0, dy0, dx1, dy1 = sx(t.dest[0]), sy(t.dest[1]), sx(t.dest[2]), sy(t.dest[3])
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        small = cv2.resize(part, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_AREA)
        # thin features vanish at preview scale, keep them visible
        small = np.clip(small * 3.0, 0, 1)
        region = cover[dy0:dy1, dx0:dx1]
        np.maximum(region, small, out=region)
    a = cover[..., None]
    canvas = (canvas.astype(np.float32) * (1 - a) + _BLUE * a).astype(np.uint8)

    if show_grid:  # the 3x3 cells of the area
        fx0, fy0, fx1, fy1 = pm.geo.footprint(0, 0)
        for k in range(1, pm.area_cols):
            X = sx(ax0 + (fx1 - fx0) + (k - 1) * pm.geo.pitch_x)
            cv2.line(canvas, (X, y0), (X, y1), (170, 170, 170), 1, cv2.LINE_AA)
        for k in range(1, pm.area_rows):
            Y = sy(ay0 + (fy1 - fy0) + (k - 1) * pm.geo.pitch_y)
            cv2.line(canvas, (x0, Y), (x1, Y), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)

    geo = pm.geo
    for t in pm.tiles:
        col = TILE_COLORS[t.slot % len(TILE_COLORS)]
        ox, oy = t.frame_origin
        if show_outlines:
            cv2.rectangle(canvas, (sx(t.dest[0]), sy(t.dest[1])), (sx(t.dest[2]), sy(t.dest[3])), col, 2, cv2.LINE_AA)
        if show_crosses:
            for name in ("tl", "tr", "bl", "br"):
                cx, cy = geo.corner(name)
                cv2.rectangle(
                    canvas,
                    (sx(ox + cx), sy(oy + cy)),
                    (sx(ox + cx + geo.cross_w), sy(oy + cy + geo.cross_h)),
                    (0, 0, 230), 1, cv2.LINE_AA,
                )
        if show_labels:
            label = f"{t.slot + 1}"
            tx = sx((t.dest[0] + t.dest[2]) / 2) - 8
            ty = sy((t.dest[1] + t.dest[3]) / 2) + 8
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2, cv2.LINE_AA)
    return canvas, (bx0, by0)


def area_for_tiles(
    tile_rows: int, tile_cols: int, geo: Optional[CrossGeometry] = None, mode: str = "trimmed"
) -> PlacementMap:
    """The area a tile_rows x tile_cols pattern needs: one more complete frame than tiles in each direction
    (2x2 tiles -> 3x3 frames), with the pattern in the middle.  Uses the CROSS_* inputs unless a measured
    geometry is given, so it works before any pattern has been chosen."""
    return build_placement_map(geo or spec_geometry(), tile_rows, tile_cols, tile_rows + 1, tile_cols + 1, mode=mode)


# --------------------------------------------------------------------------------------
# Time budget (the 5 min map / 10 min tiling / 20 min total target)
# --------------------------------------------------------------------------------------


@dataclass
class TilingTimeModel:
    """Per-tile overhead in seconds.  All of these are ASSUMPTIONS until timed on the machine."""

    move_s: float = 1.0
    settle_s: float = 2.5
    red_autofocus_s: float = 10.0
    z_offset_delay_s: float = 0.5  # the 0.5 s non_blocking_delay in the old tiling loop
    uv_autofocus_s: float = 10.0
    align_s: float = 2.0

    @property
    def overhead_s(self) -> float:
        return self.move_s + self.settle_s + self.red_autofocus_s + self.z_offset_delay_s + self.uv_autofocus_s + self.align_s


def budget_report(
    map_s: float,
    n_tiles: int = 4,
    model: TilingTimeModel = TilingTimeModel(),
    exposure_s: Optional[float] = None,
    map_limit_s: float = 300.0,
    tile_limit_s: float = 600.0,
    total_limit_s: float = 1200.0,
) -> dict:
    overhead = n_tiles * model.overhead_s
    max_exposure = max(0.0, (tile_limit_s - overhead) / n_tiles)
    out = {
        "map_s": map_s,
        "map_limit_s": map_limit_s,
        "map_ok": map_s <= map_limit_s,
        "tile_overhead_s": overhead,
        "max_exposure_per_tile_s": max_exposure,
        "tile_limit_s": tile_limit_s,
        "total_limit_s": total_limit_s,
    }
    if exposure_s is not None:
        tile_s = overhead + n_tiles * exposure_s
        out.update(tile_s=tile_s, tile_ok=tile_s <= tile_limit_s, total_s=map_s + tile_s, total_ok=map_s + tile_s <= total_limit_s)
    return out

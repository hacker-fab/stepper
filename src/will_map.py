"""Will Map: camera map of the exposure area, plus the placement map for tiled patterns.

Two stages, both free of any GUI code so they can be tested headless:

1. ``will_map()``    snake-scans a grid of camera frames over the area, registers each frame to
                     its neighbours, and checks that "left" and "top" agree (loop closure) on the
                     second tile of every new row.
2. ``build_placement_map()``  works out where each pattern tile goes inside the area, using the
                     measured cross-to-cross distances so shared crosses overlap exactly.

Conventions
-----------
* Projector/pattern pixels ("frame px") are the pixels of the 3840x2160 tile images.
* Camera pixels are the pixels of the camera snapshot (1920x1080 on this machine).
* Mosaic coordinates have x to the right and y DOWN.  Stage coordinates follow the old UI:
  stage +x = right, stage +y = up, so ``sign = (+1, -1)``.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np


def status(msg: str) -> None:
    """One status line for the terminal (mapping, patterning and stitching all report through this)."""
    print(f"[Will Map] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Focus search (Toby's autofocus: golden-section search on the sharpness of a projected ArUco grid)
# --------------------------------------------------------------------------------------
# (grid_n, +- sweep in um, stop when the interval is this narrow in um): his defaults, coarse then fine
FOCUS_STAGES = ((2, 250.0, 5.0), (4, 10.0, 0.5))
FOCUS_SETTLE_S = 0.5  # wait after every z move before grabbing (his default is 1.0 s; the old UI sweep waits 0.5 s)


def focus_score(img: Optional[np.ndarray], kernel_size: int = 9) -> float:
    """Sharpness = variance of the Sobel gradient magnitude on a half-size grey image (Toby's compute_focus_score)."""
    if img is None or getattr(img, "size", 0) == 0:
        return 0.0
    g = img
    if g.shape[0] * 0.5 >= kernel_size and g.shape[1] * 0.5 >= kernel_size:
        g = cv2.resize(g, None, fx=0.5, fy=0.5)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)  # camera_image is RGB in the old UI
    if g.shape[0] < kernel_size or g.shape[1] < kernel_size:
        return 0.0
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return float(np.hypot(sx, sy).var())


def aruco_grid(grid_n: int, canvas_size: Tuple[int, int]) -> np.ndarray:
    """grid_n x grid_n ArUco tags (DICT_4X4_250, white quiet zones on black), as in Toby's generate_ar_tag_grid.
    ``canvas_size`` is (width, height) of the projector; returns a grey uint8 image of that size."""
    width, height = canvas_size
    grid_n = max(1, int(grid_n))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    mx, my = int(width * 0.05), int(height * 0.05)
    cell_w, cell_h = (width - 2 * mx) / grid_n, (height - 2 * my) / grid_n
    marker = max(12, int(min(cell_w, cell_h) * 0.65))
    quiet = max(4, int(marker * 0.2))
    box = marker + 2 * quiet
    canvas = np.zeros((height, width), np.uint8)
    for r in range(grid_n):
        for c in range(grid_n):
            cx, cy = int(mx + (c + 0.5) * cell_w), int(my + (r + 0.5) * cell_h)
            x1, y1 = cx - box // 2, cy - box // 2
            if x1 < 0 or y1 < 0 or x1 + box > width or y1 + box > height:
                continue
            canvas[y1 : y1 + box, x1 : x1 + box] = 255
            canvas[y1 + quiet : y1 + quiet + marker, x1 + quiet : x1 + quiet + marker] = cv2.aruco.generateImageMarker(
                dictionary, (r * grid_n + c) % 250, marker
            )
    return canvas


def golden_section_focus(
    measure: Callable[[float], Optional[float]],
    z_centre: float,
    half_range: float,
    threshold: float,
    max_iterations: int = 10,
) -> Optional[Tuple[float, float, Dict[float, float]]]:
    """Golden-section search for the z with the highest sharpness in [z_centre - half_range, z_centre + half_range].

    This is the search in Toby's MaximizeImageSharpnessOperation: two probes, then each step throws away the
    outer part of the interval that cannot hold the maximum and re-uses the other probe, so every step costs
    ONE new z move + photo.  It stops when the interval is at most ``threshold`` wide.
    ``measure(z)`` moves to z, waits, grabs and returns the score (None = the move failed, search aborted).
    Returns (best_z, best_score, {z: score}) or None.
    """
    inv_phi = (5.0 ** 0.5 - 1.0) / 2.0
    a, b = z_centre - abs(half_range), z_centre + abs(half_range)
    if b <= a:
        b = a + threshold
    seen: Dict[float, float] = {}

    def probe(z: float) -> Optional[float]:
        key = round(float(z), 4)
        if key not in seen:
            v = measure(z)
            if v is None:
                return None
            seen[key] = v
        return seen[key]

    x1, x2 = a + (1.0 - inv_phi) * (b - a), a + inv_phi * (b - a)
    s1, s2 = probe(x1), probe(x2)
    if s1 is None or s2 is None:
        return None
    for _ in range(max_iterations):
        if (b - a) <= threshold:
            break
        if s1 < s2:
            a, x1, s1 = x1, x2, s2
            x2 = a + inv_phi * (b - a)
            s2 = probe(x2)
            if s2 is None:
                return None
        else:
            b, x2, s2 = x2, x1, s1
            x1 = a + (1.0 - inv_phi) * (b - a)
            s1 = probe(x1)
            if s1 is None:
                return None
    best = max(seen, key=lambda k: seen[k])
    return best, seen[best], seen

# --------------------------------------------------------------------------------------
# Machine constants (from the old UI; NOT measured on this machine unless noted)
# --------------------------------------------------------------------------------------
PROJECTOR_SIZE_PX = (3840, 2160)  # tile size used by TilingFrame / the sample patterns
# DLP471TP + 10x objective: 1037 x 583 um per tile (comments in gui.py TilingFrame/MapFrame)
PROJ_UM_PER_PX = (1037.0 / 3840.0, 583.0 / 2160.0)
CAMERA_SIZE_PX = (1920, 1080)  # "each snapshot captured is 1920 x 1080 pixels" (gui.py)
DEFAULT_CAM_UM_PER_PX = 0.678  # measured by the user

Tile = Tuple[int, int]  # (row, col)

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
# Pattern images and cross geometry
# --------------------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """cv2.imread that also works with non-ASCII Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def exposure_mask(img: np.ndarray) -> np.ndarray:
    """True where the pattern exposes (any colour channel bright). Patterns are blue on black."""
    a = np.asarray(img)
    if a.dtype == bool:
        return a
    if a.ndim == 3:
        a = a[..., :3].max(axis=2)
    return a > 127


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
_TILE_COLORS = [(40, 140, 255), (60, 170, 60), (200, 90, 200), (190, 150, 30), (80, 80, 220), (30, 190, 190)]


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
        col = _TILE_COLORS[t.slot % len(_TILE_COLORS)]
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


def background_from_map(mr: "MapResult", pm: PlacementMap, scale: float) -> np.ndarray:
    """The camera mosaic cut to the placement area, at preview scale (BGR), for drawing underneath
    the placement.  The area is centred on the stage position the map was centred on."""
    out_w = max(1, int(round(pm.area_size_px[0] * scale)))
    out_h = max(1, int(round(pm.area_size_px[1] * scale)))
    cx, cy = mr.stage_to_px(*mr.area_center_um)
    w_cam = pm.area_size_um[0] / mr.cfg.um_per_px
    f = out_w / w_cam
    small = cv2.resize(mr.mosaic, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    x0 = int(round(cx * f - out_w / 2.0))
    y0 = int(round(cy * f - out_h / 2.0))
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(small.shape[1], x0 + out_w), min(small.shape[0], y0 + out_h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = small[sy0:sy1, sx0:sx1]
    return out


# --------------------------------------------------------------------------------------
# Registration (ZNCC, coarse-to-fine, with a stage-position prior)
# --------------------------------------------------------------------------------------
def prepare_frame(img: np.ndarray) -> np.ndarray:
    """Gray float32 with the slow illumination gradient removed (cheap, done at 1/8 scale)."""
    a = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = a.astype(np.float32)
    h, w = f.shape
    small = cv2.resize(f, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (w, h), interpolation=cv2.INTER_LINEAR)
    return f - bg


@dataclass
class Registration:
    shift: Tuple[float, float]  # where the new frame's origin sits in the reference frame (px)
    peak: float  # ZNCC peak, 1.0 = perfect
    second: float  # best ZNCC away from the peak; a small gap means an ambiguous match
    ok: bool
    reason: str = ""


def register_pair(
    ref: np.ndarray,
    new: np.ndarray,
    predicted: Tuple[float, float],
    search_px: int = 96,
    down: int = 4,
    min_peak: float = 0.5,
    min_gap: float = 0.05,
    min_template: int = 96,
    prefer_prior: bool = False,
) -> Registration:
    """Measure where ``new`` sits in ``ref`` (both from prepare_frame), searching +-search_px
    around the stage-predicted position.

    ``prefer_prior``: when the best peak is not clearly better than a rival (repeating features such
    as concentric rings), pick the near-equal peak closest to the stage prediction instead of refusing.
    Only for inspection scans (finished piece): the stage is good to a few um, far less than the period.
    """
    H, W = ref.shape
    tx, ty = int(round(predicted[0])), int(round(predicted[1]))
    R = int(math.ceil(search_px / down) * down)
    bx0, bx1 = max(0, -tx), min(W, W - tx)
    by0, by1 = max(0, -ty), min(H, H - ty)
    x0, y0 = bx0 + R, by0 + R
    w = ((bx1 - R) - x0) // down * down
    h = ((by1 - R) - y0) // down * down
    if w < min_template or h < min_template:
        return Registration(predicted, 0.0, 0.0, False, "overlap too small for the search window")
    templ = new[y0 : y0 + h, x0 : x0 + w]
    rx, ry = x0 + tx - R, y0 + ty - R  # search region origin in ref
    region = ref[ry : ry + h + 2 * R, rx : rx + w + 2 * R]

    # coarse
    d = down
    tc = cv2.resize(templ, (w // d, h // d), interpolation=cv2.INTER_AREA)
    rc = cv2.resize(region, ((w + 2 * R) // d, (h + 2 * R) // d), interpolation=cv2.INTER_AREA)
    res = np.nan_to_num(cv2.matchTemplate(rc, tc, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
    v, u = np.unravel_index(int(np.argmax(res)), res.shape)
    peak_c = float(res[v, u])
    sup = res.copy()
    sup[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4] = -1.0
    second = float(sup.max())
    resolved = ""
    if prefer_prior and (peak_c - second) < min_gap:
        picks = [(peak_c, int(v), int(u))]
        sup2 = res.copy()
        sup2[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4] = -1.0
        for _ in range(6):
            vv, uu = np.unravel_index(int(np.argmax(sup2)), sup2.shape)
            sc = float(sup2[vv, uu])
            if sc < peak_c - 0.1:
                break
            picks.append((sc, int(vv), int(uu)))
            sup2[max(0, vv - 3) : vv + 4, max(0, uu - 3) : uu + 4] = -1.0
        c0 = R // d
        best = min(picks, key=lambda q: (q[1] - c0) ** 2 + (q[2] - c0) ** 2)
        resolved = f"ambiguous (best {peak_c:.2f} vs next {second:.2f}); took the peak nearest the stage position"
        rival = max((q[0] for q in picks if q is not best), default=second)
        peak_c, v, u, second = best[0], best[1], best[2], rival

    # fine: +-(d+1) px around the coarse peak, full resolution
    m = d + 1
    fx, fy = rx + u * d, ry + v * d  # template top-left in ref at the coarse peak
    sx0, sy0 = max(0, fx - m), max(0, fy - m)
    sx1, sy1 = min(W, fx + w + m), min(H, fy + h + m)
    fine = np.nan_to_num(cv2.matchTemplate(ref[sy0:sy1, sx0:sx1], templ, cv2.TM_CCOEFF_NORMED), nan=-1.0)
    vf, uf = np.unravel_index(int(np.argmax(fine)), fine.shape)
    peak = float(fine[vf, uf])

    def sub(a: float, b: float, c: float) -> float:
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    dx = sub(fine[vf, uf - 1], fine[vf, uf], fine[vf, uf + 1]) if 0 < uf < fine.shape[1] - 1 else 0.0
    dy = sub(fine[vf - 1, uf], fine[vf, uf], fine[vf + 1, uf]) if 0 < vf < fine.shape[0] - 1 else 0.0
    px_ref, py_ref = sx0 + uf + dx, sy0 + vf + dy  # template top-left in ref
    shift = (px_ref - x0, py_ref - y0)
    ok = peak >= min_peak and ((peak_c - second) >= min_gap or bool(resolved))
    reason = resolved if (ok and resolved) else ""
    if peak < min_peak:
        reason = f"weak match (peak {peak:.2f} < {min_peak:.2f}); is there anything to register on?"
    elif peak_c - second < min_gap:
        reason = f"ambiguous match (best {peak_c:.2f} vs next {second:.2f}); repeating features?"
    return Registration(shift, peak, second, ok, reason)


# --------------------------------------------------------------------------------------
# The map function
# --------------------------------------------------------------------------------------
class Rig(Protocol):
    """What will_map needs from the machine.  VirtualRig and ModelRig implement it."""

    def move_to(self, x_um: float, y_um: float) -> None: ...
    def settle(self, seconds: float) -> None: ...
    def grab(self) -> np.ndarray: ...
    # optional:  def autofocus(self) -> None   (called after stage movements when cfg.autofocus is on)


@dataclass
class MapConfig:
    rows: Optional[int] = None  # photos down / across; None = work out from the area (see resolve_grid)
    cols: Optional[int] = None
    um_per_px: float = DEFAULT_CAM_UM_PER_PX
    cam_size: Tuple[int, int] = CAMERA_SIZE_PX
    overlap_min: float = 0.25  # at least this fraction of a frame must overlap its neighbour, i.e. a step of
    #                            at most 3/4 of a frame (the grid is chosen so no step is larger than that)
    margin_um: float = 0.0  # extra border around the area
    sign: Tuple[int, int] = (1, -1)  # stage -> mosaic direction (x right, y down)
    search_px: int = 96  # registration search radius around the stage prediction
    min_peak: float = 0.5
    check_tol_um: float = 3.0  # max left/top disagreement
    check_index: int = 1  # traversal index inside a row that gets the left+top check (1 = 2nd tile)
    check_all: bool = False  # left+top check on every tile that has both neighbours
    retries: int = 1
    ambiguity: str = "reject"  # 'reject' or 'stage_prior' (see register_pair prefer_prior; finished-piece scans)
    # time model (estimates; the real run reports measured compute time on top)
    t_move_s: float = 1.0
    t_settle_s: float = 2.5
    t_capture_s: float = 0.5
    t_retry_extra_settle_s: float = 2.0
    # autofocus after stage movements (needs rig.autofocus(); the old UI's / Toby's autofocus plugs in there)
    autofocus: bool = True
    t_autofocus_s: float = 12.0  # ASSUMPTION until timed on the machine (first photo ~22 z moves, later ones ~10)
    af_min_move_um: float = 25.0  # alignment nudges smaller than this do not trigger a refocus
    # terminal status lines: one per photo taken, occasional ones while stitching
    verbose: bool = True
    label: str = "map"  # name used in the status lines ("map", "finished piece")


@dataclass
class PairMeasurement:
    a: Tile
    b: Tile
    kind: str  # 'chain' (previous tile in the snake) or 'top'
    predicted_px: Tuple[float, float]
    measured_px: Tuple[float, float]
    peak: float
    second: float
    ok: bool
    reason: str = ""

    @property
    def error_px(self) -> Tuple[float, float]:
        return (self.measured_px[0] - self.predicted_px[0], self.measured_px[1] - self.predicted_px[1])


@dataclass
class CheckResult:
    tile: Tile
    from_left_px: Tuple[float, float]
    from_top_px: Tuple[float, float]
    delta_px: float
    delta_um: float
    tol_um: float
    passed: bool


@dataclass
class MapResult:
    ok: bool
    message: str
    cfg: MapConfig
    area_center_um: Tuple[float, float]
    stage_um: Dict[Tile, Tuple[float, float]]
    positions_px: Dict[Tile, Tuple[float, float]]  # solved frame origin in mosaic px
    pairs: List[PairMeasurement]
    checks: List[CheckResult]
    mosaic: Optional[np.ndarray]
    stage_offset_um: Tuple[float, float] = (0.0, 0.0)  # see stage_to_px
    timing: Dict[str, float] = field(default_factory=dict)
    captured: int = 0
    frames: Dict[Tile, np.ndarray] = field(default_factory=dict)  # the raw camera photos (for run folders)
    photo_wall_s: List[float] = field(default_factory=list)  # measured seconds per photo (move+settle+focus+grab+register)

    def stage_to_px(self, x_um: float, y_um: float) -> Tuple[float, float]:
        sx, sy = self.cfg.sign
        s = self.cfg.um_per_px
        return ((sx * x_um - self.stage_offset_um[0]) / s, (sy * y_um - self.stage_offset_um[1]) / s)

    def px_to_stage(self, u: float, v: float) -> Tuple[float, float]:
        sx, sy = self.cfg.sign
        s = self.cfg.um_per_px
        return (sx * (u * s + self.stage_offset_um[0]), sy * (v * s + self.stage_offset_um[1]))

    def report(self) -> List[str]:
        t = self.timing
        lines = [("MAP OK" if self.ok else "MAP FAILED") + f": {self.message}"]
        lines.append(
            f"{self.captured} frames, {len(self.pairs)} pair registrations, {len(self.checks)} left+top checks"
        )
        for c in self.checks:
            lines.append(
                f"  check tile (row {c.tile[0] + 1}, col {c.tile[1] + 1}): left vs top disagree by "
                f"{c.delta_um:.2f} um (limit {c.tol_um:.1f}) -> {'ok' if c.passed else 'FAIL'}"
            )
        if t:
            lines.append(
                f"time: stage {t.get('stage_s', 0):.0f} s (estimated, incl. {int(t.get('autofocus_runs', 0))} autofocus) "
                f"+ compute {t.get('compute_s', 0):.1f} s = {t.get('total_s', 0):.0f} s"
            )
            if "wall_s" in t:
                lines.append(
                    f"measured: {t['wall_s']:.0f} s in total, photos {t.get('wall_photos_s', 0):.0f} s "
                    f"(autofocus {t.get('wall_autofocus_s', 0):.0f} s), stitching {t.get('wall_stitch_s', 0):.1f} s"
                )
        return lines


def snake_order(rows: int, cols: int) -> List[Tile]:
    """Left to right on even rows, right to left on odd rows (same as the tiling snake)."""
    out: List[Tile] = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        out.extend((r, c) for c in cs)
    return out


def resolve_grid(cfg: MapConfig, area_um: Tuple[float, float]) -> MapConfig:
    """Fill in rows/cols: the fewest photos such that no step is bigger than (1 - overlap_min) of a frame
    (3/4 of a frame by default), spread evenly so the area edges are covered."""
    if cfg.rows is not None and cfg.cols is not None:
        return cfg
    fov = (cfg.cam_size[0] * cfg.um_per_px, cfg.cam_size[1] * cfg.um_per_px)
    need = (area_um[0] + 2 * cfg.margin_um, area_um[1] + 2 * cfg.margin_um)
    n = []
    for a in (0, 1):
        extra = max(need[a] - fov[a], 0.0)
        n.append(1 + int(math.ceil(extra / ((1.0 - cfg.overlap_min) * fov[a]) - 1e-9)) if extra > 0 else 1)
    return dataclasses.replace(cfg, rows=cfg.rows or n[1], cols=cfg.cols or n[0])


def scan_summary(area_um: Tuple[float, float], cfg: Optional[MapConfig] = None) -> dict:
    """What mapping this area will involve: photos across/down, the step as a fraction of a frame, time estimate."""
    cfg = resolve_grid(cfg or MapConfig(), area_um)
    _, step, fov = plan_scan(cfg, area_um, (0.0, 0.0))
    n = cfg.rows * cfg.cols
    per = cfg.t_move_s + cfg.t_settle_s + cfg.t_capture_s + (cfg.t_autofocus_s if cfg.autofocus else 0.0)
    return {
        "rows": cfg.rows, "cols": cfg.cols, "photos": n,
        "step_fraction": (step[0] / fov[0] if cfg.cols > 1 else 0.0, step[1] / fov[1] if cfg.rows > 1 else 0.0),
        "fov_um": fov, "est_s": n * per,
    }


def area_for_tiles(
    tile_rows: int, tile_cols: int, geo: Optional[CrossGeometry] = None, mode: str = "trimmed"
) -> PlacementMap:
    """The area a tile_rows x tile_cols pattern needs: one more complete frame than tiles in each direction
    (2x2 tiles -> 3x3 frames), with the pattern in the middle.  Uses the CROSS_* inputs unless a measured
    geometry is given, so it works before any pattern has been chosen."""
    return build_placement_map(geo or spec_geometry(), tile_rows, tile_cols, tile_rows + 1, tile_cols + 1, mode=mode)


def plan_scan(
    cfg: MapConfig, area_um: Tuple[float, float], center_um: Tuple[float, float]
) -> Tuple[Dict[Tile, Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    """Stage positions for the scan grid.  Returns (positions, step_um, fov_um)."""
    fov = (cfg.cam_size[0] * cfg.um_per_px, cfg.cam_size[1] * cfg.um_per_px)
    need = (area_um[0] + 2 * cfg.margin_um, area_um[1] + 2 * cfg.margin_um)
    step = [0.0, 0.0]
    for k, (n, nd, f) in enumerate(((cfg.cols, need[0], fov[0]), (cfg.rows, need[1], fov[1]))):
        if n > 1:
            step[k] = max(nd - f, 0.0) / (n - 1)
            if step[k] > (1.0 - cfg.overlap_min) * f + 1e-9:
                axis = "x" if k == 0 else "y"
                raise ValueError(
                    f"A {cfg.rows}x{cfg.cols} scan cannot cover the area in {axis} with {cfg.overlap_min:.0%} "
                    f"overlap (step {step[k]:.0f} um > {(1 - cfg.overlap_min) * f:.0f} um). Use more rows/cols."
                )
    pos: Dict[Tile, Tuple[float, float]] = {}
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            dx = (c - (cfg.cols - 1) / 2.0) * step[0]
            dy = (r - (cfg.rows - 1) / 2.0) * step[1]
            # mosaic row r is further DOWN, which is lower stage y
            pos[(r, c)] = (center_um[0] + dx, center_um[1] - dy)
    return pos, (step[0], step[1]), fov


def _solve_positions(tiles: List[Tile], pairs: List[PairMeasurement]) -> Dict[Tile, Tuple[float, float]]:
    """Least-squares tile positions from all pair measurements; tiles[0] is the anchor at (0, 0)."""
    idx = {t: i for i, t in enumerate(tiles)}
    n = len(tiles)
    rows_a, rhs, wts = [], [], []
    for p in pairs:
        if not p.ok:
            continue
        row = np.zeros(n)
        row[idx[p.b]] = 1.0
        row[idx[p.a]] = -1.0
        rows_a.append(row)
        rhs.append(p.measured_px)
        wts.append(max(p.peak, 0.1))
    A = np.array(rows_a)[:, 1:]  # anchor column removed
    Wt = np.sqrt(np.array(wts))[:, None]
    sol, *_ = np.linalg.lstsq(A * Wt, np.array(rhs) * Wt, rcond=None)
    out = {tiles[0]: (0.0, 0.0)}
    for t in tiles[1:]:
        out[t] = (float(sol[idx[t] - 1, 0]), float(sol[idx[t] - 1, 1]))
    return out


def compose_mosaic(
    frames: Dict[Tile, np.ndarray],
    pos: Dict[Tile, Tuple[float, float]],
    say: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Feather-blend gray frames at their solved positions.  Returns (uint8 mosaic, origin px).
    ``say`` gets an occasional progress line (about every quarter of the frames)."""
    h, w = next(iter(frames.values())).shape[:2]
    ox = int(math.floor(min(p[0] for p in pos.values())))
    oy = int(math.floor(min(p[1] for p in pos.values())))
    W = int(math.ceil(max(p[0] for p in pos.values()))) - ox + w
    H = int(math.ceil(max(p[1] for p in pos.values()))) - oy + h
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    ramp_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) + 1.0
    ramp_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) + 1.0
    feather = np.minimum.outer(ramp_y, ramp_x)
    if say:
        say(f"stitching: blending {len(frames)} photos into a {W} x {H} px mosaic")
    quarter = max(1, len(frames) // 4)
    for n, (t, f) in enumerate(frames.items(), 1):
        x = int(round(pos[t][0])) - ox
        y = int(round(pos[t][1])) - oy
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        acc[y : y + h, x : x + w] += g.astype(np.float32) * feather
        wsum[y : y + h, x : x + w] += feather
        if say and n % quarter == 0 and n < len(frames):
            say(f"stitching: {n}/{len(frames)} photos blended ({100 * n // len(frames)}%)")
    if say:
        say("stitching: blending done")
    return (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8), (ox, oy)


def will_map(
    rig: Rig,
    cfg: MapConfig,
    area_um: Tuple[float, float],
    center_um: Tuple[float, float],
    progress: Optional[Callable[[float, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> MapResult:
    """Snake-scan the area, register neighbours, check left+top on every new row, build the mosaic.

    * Every tile is registered to the previous tile in the snake (the first tile of a new row to the
      tile above it).
    * On the tile at ``cfg.check_index`` of each new row (the 2nd tile by default) the tile above is
      registered as well; the two independent estimates of that tile's position must agree within
      ``cfg.check_tol_um``.  A tile that fails is re-captured (``cfg.retries``) before giving up.
    """
    note = progress or (lambda f, m: None)
    cfg = resolve_grid(cfg, area_um)
    plan, step, fov = plan_scan(cfg, area_um, center_um)
    order = snake_order(cfg.rows, cfg.cols)
    # the registration search window must leave enough overlap to correlate on (small overlaps at 3/4 steps)
    overlaps_px = [cfg.cam_size[a] - step[a] / cfg.um_per_px for a in (0, 1) if (cfg.cols, cfg.rows)[a] > 1]
    search_eff = cfg.search_px
    if overlaps_px:
        search_eff = int(max(16, min(cfg.search_px, ((min(overlaps_px) - 96) // 2) // 4 * 4)))
    af = getattr(rig, "autofocus", None) if cfg.autofocus else None
    n_af = 0
    af_wall = 0.0
    photo_wall: List[float] = []
    say = status if cfg.verbose else (lambda m: None)
    t_start = time.perf_counter()
    say(
        f"{cfg.label}: {cfg.cols} x {cfg.rows} photos ({len(order)} total) over "
        f"{area_um[0]:.0f} x {area_um[1]:.0f} um, step {step[0]:.0f} x {step[1]:.0f} um, "
        f"autofocus after every move: {'on' if callable(af) else 'off'}"
    )
    s = cfg.um_per_px
    sgn = cfg.sign
    raw: Dict[Tile, np.ndarray] = {}
    prep: Dict[Tile, np.ndarray] = {}
    chain_pos: Dict[Tile, Tuple[float, float]] = {}
    pairs: List[PairMeasurement] = []
    checks: List[CheckResult] = []
    stage_s = 0.0
    compute_s = 0.0

    def predicted(a: Tile, b: Tile) -> Tuple[float, float]:
        dx = plan[b][0] - plan[a][0]
        dy = plan[b][1] - plan[a][1]
        return (sgn[0] * dx / s, sgn[1] * dy / s)

    def capture(tile: Tile, extra_settle: float = 0.0) -> None:
        nonlocal stage_s
        nonlocal n_af, af_wall
        rig.move_to(*plan[tile])
        rig.settle(cfg.t_settle_s + extra_settle)
        if callable(af):
            t_af = time.perf_counter()
            af()
            af_wall += time.perf_counter() - t_af
            n_af += 1
            stage_s += cfg.t_autofocus_s
        raw[tile] = rig.grab()
        stage_s += cfg.t_move_s + cfg.t_settle_s + extra_settle + cfg.t_capture_s

    def fail(msg: str) -> MapResult:
        total = stage_s + compute_s
        return MapResult(
            False, msg, cfg, center_um, plan, {}, pairs, checks, None,
            timing={
                "stage_s": stage_s, "compute_s": compute_s, "total_s": total, "autofocus_runs": n_af,
                "wall_s": time.perf_counter() - t_start, "wall_photos_s": sum(photo_wall), "wall_autofocus_s": af_wall,
                "wall_stitch_s": 0.0,
            },
            captured=len(raw), frames=dict(raw), photo_wall_s=list(photo_wall),
        )

    for k, tile in enumerate(order):
        if should_abort and should_abort():
            return fail("aborted")
        r, c = tile
        in_row = k % cfg.cols
        note(k / len(order), f"capturing tile {k + 1}/{len(order)} (row {r + 1}, col {c + 1})")
        t_photo = time.perf_counter()
        prev = order[k - 1] if in_row > 0 else None
        top = (r - 1, c) if r > 0 else None
        want_check = (
            prev is not None and top is not None and (cfg.check_all or in_row == cfg.check_index)
        )
        neighbours: List[Tuple[str, Tile]] = []
        if prev is not None:
            neighbours.append(("chain", prev))
        if top is not None and (prev is None or want_check):
            neighbours.append(("top", top))

        attempt = 0
        while True:
            capture(tile, extra_settle=cfg.t_retry_extra_settle_s * attempt)
            if k == 0:
                chain_pos[tile] = (0.0, 0.0)
                prep[tile] = prepare_frame(raw[tile])
                break
            t0 = time.perf_counter()
            prep[tile] = prepare_frame(raw[tile])
            batch: List[PairMeasurement] = []
            est: Dict[str, Tuple[float, float]] = {}
            for kind, nb in neighbours:
                reg = register_pair(
                    prep[nb], prep[tile], predicted(nb, tile), search_eff, min_peak=cfg.min_peak,
                    prefer_prior=(cfg.ambiguity == "stage_prior"),
                )
                batch.append(PairMeasurement(nb, tile, kind, predicted(nb, tile), reg.shift, reg.peak, reg.second, reg.ok, reg.reason))
                if reg.ok:
                    est[kind] = (chain_pos[nb][0] + reg.shift[0], chain_pos[nb][1] + reg.shift[1])
            compute_s += time.perf_counter() - t0
            problem = ""
            if not all(p.ok for p in batch):
                bad = next(p for p in batch if not p.ok)
                problem = f"tile (row {r + 1}, col {c + 1}) vs {bad.kind} neighbour: {bad.reason}"
            chk: Optional[CheckResult] = None
            if not problem and want_check and "chain" in est and "top" in est:
                d_px = math.hypot(est["chain"][0] - est["top"][0], est["chain"][1] - est["top"][1])
                chk = CheckResult(tile, est["chain"], est["top"], d_px, d_px * s, cfg.check_tol_um, d_px * s <= cfg.check_tol_um)
                if not chk.passed:
                    problem = (
                        f"left and top disagree by {chk.delta_um:.1f} um at tile (row {r + 1}, col {c + 1}) "
                        f"(limit {cfg.check_tol_um:.1f} um)"
                    )
            if not problem:
                pairs.extend(batch)
                if chk:
                    checks.append(chk)
                chain_pos[tile] = est.get("chain") or est["top"]
                break
            if attempt >= cfg.retries:
                pairs.extend(batch)
                if chk:
                    checks.append(chk)
                return fail(problem)
            attempt += 1
            note(k / len(order), f"re-capturing tile {k + 1}: {problem}")
            say(f"{cfg.label}: photo {k + 1}/{len(order)} re-taking, attempt {attempt + 1}: {problem}")
        photo_wall.append(time.perf_counter() - t_photo)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (k + 1) * (len(order) - k - 1)
        reg_txt = "anchor photo"
        if k > 0 and pairs:
            last = pairs[-len(neighbours):]
            dev = max(math.hypot(p.measured_px[0] - p.predicted_px[0], p.measured_px[1] - p.predicted_px[1]) * s for p in last)
            reg_txt = f"registered, {dev:.1f} um from the stage prediction, match {min(p.peak for p in last):.2f}"
        say(
            f"{cfg.label}: photo {k + 1}/{len(order)} taken (row {r + 1}, col {c + 1}) at stage "
            f"({plan[tile][0]:.0f}, {plan[tile][1]:.0f}) um, {reg_txt}, {elapsed:.0f} s elapsed, ~{eta:.0f} s left"
        )

    say(f"{cfg.label}: all {len(order)} photos taken in {time.perf_counter() - t_start:.0f} s, stitching")
    t_stitch = time.perf_counter()
    t0 = time.perf_counter()
    solved = _solve_positions(order, pairs)
    say(f"stitching: solved {len(solved)} photo positions from {len(pairs)} overlaps")
    mosaic, (ox, oy) = compose_mosaic(raw, solved, say)
    solved = {t: (p[0] - ox, p[1] - oy) for t, p in solved.items()}
    # stage <-> mosaic anchor: average over tiles of (chip position of the commanded stage) - (frame centre)
    cw, ch = cfg.cam_size
    offs = []
    for t, (x, y) in plan.items():
        cx_px = solved[t][0] + cw / 2.0
        cy_px = solved[t][1] + ch / 2.0
        offs.append((sgn[0] * x - cx_px * s, sgn[1] * y - cy_px * s))
    off = (float(np.mean([o[0] for o in offs])), float(np.mean([o[1] for o in offs])))
    compute_s += time.perf_counter() - t0
    note(1.0, "map complete")
    wall_stitch = time.perf_counter() - t_stitch
    say(f"{cfg.label}: done in {time.perf_counter() - t_start:.0f} s")
    total = stage_s + compute_s
    return MapResult(
        True, "all registrations and left/top checks passed", cfg, center_um, plan, solved, pairs, checks,
        mosaic, off,
        {
            "stage_s": stage_s, "compute_s": compute_s, "total_s": total, "autofocus_runs": n_af,
            "wall_s": time.perf_counter() - t_start, "wall_photos_s": sum(photo_wall), "wall_autofocus_s": af_wall,
            "wall_stitch_s": wall_stitch,
        },
        len(raw), dict(raw), list(photo_wall),
    )


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

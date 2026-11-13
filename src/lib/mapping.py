"""Camera map of the exposure area: snake-scan photos, register neighbours, check, stitch.

``map_area()`` snake-scans a grid of camera frames over the area, registers each frame to its neighbours,
and checks that "left" and "top" agree (loop closure) on the second tile of every new row.  No GUI code,
so it is tested headless against the virtual stage (see tiling_selftest.py).

Conventions
-----------
* Camera pixels are the pixels of the camera snapshot (1920x1080 on this machine).
* Mosaic coordinates have x to the right and y DOWN.  Stage coordinates follow the old UI:
  stage +x = right, stage +y = up, so ``sign = (+1, -1)``.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .tiling import PlacementMap


def status(msg: str) -> None:
    """One status line for the terminal (mapping, patterning and stitching all report through this)."""
    print(f"[Map] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Camera (from the old UI; NOT measured on this machine unless noted)
# --------------------------------------------------------------------------------------


CAMERA_SIZE_PX = (1920, 1080)  # "each snapshot captured is 1920 x 1080 pixels" (gui.py)
DEFAULT_CAM_UM_PER_PX = 0.678  # measured by the user

Tile = Tuple[int, int]  # (row, col)


# --------------------------------------------------------------------------------------
# Focus search (Toby's autofocus: golden-section search on the sharpness of a projected ArUco grid)
# --------------------------------------------------------------------------------------


# (grid_n, +- sweep in um, stop when the interval is this narrow in um): his defaults, coarse then fine
FOCUS_STAGES = ((2, 250.0, 5.0), (4, 10.0, 0.5))
FOCUS_SETTLE_S = 0.5  # wait after every z move before grabbing (his default is 1.0 s; the old UI sweep waits 0.5 s)

# Projector changes are confirmed by the camera, not by a delay: after a change, frames are read until one
# differs from the old picture by more than the camera noise AND matches the frame before it (the projector
# has finished changing, so no half-changed frame is used).  Noise is measured from two frames just before.
PROJ_CHANGE_NOISE_FACTOR = 4.0  # "different" = more than this many times the frame-to-frame noise
PROJ_CHANGE_MIN_DN = 1.0  # ...and at least this mean grey difference (0-255), for a nearly noiseless camera
PROJ_CHANGE_MAX_FRAMES = 60  # give up (with a warning) if the picture has not settled after this many frames
CAMERA_DEAD_S = 5.0  # no new frame for this long = the camera stopped (error, not a settling delay)


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
# The map
# --------------------------------------------------------------------------------------


class Rig(Protocol):
    """What map_area needs from the machine.  VirtualRig and ModelRig implement it."""

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


def map_area(
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

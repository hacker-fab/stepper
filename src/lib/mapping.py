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
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

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
# Only the middle of the camera frame is lit by the projector (1037 x 583 um of the 1302 x 732 um view); the
# dark edge around it made dark lines in the map and pulled the matching.  Same 1 / 1.3 as the old stitcher.
CAMERA_LIT_FRACTION = 1 / 1.3

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


FOCUS_LOST_RATIO = 0.6  # a fine search whose best sharpness is below this x the usual focused value missed focus
FOCUS_MAX_PREDICT_UM = 40.0  # a tilt prediction never moves z further than this from the nearest focused point


def predict_focus_z(points: Sequence[Tuple[float, float, float]], x: float, y: float) -> Optional[float]:
    """z of best focus at stage (x, y) from earlier (x, y, z) focus results: a plane through them (the chip's
    tilt) once they span an area, the nearest one before that.  None when there are none yet."""
    if not points:
        return None
    pts = np.array(points, float)
    near = pts[int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))]
    if len(pts) < 3:
        return float(near[2])
    c = pts.mean(axis=0)
    a = np.c_[np.ones(len(pts)), pts[:, 0] - c[0], pts[:, 1] - c[1]]
    # centred, so points along one line (the first row of a map) give no slope across it (min-norm solution)
    coef, *_ = np.linalg.lstsq(a, pts[:, 2], rcond=1e-6)
    z = float(coef[0] + coef[1] * (x - c[0]) + coef[2] * (y - c[1]))
    return float(np.clip(z, near[2] - FOCUS_MAX_PREDICT_UM, near[2] + FOCUS_MAX_PREDICT_UM))


def focus_missed(
    best_z: float, best_score: float, start_z: float, half_range: float, threshold: float, usual: Sequence[float]
) -> Optional[str]:
    """Why a fine focus search did not find focus (None = it did): it ended on the edge of its range, or its
    best sharpness is far below what earlier fine searches reached (the projected grid is the same every time,
    so the focused sharpness hardly changes from place to place)."""
    if abs(best_z - start_z) >= half_range - 2 * threshold:
        return "ended on the edge of its range"
    if usual and best_score < FOCUS_LOST_RATIO * float(np.median(usual)):
        return f"sharpness {best_score:.3g} < {FOCUS_LOST_RATIO:.0%} of the usual {float(np.median(usual)):.3g}"
    return None


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
    peak: float  # confidence: ZNCC peak on the 1/down-scale images (camera noise averaged out), 1.0 = perfect
    second: float  # best ZNCC outside the main peak's own lobe; a small gap means an ambiguous match
    ok: bool
    reason: str = ""
    fine_peak: float = 0.0  # full-resolution ZNCC at the sub-pixel position (camera noise lowers it; info only)


def _second_peak(res: np.ndarray, v: int, u: int) -> float:
    """Best score outside the lobe of the peak at (v, u).

    The lobe is the connected region around the peak above half its height (above the median of the
    surface).  A smooth chip gives a broad peak; a fixed +-3 px exclusion box then found the peak's own
    shoulder as the "second peak" and called a correct match ambiguous."""
    pk = float(res[v, u])
    half = (pk + float(np.median(res))) / 2.0
    _, lab = cv2.connectedComponents((res >= half).astype(np.uint8), connectivity=8)
    lobe = cv2.dilate((lab == lab[v, u]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    lobe[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4] = True
    rest = res[~lobe]
    return float(rest.max()) if rest.size else -1.0


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
    second = _second_peak(res, int(v), int(u))
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
    # Confidence is judged on the coarse surface: at full resolution the ZNCC of a low-feature chip is
    # dominated by camera noise (0.4 for a match that is right to 0.5 px), so the full-res peak only says
    # how noisy the photo is, not whether the position is right.  The fine search only refines the position.
    ok = peak_c >= min_peak and ((peak_c - second) >= min_gap or bool(resolved))
    reason = resolved if (ok and resolved) else ""
    if peak_c < min_peak:
        reason = f"weak match (score {peak_c:.2f} < {min_peak:.2f}); is there anything to register on?"
    elif peak_c - second < min_gap:
        reason = f"ambiguous match (best {peak_c:.2f} vs next {second:.2f}); repeating features?"
    return Registration(shift, peak_c, second, ok, reason, fine_peak=peak)


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
    # camera px moved per stage um, (m00, m01, m10, m11): px = M @ stage_um.  None = sign / um_per_px (square
    # pixels, no rotation).  A map measures the real one (MapResult.stage_px); pass it back in for later runs.
    stage_px: Optional[Tuple[float, float, float, float]] = None
    search_px: int = 96  # registration search radius around the stage prediction
    min_peak: float = 0.5
    check_tol_um: float = 3.0  # max left/top disagreement
    check_index: int = 1  # traversal index inside a row that gets the left+top check (1 = 2nd tile)
    check_all: bool = False  # left+top check on every tile that has both neighbours
    retries: int = 1  # re-captures when left and top disagree (a low-confidence match is not re-captured)
    # A low-confidence match (weak or ambiguous) does not stop the map: that overlap uses the stage-predicted
    # position instead, with a small weight in the solve so real matches win.  False = stop the map instead.
    stage_fallback: bool = True
    estimate_weight: float = 0.05  # solve weight of a stage estimate (real matches weigh their score, 0.5-1)
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
    estimated: bool = False  # True: the match was not trusted, measured_px is the stage prediction

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


Matrix = Tuple[float, float, float, float]  # (m00, m01, m10, m11), row-major 2x2


def nominal_stage_px(cfg: MapConfig) -> Matrix:
    """Camera px per stage um from the configured sign and um/px (or the measured matrix, if given)."""
    if cfg.stage_px is not None:
        return tuple(float(v) for v in cfg.stage_px)  # type: ignore[return-value]
    s = cfg.um_per_px
    return (cfg.sign[0] / s, 0.0, 0.0, cfg.sign[1] / s)


def fov_um(cfg: MapConfig) -> Tuple[float, float]:
    """Stage um covered by one photo along stage x and stage y (per-axis um/px of the matrix, if measured)."""
    m = nominal_stage_px(cfg)
    ux = 1.0 / max(math.hypot(m[0], m[2]), 1e-12)
    uy = 1.0 / max(math.hypot(m[1], m[3]), 1e-12)
    return (cfg.cam_size[0] * ux, cfg.cam_size[1] * uy)


def fit_stage_px(
    steps: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]], prior: Matrix, weight_um2: float = 1e3
) -> Matrix:
    """Least-squares camera-px-per-stage-um matrix from (stage step um, measured px shift) pairs.

    Each row is fitted with a weak pull toward ``prior`` so that a direction that has not been stepped yet
    (only x steps so far) keeps its prior value instead of being undetermined."""
    a = np.eye(2) * weight_um2
    m0 = np.array(prior, float).reshape(2, 2)
    b = m0 * weight_um2  # rows: b_r = lambda * m0_r
    for d, p in steps:
        dv = np.array(d, float)
        a += np.outer(dv, dv)
        b += np.outer(np.array(p, float), dv)
    m = np.linalg.solve(a, b.T).T
    return (float(m[0, 0]), float(m[0, 1]), float(m[1, 0]), float(m[1, 1]))


def fit_stage_px_absolute(
    stage_um: Sequence[Tuple[float, float]], centre_px: Sequence[Tuple[float, float]]
) -> Optional[Tuple[Matrix, Tuple[float, float]]]:
    """Camera px per stage um and offset (mosaic px = M @ stage - offset) from every photo's commanded stage
    position and its solved mosaic position.  Stage backlash and placement noise (a few um per move) average
    out over the span of the map, unlike in a fit to single steps.  None if the photos do not span an area."""
    s = np.array(stage_um, float)
    a = np.c_[s, np.ones(len(s))]
    if len(s) < 3 or np.linalg.matrix_rank(s - s.mean(axis=0), tol=1e-6) < 2:  # all on one line
        return None
    coef, *_ = np.linalg.lstsq(a, np.array(centre_px, float), rcond=None)  # rows: x, y, 1 -> (u, v)
    m = (float(coef[0, 0]), float(coef[1, 0]), float(coef[0, 1]), float(coef[1, 1]))
    return m, (-float(coef[2, 0]), -float(coef[2, 1]))


def describe_stage_px(m: Matrix, sign: Tuple[int, int] = (1, -1)) -> str:
    """um/px along each stage axis and the angle of each stage axis in the photo (0 deg = as configured)."""
    mx, my = np.array([m[0], m[2]]), np.array([m[1], m[3]])  # photo px per um of stage x / stage y
    ax = math.degrees(math.atan2(sign[0] * m[2], sign[0] * m[0]))
    ay = math.degrees(math.atan2(-sign[1] * m[1], sign[1] * m[3]))
    return (
        f"stage x: {1 / max(np.linalg.norm(mx), 1e-12):.4f} um/px at {ax:+.2f} deg, "
        f"stage y: {1 / max(np.linalg.norm(my), 1e-12):.4f} um/px at {ay:+.2f} deg"
    )


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
    stage_px: Optional[Matrix] = None  # measured camera px per stage um (None = nominal_stage_px(cfg))
    stage_px_offset: Tuple[float, float] = (0.0, 0.0)  # mosaic px = stage_px @ stage_um - offset
    timing: Dict[str, float] = field(default_factory=dict)
    captured: int = 0
    frames: Dict[Tile, np.ndarray] = field(default_factory=dict)  # the raw camera photos (for run folders)
    photo_wall_s: List[float] = field(default_factory=list)  # measured seconds per photo (move+settle+focus+grab+register)

    def matrix(self) -> Matrix:
        return self.stage_px if self.stage_px is not None else nominal_stage_px(self.cfg)

    def stage_to_px(self, x_um: float, y_um: float) -> Tuple[float, float]:
        m, o = self.matrix(), self.stage_px_offset
        return (m[0] * x_um + m[1] * y_um - o[0], m[2] * x_um + m[3] * y_um - o[1])

    def px_to_stage(self, u: float, v: float) -> Tuple[float, float]:
        o = self.stage_px_offset
        return self.px_to_stage_delta(u + o[0], v + o[1])

    def px_to_stage_delta(self, du: float, dv: float) -> Tuple[float, float]:
        """Stage um that move the view by (du, dv) photo px."""
        inv = np.linalg.inv(np.array(self.matrix(), float).reshape(2, 2))
        d = inv @ np.array([du, dv], float)
        return (float(d[0]), float(d[1]))

    def um_per_px(self) -> float:
        """Average um per photo px of the (measured) matrix."""
        return 1.0 / math.sqrt(abs(np.linalg.det(np.array(self.matrix(), float).reshape(2, 2))))

    def report(self) -> List[str]:
        t = self.timing
        lines = [("MAP OK" if self.ok else "MAP FAILED") + f": {self.message}"]
        lines.append(
            f"{self.captured} frames, {len(self.pairs)} pair registrations, {len(self.checks)} left+top checks"
        )
        if self.stage_px is not None:
            lines.append(f"camera vs stage, measured: {describe_stage_px(self.stage_px, self.cfg.sign)}")
        for p in self.pairs:
            if p.estimated:
                lines.append(
                    f"  stage estimate for tile (row {p.b[0] + 1}, col {p.b[1] + 1}) vs {p.kind} neighbour: {p.reason}"
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
    fov = fov_um(cfg)
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


def scan_area_um(pm: PlacementMap, cfg: MapConfig) -> Tuple[float, float]:
    """The area to map for a placement: what the alignment needs, and no more photos than necessary.

    The alignment photographs each tile centred on its stage target, so every such view must be on the map:
    tile-centre span + one photo.  That is widened to what (tiles + 1) photos per direction cover at the
    minimum overlap (more context for free), but never beyond the placement's full area of complete frames.
    With the photo cropped to the lit part of the frame (about one projector frame), mapping the full
    (tiles + 1)-frame area would need one more photo per direction for no benefit."""
    fov = fov_um(cfg)
    out = []
    for a, n in ((0, pm.cols), (1, pm.rows)):
        centres = [t.center_um[a] for t in pm.tiles]
        need = (max(centres) - min(centres)) + fov[a]
        grid = fov[a] * (1.0 + n * (1.0 - cfg.overlap_min)) * (1.0 - 1e-6)  # n + 1 photos at the minimum overlap
        out.append(max(need, min(grid, pm.area_size_um[a])))
    return (out[0], out[1])


def plan_scan(
    cfg: MapConfig, area_um: Tuple[float, float], center_um: Tuple[float, float]
) -> Tuple[Dict[Tile, Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    """Stage positions for the scan grid.  Returns (positions, step_um, fov_um)."""
    fov = fov_um(cfg)
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


def _solve_positions(
    tiles: List[Tile], pairs: List[PairMeasurement], estimate_weight: float = 0.05
) -> Dict[Tile, Tuple[float, float]]:
    """Least-squares tile positions from all pair measurements; tiles[0] is the anchor at (0, 0).
    Stage estimates (pairs whose match was not trusted) take part with ``estimate_weight``."""
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
        wts.append(estimate_weight if p.estimated else max(p.peak, 0.1))
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


def _stage_estimate(
    a: Tile, b: Tile, kind: str, predicted_px: Tuple[float, float], peak: float, second: float, why: str
) -> PairMeasurement:
    """An overlap whose match is not trusted: the stage-predicted offset stands in for the measurement."""
    return PairMeasurement(a, b, kind, predicted_px, predicted_px, peak, second, True, why, estimated=True)


def _map_message(pairs: List[PairMeasurement]) -> str:
    n_est = sum(p.estimated for p in pairs)
    if not n_est:
        return "all registrations and left/top checks passed"
    return f"map complete; {n_est} of {len(pairs)} overlaps used the stage position (low-confidence match)"


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
    fov = fov_um(cfg)
    overlaps_px = [cfg.cam_size[a] * (1.0 - step[a] / fov[a]) for a in (0, 1) if (cfg.cols, cfg.rows)[a] > 1]
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

    # camera px per stage um: starts from the configured / last measured value and is re-fitted from every
    # trusted overlap, so later predictions (and the search window around them) follow the real stage
    prior = nominal_stage_px(cfg)
    steps: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    m_now: List[Matrix] = [prior]

    def predicted(a: Tile, b: Tile) -> Tuple[float, float]:
        dx = plan[b][0] - plan[a][0]
        dy = plan[b][1] - plan[a][1]
        m = m_now[0]
        return (m[0] * dx + m[1] * dy, m[2] * dx + m[3] * dy)

    def learn(batch: List[PairMeasurement]) -> None:
        for p in batch:
            if not p.estimated:
                steps.append(((plan[p.b][0] - plan[p.a][0], plan[p.b][1] - plan[p.a][1]), p.measured_px))
        if steps:
            m_now[0] = fit_stage_px(steps, prior)

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
                pred = predicted(nb, tile)
                reg = register_pair(
                    prep[nb], prep[tile], pred, search_eff, min_peak=cfg.min_peak,
                    prefer_prior=(cfg.ambiguity == "stage_prior"),
                )
                if reg.ok or not cfg.stage_fallback:
                    pm = PairMeasurement(nb, tile, kind, pred, reg.shift, reg.peak, reg.second, reg.ok, reg.reason)
                else:
                    pm = _stage_estimate(nb, tile, kind, pred, reg.peak, reg.second, reg.reason)
                batch.append(pm)
                if pm.ok:
                    est[kind] = (chain_pos[nb][0] + pm.measured_px[0], chain_pos[nb][1] + pm.measured_px[1])
            compute_s += time.perf_counter() - t0
            problem = ""
            if not all(p.ok for p in batch):
                bad = next(p for p in batch if not p.ok)
                problem = f"tile (row {r + 1}, col {c + 1}) vs {bad.kind} neighbour: {bad.reason}"
            measured = {p.kind for p in batch if not p.estimated}
            chk: Optional[CheckResult] = None
            # the check compares two real measurements; a stage estimate would only compare the stage with itself
            if not problem and want_check and {"chain", "top"} <= measured:
                d_px = math.hypot(est["chain"][0] - est["top"][0], est["chain"][1] - est["top"][1])
                chk = CheckResult(tile, est["chain"], est["top"], d_px, d_px * s, cfg.check_tol_um, d_px * s <= cfg.check_tol_um)
                if not chk.passed:
                    problem = (
                        f"left and top disagree by {chk.delta_um:.1f} um at tile (row {r + 1}, col {c + 1}) "
                        f"(limit {cfg.check_tol_um:.1f} um)"
                    )
            if not problem:
                pairs.extend(batch)
                learn(batch)
                if chk:
                    checks.append(chk)
                chain_pos[tile] = est.get("chain") or est["top"]
                break
            if attempt >= cfg.retries:
                if chk is None or not cfg.stage_fallback:
                    pairs.extend(batch)
                    if chk:
                        checks.append(chk)
                    return fail(problem)
                # left and top still disagree: one of them is a false lock, and nothing says which one, so
                # this tile is placed from the stage instead of from either match
                checks.append(chk)
                why = f"left and top disagree by {chk.delta_um:.1f} um"
                batch = [_stage_estimate(p.a, p.b, p.kind, p.predicted_px, p.peak, p.second, why) for p in batch]
                pairs.extend(batch)
                nb, pm = (prev, batch[0]) if prev is not None else (top, batch[0])
                chain_pos[tile] = (chain_pos[nb][0] + pm.measured_px[0], chain_pos[nb][1] + pm.measured_px[1])
                say(f"{cfg.label}: photo {k + 1}/{len(order)}: {why} after {attempt + 1} tries, using the stage position")
                break
            attempt += 1
            note(k / len(order), f"re-capturing tile {k + 1}: {problem}")
            say(f"{cfg.label}: photo {k + 1}/{len(order)} re-taking, attempt {attempt + 1}: {problem}")
        photo_wall.append(time.perf_counter() - t_photo)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (k + 1) * (len(order) - k - 1)
        reg_txt = "anchor photo"
        if k > 0 and pairs:
            last = pairs[-len(neighbours):]
            real = [p for p in last if not p.estimated]
            if real:
                dev = max(math.hypot(p.error_px[0], p.error_px[1]) * s for p in real)
                reg_txt = f"registered, {dev:.1f} um from the stage prediction, match {min(p.peak for p in real):.2f}"
            else:
                reg_txt = f"stage position used ({last[0].reason})"
        say(
            f"{cfg.label}: photo {k + 1}/{len(order)} taken (row {r + 1}, col {c + 1}) at stage "
            f"({plan[tile][0]:.0f}, {plan[tile][1]:.0f}) um, {reg_txt}, {elapsed:.0f} s elapsed, ~{eta:.0f} s left"
        )

    say(f"{cfg.label}: all {len(order)} photos taken in {time.perf_counter() - t_start:.0f} s, stitching")
    t_stitch = time.perf_counter()
    t0 = time.perf_counter()
    solved = _solve_positions(order, pairs, cfg.estimate_weight)
    say(f"stitching: solved {len(solved)} photo positions from {len(pairs)} overlaps")
    mosaic, (ox, oy) = compose_mosaic(raw, solved, say)
    solved = {t: (p[0] - ox, p[1] - oy) for t, p in solved.items()}
    # stage <-> mosaic: fitted to every photo's commanded stage position vs its solved mosaic centre; a single
    # row or column of photos cannot measure both axes, so it keeps the step fit / prior for the other one
    cw, ch = cfg.cam_size
    tiles = list(plan)
    centres = [(solved[t][0] + cw / 2.0, solved[t][1] + ch / 2.0) for t in tiles]
    fit = fit_stage_px_absolute([plan[t] for t in tiles], centres)
    if fit is not None:
        m, off = fit
    else:
        m = m_now[0] if steps else prior
        offs = [(m[0] * plan[t][0] + m[1] * plan[t][1] - c[0], m[2] * plan[t][0] + m[3] * plan[t][1] - c[1]) for t, c in zip(tiles, centres)]
        off = (float(np.mean([o[0] for o in offs])), float(np.mean([o[1] for o in offs])))
    measured = fit is not None or bool(steps)
    if measured:
        say(f"{cfg.label}: camera vs stage, measured: {describe_stage_px(m, sgn)}")
    compute_s += time.perf_counter() - t0
    note(1.0, "map complete")
    wall_stitch = time.perf_counter() - t_stitch
    say(f"{cfg.label}: done in {time.perf_counter() - t_start:.0f} s")
    total = stage_s + compute_s
    return MapResult(
        True, _map_message(pairs), cfg, center_um, plan, solved, pairs, checks,
        mosaic, m if measured else None, off,
        {
            "stage_s": stage_s, "compute_s": compute_s, "total_s": total, "autofocus_runs": n_af,
            "wall_s": time.perf_counter() - t_start, "wall_photos_s": sum(photo_wall), "wall_autofocus_s": af_wall,
            "wall_stitch_s": wall_stitch,
        },
        len(raw), dict(raw), list(photo_wall),
    )



# --------------------------------------------------------------------------------------
# Projector -> camera (measured from the autofocus grid) and where each tile goes on the map
# --------------------------------------------------------------------------------------
@dataclass
class ProjectorCalibration:
    """Where the projected frame lands in the photo: photo px = a @ (u, v) + b, with (u, v) the position in
    the projected frame as a fraction (0..1, x right, y down).  Measured by projecting the autofocus grid
    through the same image pipeline as the patterns (image adjust included), so it holds the real
    projector-to-camera scale, rotation and any mirroring.  ``b`` and ``cam_size`` are of the cropped photo."""

    a: Matrix
    b: Tuple[float, float]
    cam_size: Tuple[int, int]
    n_points: int = 0
    residual_px: float = 0.0

    def to_px(self, u: float, v: float) -> Tuple[float, float]:
        a = self.a
        return (a[0] * u + a[1] * v + self.b[0], a[2] * u + a[3] * v + self.b[1])

    def describe(self) -> str:
        a = np.array(self.a).reshape(2, 2)
        w, h = np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1])
        ang = math.degrees(math.atan2(a[1, 0], a[0, 0]))
        off = self.frame_centre_offset_px()
        mirror = ", mirrored" if np.linalg.det(a) < 0 else ""
        return (f"projected frame {w:.0f} x {h:.0f} photo px, turned {ang:+.2f} deg{mirror}, centre "
                f"{off[0]:+.0f}, {off[1]:+.0f} px from the photo centre ({self.n_points // 4} markers, fit {self.residual_px:.2f} px)")

    def frame_centre_offset_px(self) -> Tuple[float, float]:
        """Projected frame centre minus photo centre (the projector need not be centred on the camera)."""
        c = self.to_px(0.5, 0.5)
        return (c[0] - self.cam_size[0] / 2.0, c[1] - self.cam_size[1] / 2.0)


def fit_projector_calibration(
    frame_pts: np.ndarray, photo_pts: np.ndarray, cam_size: Tuple[int, int], max_residual_px: float = 3.0
) -> Optional[ProjectorCalibration]:
    """Affine fit of photo px to projected-frame fractions (find_aruco_grid output).  None with fewer than
    three markers' worth of corners or a poor fit (a marker found in the wrong place)."""
    if len(frame_pts) < 12:
        return None
    a = np.c_[frame_pts, np.ones(len(frame_pts))]
    coef, *_ = np.linalg.lstsq(a, photo_pts, rcond=None)
    res = float(np.sqrt(np.mean(np.sum((a @ coef - photo_pts) ** 2, axis=1))))
    if res > max_residual_px:
        return None
    m = (float(coef[0, 0]), float(coef[1, 0]), float(coef[0, 1]), float(coef[1, 1]))
    return ProjectorCalibration(m, (float(coef[2, 0]), float(coef[2, 1])), tuple(cam_size), len(frame_pts), res)


def average_projector_calibration(cals: Sequence[ProjectorCalibration]) -> Optional[ProjectorCalibration]:
    """Element-wise median of several measurements (one per autofocus)."""
    if not cals:
        return None
    a = tuple(float(np.median([c.a[k] for c in cals])) for k in range(4))
    b = (float(np.median([c.b[0] for c in cals])), float(np.median([c.b[1] for c in cals])))
    return ProjectorCalibration(a, b, cals[-1].cam_size, sum(c.n_points for c in cals),  # type: ignore[arg-type]
                                float(np.median([c.residual_px for c in cals])))


def tile_point_map_px(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float], proj: Optional[ProjectorCalibration],
    slot: int, fx: float, fy: float,
) -> Tuple[float, float]:
    """Mosaic px where pattern-frame px (fx, fy) of tile ``slot`` lands on the chip.

    With a projector calibration the tile offsets go through the measured projector -> camera mapping, so the
    spacing follows the optics (not the stage, whose errors would scale it); the pattern block is centred on
    the map position of ``center_um``.  Without one, through the stage and stage_to_px as before."""
    t, g = pm.tiles[slot], pm.geo
    ox = t.center_um[0] / pm.um_per_px[0] + fx - g.frame_w / 2.0  # frame px from the pattern-block centre
    oy = t.center_um[1] / pm.um_per_px[1] + fy - g.frame_h / 2.0
    if proj is not None:
        anchor = mr.stage_to_px(*center_um)
        a = proj.a
        u, v = ox / g.frame_w, oy / g.frame_h
        return (anchor[0] + a[0] * u + a[1] * v, anchor[1] + a[2] * u + a[3] * v)
    sg = mr.cfg.sign
    return mr.stage_to_px(center_um[0] + sg[0] * ox * pm.um_per_px[0], center_um[1] + sg[1] * oy * pm.um_per_px[1])


def tile_stage_targets(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float], proj: Optional[ProjectorCalibration]
) -> List[Tuple[float, float]]:
    """Stage position for each tile (reading order) at which the projector frame lands where it should.

    With a projector calibration: the camera must sit so that the projected frame centre (which may be off
    the photo centre) lands on the tile's map position; the stage command is that map position through the
    measured camera-vs-stage matrix, and closed-loop alignment to the map removes what that matrix gets
    wrong.  Without one: the placement's own stage targets."""
    if proj is None:
        return pm.stage_targets(center_um, mr.cfg.sign)
    off = proj.frame_centre_offset_px()
    out = []
    for t in pm.tiles:
        px = tile_point_map_px(mr, pm, center_um, proj, t.slot, pm.geo.frame_w / 2.0, pm.geo.frame_h / 2.0)
        out.append(mr.px_to_stage(px[0] - off[0], px[1] - off[1]))
    return out


def background_from_map(mr: "MapResult", pm: PlacementMap, scale: float) -> np.ndarray:
    """The camera mosaic cut to the placement area, at preview scale (BGR), for drawing underneath
    the placement.  The area is centred on the stage position the map was centred on."""
    out_w = max(1, int(round(pm.area_size_px[0] * scale)))
    out_h = max(1, int(round(pm.area_size_px[1] * scale)))
    cx, cy = mr.stage_to_px(*mr.area_center_um)
    w_cam = pm.area_size_um[0] / mr.um_per_px()
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

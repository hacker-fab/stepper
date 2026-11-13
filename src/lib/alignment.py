"""Closed-loop alignment for tiling: move, look, measure the error, correct, repeat.

No GUI code, so it runs headless against the virtual chip (see tiling_selftest.py).

How it works
------------
``align_to_target(rig, target_um, measurers)`` moves the stage to ``target_um``, grabs a photo and asks
each *measurer* how far the view is from where it should be.  The measurers answer in stage um
("move by (dx, dy) to be on target").  The stage is then commanded to ``command + correction`` and the
loop repeats until the measured error is below ``tol_um``.  Because the error is measured from the
photo, stage backlash and placement noise are corrected too (backlash only costs one extra iteration).

Two measurers, which can be used together and compared:

* ``FeatureMeasurer``  registers the live photo against the camera-map mosaic (ZNCC on the chip's own
  scratches / dust / texture).  Needs no marks, but it only knows "same place as when I mapped", and
  the chip surface changes once something has been patterned or developed, so it must be re-mapped
  after each layer.
* ``CrossMeasurer``    finds the alignment crosses in the photo (template matching with the cross cut
  out of your pattern; an optional ML detector such as the YOLO model in gui.py can seed the search)
  and compares them with where they should be.  It follows the *previous layer*, so it survives
  patterning, but it needs the crosses to be there.

When both are available every iteration records both answers and the difference between them.  That
difference is the useful number: it is how far the previous layer's crosses sit from where the map says
that position is, i.e. the placement error of the earlier exposure.

Sign conventions are those of lib/mapping.py: mosaic x right / y down, stage +x right / +y up (sign = (1, -1)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .mapping import MapConfig, MapResult, Rig, prepare_frame, register_pair
from .tiling import PROJ_UM_PER_PX, CrossGeometry

Point = Tuple[float, float]


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class Measurement:
    method: str  # 'features' | 'crosses'
    ok: bool
    correction_um: Point  # stage move that brings the view onto the target
    confidence: float  # 0..1 (ZNCC peak / mean cross score)
    n_used: int  # crosses used (features: 1)
    detail: str = ""

    @property
    def error_um(self) -> float:
        return math.hypot(*self.correction_um)


class Measurer:
    name = "measurer"

    def measure(self, frame: np.ndarray, target_um: Point) -> Measurement:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Feature measurer: live photo vs the camera-map mosaic
# --------------------------------------------------------------------------------------
class FeatureMeasurer(Measurer):
    """Registers the live photo against the sharpest reference available.

    By default that is the nearest raw map photo(s): a blended mosaic is slightly soft wherever frames
    overlap (every frame sits at a rounded pixel position), which costs correlation on fine chip
    texture.  The mosaic is only used when the raw photos are not available (``use_mosaic=True``).
    """

    name = "features"

    def __init__(
        self,
        map_result: MapResult,
        search_px: int = 96,
        min_peak: float = 0.45,
        n_refs: int = 2,
        use_mosaic: bool = False,
        margin_px: int = 160,
    ):
        if not map_result.ok or map_result.mosaic is None:
            raise ValueError("FeatureMeasurer needs a successful map")
        self.mr = map_result
        self.search = search_px
        self.min_peak = min_peak
        self.n_refs = n_refs
        self.margin = margin_px
        self.use_mosaic = use_mosaic or not map_result.frames
        self._prep: Dict[Tuple[int, int], np.ndarray] = {}

    def _prepared(self, tile: Tuple[int, int]) -> np.ndarray:
        if tile not in self._prep:
            self._prep[tile] = prepare_frame(self.mr.frames[tile])
        return self._prep[tile]

    def measure(self, frame: np.ndarray, target_um: Point) -> Measurement:
        mr = self.mr
        cw, ch = mr.cfg.cam_size
        s, sg = mr.cfg.um_per_px, mr.cfg.sign
        cx, cy = mr.stage_to_px(*target_um)
        origin = (cx - cw / 2.0, cy - ch / 2.0)  # where the target view's top-left sits in the mosaic
        new = prepare_frame(frame)
        if self.use_mosaic:
            return self._from_refs(new, [(None, None)], origin, s, sg)
        near = sorted(mr.positions_px, key=lambda t: math.hypot(mr.positions_px[t][0] - origin[0], mr.positions_px[t][1] - origin[1]))
        return self._from_refs(new, [(t, mr.positions_px[t]) for t in near[: self.n_refs]], origin, s, sg)

    def _from_refs(self, new, refs, origin, s, sg) -> Measurement:
        mr, m = self.mr, self.margin
        devs: List[Tuple[float, float, float]] = []
        reasons: List[str] = []
        for tile, pos in refs:
            if tile is None:  # mosaic crop around the target
                cw, ch = mr.cfg.cam_size
                x0, y0 = int(round(origin[0])) - m, int(round(origin[1])) - m
                w, h = cw + 2 * m, ch + 2 * m
                mos = mr.mosaic
                crop = mos[max(0, y0) : y0 + h, max(0, x0) : x0 + w]
                if crop.size == 0:
                    return Measurement(self.name, False, (0.0, 0.0), 0.0, 0, "target is outside the map")
                pad = ((max(0, -y0), max(0, y0 + h - mos.shape[0])), (max(0, -x0), max(0, x0 + w - mos.shape[1])))
                crop = np.pad(crop, pad, mode="edge")
                ref, predicted = prepare_frame(crop), (float(m), float(m))
            else:
                ref = self._prepared(tile)
                predicted = (origin[0] - pos[0], origin[1] - pos[1])
            reg = register_pair(ref, new, predicted, self.search, min_peak=self.min_peak)
            if reg.ok:
                devs.append((reg.shift[0] - predicted[0], reg.shift[1] - predicted[1], reg.peak))
            else:
                reasons.append(reg.reason)
        if not devs:
            return Measurement(self.name, False, (0.0, 0.0), 0.0, 0, reasons[0] if reasons else "no reference")
        w8 = np.array([d[2] for d in devs])
        dev = (sum(d[0] * d[2] for d in devs) / w8.sum(), sum(d[1] * d[2] for d in devs) / w8.sum())
        spread = max((math.hypot(d[0] - dev[0], d[1] - dev[1]) for d in devs), default=0.0) * s
        corr = (-sg[0] * dev[0] * s, -sg[1] * dev[1] * s)
        detail = f"{len(devs)} ref(s), ZNCC {float(w8.mean()):.2f}" + (f", refs differ {spread:.2f} um" if len(devs) > 1 else "")
        return Measurement(self.name, True, corr, float(w8.mean()), len(devs), detail)


# --------------------------------------------------------------------------------------
# Cross measurer: find the crosses in the photo and compare with where they should be
# --------------------------------------------------------------------------------------
def cross_positions_um(
    geo: CrossGeometry,
    target_um: Point,
    proj_um_per_px: Point = PROJ_UM_PER_PX,
    sign: Tuple[int, int] = (1, -1),
    mirror: Tuple[bool, bool] = (False, False),
    names: Sequence[str] = ("tl", "tr", "bl", "br"),
) -> List[Point]:
    """Stage um of the centre of each (named) cross of a frame exposed with its centre at ``target_um``."""
    out: List[Point] = []
    for name in names:
        x, y = geo.corner(name)
        cx, cy = x + geo.center_in_cross[0], y + geo.center_in_cross[1]
        dx = (cx - geo.frame_w / 2.0) * proj_um_per_px[0] * (-1 if mirror[0] else 1)
        dy = (cy - geo.frame_h / 2.0) * proj_um_per_px[1] * (-1 if mirror[1] else 1)
        out.append((target_um[0] + sign[0] * dx, target_um[1] + sign[1] * dy))
    return out


def earlier_crosses(pm, slot: int) -> List[str]:
    """Names of the crosses of tile ``slot`` that an EARLIER tile has already exposed on the chip.

    The same physical cross is the TL of one frame, the TR of its left neighbour, the BL of the frame above,
    and so on; with trimmed footprints only one tile exposes it.  A tile that has a left or top neighbour
    therefore finds some of its own crosses already printed: aligning to those places the new tile relative
    to what is actually on the chip, whatever the map or the stage says.
    """
    geo = pm.geo
    me = pm.tiles[slot]
    by_rc = {(t.row, t.col): t for t in pm.tiles}
    out: List[str] = []
    for name in ("tl", "tr", "bl", "br"):
        dr = 1 if name in ("bl", "br") else 0
        dc = 1 if name in ("tr", "br") else 0
        found = False
        # the four frames that share the grid vertex (me.row + dr, me.col + dc)
        for rr, cc, nm in (
            (me.row + dr, me.col + dc, "tl"),
            (me.row + dr, me.col + dc - 1, "tr"),
            (me.row + dr - 1, me.col + dc, "bl"),
            (me.row + dr - 1, me.col + dc - 1, "br"),
        ):
            t = by_rc.get((rr, cc))
            if t is None or t.slot >= slot:
                continue
            x, y = geo.corner(nm)
            x0, y0, x1, y1 = t.crop
            if x0 <= x and x + geo.cross_w <= x1 and y0 <= y and y + geo.cross_h <= y1:
                found = True
                break
        if found:
            out.append(name)
    return out


MLDetector = Callable[[np.ndarray], List[Point]]


def yolo_detector(model, conf: float = 0.1) -> MLDetector:
    """Adapter for the ultralytics YOLO model used by gui.py (ckpts/best.pt): returns box centres in photo px.

    ``conf`` is lower than the model's 0.25 default on purpose: the boxes only seed the template search
    (which gives the precise position), and on virtual crosses 0.25 found 1 of 4, 0.1 found 3 of 4."""

    def detect(frame: np.ndarray) -> List[Point]:
        rgb = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        res = model(cv2.resize(rgb, (640, 640)), conf=conf, verbose=False)  # low conf: only used as a coarse seed
        pts: List[Point] = []
        for box in res[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            pts.append(((x1 + x2) / 2.0 * w / 640.0, (y1 + y2) / 2.0 * h / 640.0))
        return pts

    return detect


class CrossMeasurer(Measurer):
    name = "crosses"

    def __init__(
        self,
        cross_mask: np.ndarray,  # one cross, bool, in projector px (cross_patch of a pattern)
        center_in_cross: Point,
        expected_um: Sequence[Point],  # where the crosses are on the chip (stage um)
        cfg: MapConfig,
        proj_um_per_px: Point = PROJ_UM_PER_PX,
        mirror: Tuple[bool, bool] = (False, False),
        search_px: int = 110,
        min_score: float = 0.30,
        min_gap: float = 0.08,
        min_crosses: int = 2,
        scales: Sequence[float] = (0.94, 1.0, 1.06),
        ml_detector: Optional[MLDetector] = None,
        ml_gate_px: float = 90.0,
    ):
        self.cfg = cfg
        self.expected = list(expected_um)
        self.proj = proj_um_per_px
        self.search = search_px
        self.min_score = min_score
        self.min_gap = min_gap
        self.min_crosses = min_crosses
        self.ml = ml_detector
        self.ml_gate = ml_gate_px
        m = cross_mask.astype(np.float32)
        if mirror[0]:
            m = m[:, ::-1]
        if mirror[1]:
            m = m[::-1, :]
        self._mask = np.ascontiguousarray(m)
        cic = (
            (self._mask.shape[1] - 1 - center_in_cross[0]) if mirror[0] else center_in_cross[0],
            (self._mask.shape[0] - 1 - center_in_cross[1]) if mirror[1] else center_in_cross[1],
        )
        self._cic = cic
        self._scales = list(scales)
        self.scale = 1.0  # best scale seen so far (a cheap measurement of the projector um/px error)
        self._templates: Dict[float, np.ndarray] = {}

    def _template(self, k: float) -> np.ndarray:
        if k not in self._templates:
            sx = self.proj[0] / self.cfg.um_per_px * k
            sy = self.proj[1] / self.cfg.um_per_px * k
            w = max(8, int(round(self._mask.shape[1] * sx)))
            h = max(8, int(round(self._mask.shape[0] * sy)))
            self._templates[k] = cv2.resize(self._mask, (w, h), interpolation=cv2.INTER_AREA)
        return self._templates[k]

    def _locate(self, prep: np.ndarray, centre: Point, radius: float, scales: Sequence[float]):
        """Best template position near ``centre`` (photo px).  Returns (cx, cy, score, gap, scale) or None."""
        H, W = prep.shape
        best = None
        for k in scales:
            t = self._template(k)
            th, tw = t.shape
            # the template's reference point sits at (cic * scale) inside it
            rx, ry = self._cic[0] * t.shape[1] / self._mask.shape[1], self._cic[1] * t.shape[0] / self._mask.shape[0]
            x0 = int(round(centre[0] - rx - radius))
            y0 = int(round(centre[1] - ry - radius))
            x1 = int(round(centre[0] - rx + radius)) + tw
            y1 = int(round(centre[1] - ry + radius)) + th
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < tw + 2 or y1 - y0 < th + 2:
                continue
            res = cv2.matchTemplate(prep[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)
            a = np.abs(np.nan_to_num(res, nan=0.0))  # the printed cross may be darker or brighter than the chip
            v, u = np.unravel_index(int(np.argmax(a)), a.shape)
            peak = float(a[v, u])
            sup = a.copy()
            r = max(3, min(tw, th) // 2)
            sup[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1] = 0.0
            gap = peak - float(sup.max())

            def sub(p: float, q: float, r_: float) -> float:
                den = p - 2 * q + r_
                return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (p - r_) / den, -0.5, 0.5))

            du = sub(a[v, u - 1], a[v, u], a[v, u + 1]) if 0 < u < a.shape[1] - 1 else 0.0
            dv = sub(a[v - 1, u], a[v, u], a[v + 1, u]) if 0 < v < a.shape[0] - 1 else 0.0
            cx, cy = x0 + u + du + rx, y0 + v + dv + ry
            if best is None or peak > best[2]:
                best = (cx, cy, peak, gap, k)
        return best

    def measure(self, frame: np.ndarray, target_um: Point) -> Measurement:
        cfg = self.cfg
        cw, ch = cfg.cam_size
        s, sg = cfg.um_per_px, cfg.sign
        prep = prepare_frame(frame)
        dets = self.ml(frame) if self.ml else []
        errs: List[Tuple[float, float, float]] = []  # (ex, ey, score)
        notes: List[str] = []
        for i, (X, Y) in enumerate(self.expected):
            exp = (cw / 2.0 + sg[0] * (X - target_um[0]) / s, ch / 2.0 + sg[1] * (Y - target_um[1]) / s)
            if not (-20 <= exp[0] <= cw + 20 and -20 <= exp[1] <= ch + 20):
                notes.append(f"cross {i + 1} out of view")
                continue
            centre, radius, scales = exp, self.search, self._scales
            seeded = False
            if dets:
                near = min(dets, key=lambda d: math.hypot(d[0] - exp[0], d[1] - exp[1]))
                if math.hypot(near[0] - exp[0], near[1] - exp[1]) <= self.ml_gate:
                    centre, radius, scales, seeded = near, 24.0, [self.scale], True
            loc = self._locate(prep, centre, radius, scales)
            if seeded and (loc is None or loc[2] < self.min_score):
                errs.append((centre[0] - exp[0], centre[1] - exp[1], 0.25))  # ML centre alone, low weight
                notes.append(f"cross {i + 1}: ML box only")
                continue
            if loc is None:
                notes.append(f"cross {i + 1}: no search window")
                continue
            cx, cy, score, gap, k = loc
            if score < self.min_score:
                notes.append(f"cross {i + 1}: weak ({score:.2f})")
                continue
            if gap < self.min_gap and not seeded:
                notes.append(f"cross {i + 1}: ambiguous ({score:.2f} vs {score - gap:.2f})")
                continue
            self.scale = k
            errs.append((cx - exp[0], cy - exp[1], score))
        if len(errs) < self.min_crosses:
            why = "; ".join(notes) or "none found"
            return Measurement(self.name, False, (0.0, 0.0), 0.0, len(errs), f"only {len(errs)} cross(es) found ({why})")
        e = np.array([[a, b] for a, b, _ in errs])
        w = np.array([c for _, _, c in errs])
        med = np.median(e, axis=0)
        keep = np.hypot(*(e - med).T) <= max(4.0, 3.0 * float(np.median(np.hypot(*(e - med).T))))
        if keep.sum() < self.min_crosses:
            return Measurement(self.name, False, (0.0, 0.0), 0.0, int(keep.sum()), "crosses disagree with each other")
        mean = (e[keep] * w[keep, None]).sum(axis=0) / w[keep].sum()
        # crosses appear displaced by +mean in the photo, so the camera sits at -mean: move by +mean
        corr = (sg[0] * float(mean[0]) * s, sg[1] * float(mean[1]) * s)
        spread = float(np.max(np.hypot(*(e[keep] - mean).T))) * s
        detail = f"{int(keep.sum())} crosses, scale x{self.scale:.2f}, spread {spread:.2f} um" + (" (ML seeded)" if dets else "")
        return Measurement(self.name, True, corr, float(w[keep].mean()), int(keep.sum()), detail)


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------
@dataclass
class AlignStep:
    iteration: int
    commanded_um: Point
    measurements: List[Measurement]
    used: str  # method that drove the correction ('' if none could)
    error_um: float  # error according to ``used``
    methods_delta_um: Optional[float]  # |features - crosses| when both measured, else None


@dataclass
class AlignResult:
    ok: bool
    message: str
    target_um: Point
    final_command_um: Point
    final_error_um: float
    steps: List[AlignStep]
    stage_s: float  # estimated stage time
    last_frame: Optional[np.ndarray] = None
    frames: List[np.ndarray] = field(default_factory=list)  # photo taken at each iteration

    def method_differences(self) -> List[float]:
        return [s.methods_delta_um for s in self.steps if s.methods_delta_um is not None]

    def report(self) -> List[str]:
        lines = [("ALIGNED" if self.ok else "NOT ALIGNED") + f": {self.message}"]
        lines.append(
            f"target ({self.target_um[0]:.1f}, {self.target_um[1]:.1f}) um, final command "
            f"({self.final_command_um[0]:.1f}, {self.final_command_um[1]:.1f}) um, "
            f"final error {self.final_error_um:.2f} um, {len(self.steps)} photo(s), ~{self.stage_s:.0f} s of stage time"
        )
        for s in self.steps:
            parts = []
            for m in s.measurements:
                parts.append(
                    f"{m.method}: ({m.correction_um[0]:+.2f}, {m.correction_um[1]:+.2f}) um [{m.detail}]"
                    if m.ok
                    else f"{m.method}: FAILED ({m.detail})"
                )
            d = f"  methods differ by {s.methods_delta_um:.2f} um" if s.methods_delta_um is not None else ""
            lines.append(f"  step {s.iteration}: drove with {s.used or 'nothing'} -> error {s.error_um:.2f} um | " + " | ".join(parts) + d)
        diffs = self.method_differences()
        if diffs:
            lines.append(
                f"features vs crosses: last difference {diffs[-1]:.2f} um "
                "(= how far the earlier layer's crosses sit from where the map says that spot is)"
            )
        return lines


def align_to_target(
    rig: Rig,
    target_um: Point,
    measurers: Sequence[Measurer],
    cfg: Optional[MapConfig] = None,
    tol_um: float = 1.5,
    max_iter: int = 5,
    max_step_um: float = 80.0,
    gain: float = 1.0,
    approach_um: float = 10.0,
    autofocus: bool = True,
    prefer: Optional[Sequence[str]] = None,
    keep_frames: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> AlignResult:
    """Move to ``target_um`` and correct until the measured error is under ``tol_um``.

    ``prefer`` orders the measurers that may drive the correction (default: the order given); if the
    preferred one fails on a photo the next one takes over for that iteration, so you can switch between
    features and crosses automatically.  Every measurer is still evaluated and recorded each iteration.

    ``approach_um``: backlash handling.  Corrections smaller than the stage backlash would otherwise
    reverse direction and push the stage the wrong way (the loop then hunts).  Each move therefore
    first backs off by ``approach_um`` in -x/-y and then moves to the commanded point, so the last
    motion on both axes is always in the same direction.  Use a value larger than the stage backlash;
    0 turns it off (costs one extra short move per photo).

    ``autofocus``: if the rig has ``autofocus()``, run it after the first move to the target and after any
    later correction larger than ``cfg.af_min_move_um``; a few-um nudge does not change the focus.
    """
    cfg = cfg or MapConfig()
    note = progress or (lambda m: None)
    order = list(prefer) if prefer else [m.name for m in measurers]
    cmd: Point = (float(target_um[0]), float(target_um[1]))
    steps: List[AlignStep] = []
    frames: List[np.ndarray] = []
    per_photo = cfg.t_move_s * (2 if approach_um > 0 else 1) + cfg.t_settle_s + cfg.t_capture_s
    frame: Optional[np.ndarray] = None
    af = getattr(rig, "autofocus", None) if autofocus and cfg.autofocus else None
    af_s = 0.0
    last_focus: Optional[Point] = None

    def finish(ok: bool, msg: str, err: float) -> AlignResult:
        return AlignResult(ok, msg, target_um, cmd, err, steps, per_photo * len(steps) + af_s, frame, frames)

    prev_errs: List[float] = []
    for it in range(max_iter + 1):
        note(f"align iteration {it}: moving to ({cmd[0]:.1f}, {cmd[1]:.1f}) um")
        if approach_um > 0:
            rig.move_to(cmd[0] - approach_um, cmd[1] - approach_um)
        rig.move_to(*cmd)
        rig.settle(cfg.t_settle_s)
        if callable(af) and (last_focus is None or math.dist(cmd, last_focus) >= cfg.af_min_move_um):
            af()
            af_s += cfg.t_autofocus_s
            last_focus = cmd
        frame = rig.grab()
        if keep_frames:
            frames.append(frame)
        ms = [m.measure(frame, target_um) for m in measurers]
        by_name = {m.method: m for m in ms}
        used = next((n for n in order if n in by_name and by_name[n].ok), None)
        delta = None
        good = [m for m in ms if m.ok]
        if len(good) >= 2:
            a, b = good[0], good[1]
            delta = math.hypot(a.correction_um[0] - b.correction_um[0], a.correction_um[1] - b.correction_um[1])
        if used is None:
            steps.append(AlignStep(it, cmd, ms, "", float("nan"), delta))
            why = "; ".join(f"{m.method}: {m.detail}" for m in ms)
            return finish(False, f"no method could measure the position ({why})", float("nan"))
        m_used = by_name[used]
        err = m_used.error_um
        steps.append(AlignStep(it, cmd, ms, used, err, delta))
        if err <= tol_um:
            return finish(True, f"within {tol_um:.1f} um after {it + 1} photo(s), driven by {used}", err)
        if it == max_iter:
            return finish(False, f"still {err:.2f} um off after {max_iter} corrections (limit {tol_um:.1f} um)", err)
        prev_errs.append(err)
        if len(prev_errs) >= 3 and prev_errs[-1] > prev_errs[-2] > prev_errs[-3] and prev_errs[-1] > 2.0 * min(prev_errs):
            return finish(False, f"the error is growing ({prev_errs[-3]:.1f} -> {prev_errs[-1]:.1f} um); check the stage direction/sign", err)
        cx, cy = m_used.correction_um
        n = math.hypot(cx, cy)
        if n > max_step_um:
            cx, cy = cx * max_step_um / n, cy * max_step_um / n
        cmd = (cmd[0] + gain * cx, cmd[1] + gain * cy)
    return finish(False, "loop ended", float("nan"))  # pragma: no cover


def compare_methods(frame: np.ndarray, target_um: Point, measurers: Sequence[Measurer]) -> Dict[str, object]:
    """Measure one photo with every measurer and report how much they disagree (no stage movement)."""
    ms = [m.measure(frame, target_um) for m in measurers]
    good = [m for m in ms if m.ok]
    delta = None
    if len(good) >= 2:
        delta = math.hypot(good[0].correction_um[0] - good[1].correction_um[0], good[0].correction_um[1] - good[1].correction_um[1])
    return {"measurements": ms, "delta_um": delta}

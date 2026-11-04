"""Virtual environment for testing Will Map without the machine.

A white chip (the "3x3 area" plus a margin) with faint scratches, dust and surface texture so the
camera has something to register on, a stage with backlash / noise / drift, and a camera that
renders 1920x1080 frames of the chip at the true stage position.

Chip coordinates are um with x right, y DOWN, and (0, 0) at the centre of the area.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from will_map import CAMERA_SIZE_PX, DEFAULT_CAM_UM_PER_PX


class VirtualChip:
    """Mostly-white chip surface.  ``features=False`` gives a perfectly blank white chip."""

    def __init__(
        self,
        area_um: Tuple[float, float],
        margin_um: float = 900.0,
        res_um: float = DEFAULT_CAM_UM_PER_PX,
        seed: int = 1,
        features: bool = True,
        periodic: bool = False,
        period_um: float = 120.0,
    ):
        self.res = res_um
        self.area_um = area_um
        w_um = area_um[0] + 2 * margin_um
        h_um = area_um[1] + 2 * margin_um
        self.origin_um = (-w_um / 2.0, -h_um / 2.0)
        W, H = int(w_um / res_um), int(h_um / res_um)
        rng = np.random.default_rng(seed)
        world = np.full((H, W), 246.0, np.float32)
        if features:
            if periodic:
                # repeating grid: every shift by the pitch looks the same (ambiguity test)
                pitch = max(2, int(period_um / res_um))
                world[::pitch, :] -= 70
                world[:, ::pitch] -= 70
            else:
                tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.6)
                world += tex * 7.0
                small = np.zeros((H // 8 + 1, W // 8 + 1), np.float32)
                for _ in range(120):  # soft smudges
                    cv2.circle(small, (int(rng.integers(0, small.shape[1])), int(rng.integers(0, small.shape[0]))),
                               int(rng.integers(5, 25)), float(rng.uniform(-14, -5)), -1)
                world += cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (W, H))[:H, :W]
                canvas = np.zeros((H, W), np.uint8)
                for _ in range(110):  # scratches
                    p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
                    a = rng.uniform(0, math.pi)
                    L = rng.uniform(300, 2500)
                    q = (int(p[0] + L * math.cos(a)), int(p[1] + L * math.sin(a)))
                    cv2.line(canvas, p, q, int(rng.integers(30, 90)), int(rng.integers(1, 3)), cv2.LINE_AA)
                for _ in range(900):  # dust
                    cv2.circle(canvas, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(1, 5)),
                               int(rng.integers(25, 90)), -1, cv2.LINE_AA)
                world -= canvas.astype(np.float32)
        self.world = np.clip(world, 0, 255).astype(np.uint8)

    def paint(
        self,
        mask: np.ndarray,
        center_um: Tuple[float, float],
        proj_um_per_px: Tuple[float, float],
        crop: Optional[Tuple[int, int, int, int]] = None,
        strength: float = 110.0,
    ) -> None:
        """Darken the chip where an exposed pattern lands (a stand-in for a developed layer).

        ``mask`` is the full projector frame (bool, projector px); its centre lands at ``center_um``
        (chip um, x right / y down).  ``crop`` limits it to the part of the frame the tile exposes.
        """
        fh, fw = mask.shape[:2]
        x0, y0, x1, y1 = crop or (0, 0, fw, fh)
        sub = mask[y0:y1, x0:x1].astype(np.float32)
        a = proj_um_per_px[0] / self.res  # world px per projector px
        d = proj_um_per_px[1] / self.res
        tl_um = (center_um[0] + (x0 - fw / 2.0) * proj_um_per_px[0], center_um[1] + (y0 - fh / 2.0) * proj_um_per_px[1])
        tx = (tl_um[0] - self.origin_um[0]) / self.res
        ty = (tl_um[1] - self.origin_um[1]) / self.res
        sw, sh = max(1, int(round(sub.shape[1] * a))), max(1, int(round(sub.shape[0] * d)))
        small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)  # anti-aliased at the camera scale
        W, H = sw + 4, sh + 4
        ix, iy = int(math.floor(tx)) - 1, int(math.floor(ty)) - 1
        M = np.array([[1.0, 0.0, tx - ix], [0.0, 1.0, ty - iy]])  # sub-pixel placement
        layer = cv2.warpAffine(small, M, (W, H), flags=cv2.INTER_LINEAR)
        wh, ww = self.world.shape
        gx0, gy0 = max(0, ix), max(0, iy)
        gx1, gy1 = min(ww, ix + W), min(wh, iy + H)
        if gx1 <= gx0 or gy1 <= gy0:
            return
        sl = layer[gy0 - iy : gy1 - iy, gx0 - ix : gx1 - ix]
        reg = self.world[gy0:gy1, gx0:gx1].astype(np.float32)
        self.world[gy0:gy1, gx0:gx1] = np.clip(reg - sl * strength, 0, 255).astype(np.uint8)

    def render(
        self,
        chip_xy_um: Tuple[float, float],
        cam_size: Tuple[int, int] = CAMERA_SIZE_PX,
        true_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
        rot_deg: float = 0.0,
    ) -> np.ndarray:
        W, H = cam_size
        a = true_um_per_px / self.res
        th = math.radians(rot_deg)
        c, s = math.cos(th), math.sin(th)
        lin = np.array([[a * c, -a * s], [a * s, a * c]])
        centre_world = np.array(
            [(chip_xy_um[0] - self.origin_um[0]) / self.res, (chip_xy_um[1] - self.origin_um[1]) / self.res]
        )
        b = centre_world - lin @ np.array([W / 2.0, H / 2.0])
        M = np.hstack([lin, b[:, None]])
        return cv2.warpAffine(
            self.world, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )


class VirtualRig:
    """Stage + camera over a VirtualChip.  Implements the ``Rig`` protocol from will_map.

    The stage obeys the old-UI convention (stage +x = right, stage +y = up) and has:
    * ``backlash_um``   lost motion that flips each time an axis reverses direction,
    * ``noise_um``      random placement error per move,
    * ``drift_um``      slow drift per move (thermal),
    * ``chip_offset_um``  where the chip really is relative to where the stage thinks the centre is,
    * ``true_um_per_px`` / ``rot_deg``  what the camera really does (the config may be wrong).
    """

    def __init__(
        self,
        chip: VirtualChip,
        center_um: Tuple[float, float] = (0.0, 0.0),
        chip_offset_um: Tuple[float, float] = (140.0, -85.0),
        sign: Tuple[int, int] = (1, -1),
        true_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
        rot_deg: float = 0.0,
        noise_um: float = 0.8,
        backlash_um: float = 6.0,
        drift_um: Tuple[float, float] = (0.12, 0.07),  # thermal drift in um per simulated second of settling
        sensor_noise: float = 1.5,
        seed: int = 0,
        cam_size: Tuple[int, int] = CAMERA_SIZE_PX,
        z_drift_per_um: float = 0.012,
        blur_px_per_um: float = 0.25,
    ):
        self.chip = chip
        self.center = center_um
        self.chip_offset = chip_offset_um
        self.sign = sign
        self.true_um_per_px = true_um_per_px
        self.rot_deg = rot_deg
        self.noise_um = noise_um
        self.backlash_um = backlash_um
        self.drift_um = drift_um
        self.sensor_noise = sensor_noise
        self.cam_size = cam_size
        self.rng = np.random.default_rng(seed)
        self.cmd = list(center_um)
        self.actual = list(center_um)
        self._dir = [1, 1]
        self._moves = 0
        self.sim_settle_s = 0.0
        # focus model: moving the stage makes the chip drift out of focus (proportional to the travel), the
        # photo blurs with the focus error, and autofocus() puts it back
        self.z_drift_per_um = z_drift_per_um
        self.blur_px_per_um = blur_px_per_um
        self.z_err = 0.0
        self.af_calls = 0
        self.log: List[dict] = []  # one entry per grab: commanded stage and true chip position

    # ---- Rig protocol ----
    def move_to(self, x_um: float, y_um: float) -> None:
        target = [x_um, y_um]
        for ax in (0, 1):
            d = target[ax] - self.cmd[ax]
            if abs(d) > 1e-9:
                self._dir[ax] = 1 if d > 0 else -1
        self._moves += 1
        travel = math.hypot(target[0] - self.cmd[0], target[1] - self.cmd[1])
        self.z_err += float(self.rng.normal(0, self.z_drift_per_um * travel + 0.2))
        self.cmd = target
        self.actual = [
            target[ax]
            - self._dir[ax] * self.backlash_um / 2.0
            + float(self.rng.normal(0, self.noise_um))
            for ax in (0, 1)
        ]

    def settle(self, seconds: float) -> None:
        self.sim_settle_s += seconds  # simulated, no real waiting

    def autofocus(self) -> None:
        self.z_err = float(self.rng.normal(0, 0.4))  # residual focus error after autofocus
        self.af_calls += 1

    def grab(self) -> np.ndarray:
        chip_xy = self.true_chip_xy()
        g = self.chip.render(chip_xy, self.cam_size, self.true_um_per_px, self.rot_deg).astype(np.float32)
        blur = self.blur_px_per_um * abs(self.z_err)
        if blur > 0.3:
            g = cv2.GaussianBlur(g, (0, 0), blur)
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        vign = 1.0 - 0.06 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        g = g * vign + self.rng.normal(0, self.sensor_noise, g.shape)
        g = np.clip(g, 0, 255).astype(np.uint8)
        self.log.append({"cmd": tuple(self.cmd), "chip_um": chip_xy})
        return np.dstack([g, g, g])  # the old UI's camera_image is 3-channel RGB

    def expose(
        self,
        mask: np.ndarray,
        proj_um_per_px: Tuple[float, float],
        crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[float, float]:
        """Pretend to expose one pattern tile with the stage where it really is now.  Returns the true
        chip position (um) the frame centre landed on."""
        xy = self.true_chip_xy()
        self.chip.paint(mask, xy, proj_um_per_px, crop)
        return xy

    def ideal_chip_xy(self, target_um: Tuple[float, float]) -> Tuple[float, float]:
        """Where the chip would be under the camera if the stage reached ``target_um`` exactly."""
        return (
            self.sign[0] * (target_um[0] - self.center[0]) + self.chip_offset[0],
            self.sign[1] * (target_um[1] - self.center[1]) + self.chip_offset[1],
        )

    # ---- ground truth ----
    def true_chip_xy(self) -> Tuple[float, float]:
        t = self.sim_settle_s  # drift grows with (simulated) time, not with the number of moves
        return (
            self.sign[0] * (self.actual[0] + self.drift_um[0] * t - self.center[0]) + self.chip_offset[0],
            self.sign[1] * (self.actual[1] + self.drift_um[1] * t - self.center[1]) + self.chip_offset[1],
        )

    def true_shift_px(self, i: int, j: int) -> Tuple[float, float]:
        """True position of capture j's frame origin in capture i's frame, in camera px."""
        a, b = self.log[i]["chip_um"], self.log[j]["chip_um"]
        th = math.radians(self.rot_deg)
        dx, dy = b[0] - a[0], b[1] - a[1]
        # express the chip displacement in camera axes (camera rotated by +th relative to the chip)
        ux = (dx * math.cos(th) + dy * math.sin(th)) / self.true_um_per_px
        uy = (-dx * math.sin(th) + dy * math.cos(th)) / self.true_um_per_px
        return (ux, uy)


# --------------------------------------------------------------------------------------
# A whole virtual session: map -> placement -> (closed-loop) patterning -> finished piece
# --------------------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "will_map_samples")
# reading order TL, TR, BL, BR (the top-left tile was not supplied: crosses-only stand-in)
SAMPLE_SET = ["pattern_4_crosses_only.png", "pattern_3.png", "pattern_2.png", "pattern_1.png"]


def simulate_patterning(
    rig: "VirtualRig",
    mr,
    pm,
    images,
    cfg,
    center_um: Tuple[float, float],
    closed_loop: bool = True,
    recorder=None,
    progress=None,
    use_crosses: bool = False,
    ml_detector=None,
) -> dict:
    """Move to every tile target (optionally with the closed-loop alignment) and 'expose' it onto the
    virtual chip.  Returns the true tile positions and how well the tile spacing came out.

    ``use_crosses``: tiles that have a left/top neighbour find some of their crosses already printed by
    an earlier tile; with this on, the alignment uses those crosses (cross detection on).  Off, the
    alignment only uses the chip's own features (convolution against the map photos)."""
    from will_align import CrossMeasurer, FeatureMeasurer, align_to_target, cross_positions_um, earlier_crosses
    from will_map import cross_patch, exposure_mask, status

    note = progress or (lambda m: None)
    targets = pm.stage_targets(center_um, cfg.sign)
    fm = FeatureMeasurer(mr) if closed_loop else None
    tmpl = cross_patch(images[-1], pm.geo, "tl")
    pos, photos, aligns, used = [], 0, [], []
    t_pat = time.perf_counter()
    status(f"patterning: {len(pm.tiles)} tiles ({'closed-loop alignment' if closed_loop else 'open loop'})")
    for t, T in zip(pm.tiles, targets):
        note(f"tile {t.slot + 1}: {'aligning' if closed_loop else 'moving'}")
        if closed_loop:
            measurers = [fm]
            names = earlier_crosses(pm, t.slot) if use_crosses else []
            if len(names) >= 2:
                exp = cross_positions_um(pm.geo, T, pm.um_per_px, cfg.sign, names=names)
                measurers = [CrossMeasurer(tmpl, pm.geo.center_in_cross, exp, cfg, pm.um_per_px, ml_detector=ml_detector), fm]
            ar = align_to_target(rig, T, measurers, cfg)
            photos += len(ar.steps)
            aligns.append(ar)
            used.append(ar.steps[-1].used if ar.steps else "")
            if recorder is not None:
                recorder.save_alignment(ar, f"tile{t.slot + 1}")
        else:
            rig.move_to(*T)
            rig.settle(cfg.t_settle_s)
            if hasattr(rig, "autofocus") and cfg.autofocus:
                rig.autofocus()
        note(f"tile {t.slot + 1}: exposing (virtual)")
        pos.append(rig.expose(exposure_mask(images[t.slot]), pm.um_per_px, t.crop))
        status(f"patterning: tile {len(pos)}/{len(pm.tiles)} done (slot {t.slot + 1}), {time.perf_counter() - t_pat:.0f} s elapsed")
    ideal = [rig.ideal_chip_xy(T) for T in targets]
    spacing = [
        math.hypot(
            (pos[k][0] - pos[0][0]) - (ideal[k][0] - ideal[0][0]),
            (pos[k][1] - pos[0][1]) - (ideal[k][1] - ideal[0][1]),
        )
        for k in range(1, len(pos))
    ]
    # what stitching cares about: do NEIGHBOURING tiles sit where they should relative to each other?
    seams = []
    for t in pm.tiles:
        for dr, dc in ((0, 1), (1, 0)):
            nb = next((u for u in pm.tiles if (u.row, u.col) == (t.row + dr, t.col + dc)), None)
            if nb is not None:
                seams.append(
                    math.hypot(
                        (pos[nb.slot][0] - pos[t.slot][0]) - (ideal[nb.slot][0] - ideal[t.slot][0]),
                        (pos[nb.slot][1] - pos[t.slot][1]) - (ideal[nb.slot][1] - ideal[t.slot][1]),
                    )
                )
    return {
        "true_positions_um": pos, "spacing_error_um": spacing, "seam_error_um": seams, "photos": photos,
        "alignments": aligns, "driven_by": used, "alignment_s": sum(a.stage_s for a in aligns),
    }


def load_samples():
    from will_map import load_image

    paths = [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]
    return paths, [load_image(p) for p in paths]


def run_process(
    rig: "VirtualRig",
    rec,
    center_um: Tuple[float, float],
    paths,
    images,
    tile_rows: int = 2,
    tile_cols: int = 2,
    use_crosses: bool = False,
    closed_loop: bool = True,
    ml_detector=None,
    progress=None,
) -> dict:
    """One complete pass on one area: size the area from the tile counts (+1 frame each way) -> map it ->
    only if the map is good, place the tiles -> align + (virtually) expose each tile -> map the finished piece.
    Everything is saved into ``rec``'s folder."""
    from will_map import MapConfig, area_for_tiles, build_placement_map, measure_cross_geometry, will_map
    from will_map_runs import map_finished_piece

    note = progress or (lambda m: None)
    cfg = MapConfig()
    area_pm = area_for_tiles(tile_rows, tile_cols)  # sized from the CROSS_* inputs, before any pattern is chosen
    note("mapping the white area")
    mr = will_map(rig, cfg, area_pm.area_size_um, center_um)
    rec.save_map(mr)
    out = {"recorder": rec, "map": mr, "area_um": area_pm.area_size_um, "center_um": center_um, "rig": rig}
    if not mr.ok:
        return out
    # tiles can only be inserted now that the area is mapped
    geo = measure_cross_geometry(images[-1])
    pm = build_placement_map(geo, tile_rows, tile_cols, tile_rows + 1, tile_cols + 1)
    out["placement"] = pm
    rec.save_placement(pm, images, paths, center_um, mr)
    out["patterning"] = simulate_patterning(
        rig, mr, pm, images, cfg, center_um, closed_loop, rec, progress, use_crosses, ml_detector
    )
    note("mapping the finished piece")
    out["finished"] = map_finished_piece(rig, cfg, pm.area_size_um, center_um, rec, pm)
    rec.log(
        "tile spacing error vs ideal (um): " + ", ".join(f"{v:.2f}" for v in out["patterning"]["spacing_error_um"])
        + f"  (cross detection {'ON' if use_crosses else 'OFF'}; tiles driven by {out['patterning']['driven_by']})"
    )
    return out


def run_sample_comparison(
    base_dir: Optional[str] = None,
    label: str = "sample",
    seed: int = 5,
    tile_rows: int = 2,
    tile_cols: int = 2,
    gap_um: float = 400.0,
    ml_model=None,
    progress=None,
    **rig_kw,
) -> dict:
    """The sample button: the whole process twice on one chip, on two different areas so the patterns do not
    overlap.  Area A uses convolution against the map photos only (cross detection off); area B also uses
    the crosses that earlier tiles have already printed (cross detection on, with the YOLO model as a seed
    when one is given).  Both finished pieces are mapped and saved."""
    from will_map import area_for_tiles
    from will_map_runs import RunRecorder

    paths, images = load_samples()
    if len(images) != tile_rows * tile_cols:
        raise ValueError("the bundled samples are a 2x2 pattern")
    w, h = area_for_tiles(tile_rows, tile_cols).area_size_um
    d = (w + gap_um) / 2.0
    centre = (5000.0, 3000.0)
    cA, cB = (centre[0] - d, centre[1]), (centre[0] + d, centre[1])
    rig = VirtualRig(VirtualChip((2 * w + gap_um, h), seed=11), center_um=centre, seed=seed, **rig_kw)
    top = RunRecorder(base_dir, label)
    ml = None
    if ml_model is not None:
        from will_align import yolo_detector

        ml = yolo_detector(ml_model)
    runs = {}
    for key, name, c, crosses in (
        ("A", "A_convolution_only", cA, False),
        ("B", "B_cross_detection", cB, True),
    ):
        runs[key] = run_process(
            rig, top.sub(name), c, paths, images, tile_rows, tile_cols, use_crosses=crosses,
            ml_detector=ml if crosses else None,
            progress=(lambda m, n=name: progress(f"{n}: {m}")) if progress else None,
        )
    separation = abs(cB[0] - cA[0])
    out = {"recorder": top, "runs": runs, "area_um": (w, h), "separation_um": separation, "rig": rig}
    out["areas_overlap"] = separation < w
    lines = [
        f"Two complete runs on one chip, on different areas (centres {separation:.0f} um apart, each area {w:.0f} x {h:.0f} um): "
        + ("OVERLAP!" if out["areas_overlap"] else "no overlap"),
        "",
    ]
    for key, label_ in (("A", "A  convolution only (cross detection OFF)"), ("B", "B  cross detection ON")):
        r = runs[key]
        if "patterning" not in r:
            lines.append(f"{label_}: map failed: {r['map'].message}")
            continue
        pat, fin = r["patterning"], r["finished"]
        sp = pat["spacing_error_um"]
        lines += [
            label_,
            f"  map: {r['map'].timing['total_s']:.0f} s (limit 300), {r['map'].captured} photos, {int(r['map'].timing.get('autofocus_runs', 0))} autofocus",
            f"  tile position error vs the ideal grid, relative to tile 1 (um): {', '.join(f'{v:.2f}' for v in sp)}  (mean {sum(sp) / len(sp):.2f}, worst {max(sp):.2f})",
            f"  seam error between neighbouring tiles (um): {', '.join(f'{v:.2f}' for v in pat['seam_error_um'])}  (mean {sum(pat['seam_error_um']) / len(pat['seam_error_um']):.2f}, worst {max(pat['seam_error_um']):.2f})",
            f"  alignment: {pat['photos']} photos, ~{pat['alignment_s']:.0f} s; tiles driven by {pat['driven_by']}",
            f"  finished piece mapped: {'ok' if fin.ok else 'FAILED - ' + fin.message}  ({fin.timing['total_s']:.0f} s)",
        ]
    with open(os.path.join(top.dir, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out["summary"] = lines
    _save_overview(top, runs)
    top.log("sample comparison finished")
    return out


def _save_overview(top, runs) -> None:
    """Both finished pieces side by side (stitched, with the expected tile outlines)."""
    from will_map import PlacementMap  # noqa: F401
    from will_map_runs import _write_png_bgr, draw_expected_outline

    panels = []
    for key, title in (("A", "A: convolution only"), ("B", "B: cross detection on")):
        r = runs[key]
        if "finished" not in r or not r["finished"].ok:
            continue
        img = draw_expected_outline(r["finished"], r["placement"], r["center_um"])
        img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 200), 3, cv2.LINE_AA)
        panels.append(img)
    if panels:
        h = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
        _write_png_bgr(os.path.join(top.dir, "both_finished_pieces.png"), np.hstack(panels))


def run_virtual_session(
    base_dir: Optional[str] = None,
    label: str = "virtual",
    closed_loop: bool = True,
    seed: int = 5,
    center_um: Tuple[float, float] = (5000.0, 3000.0),
    use_crosses: bool = False,
    progress=None,
    **rig_kw,
) -> dict:
    """One pass on one area (kept for the single-run demo)."""
    from will_map import area_for_tiles
    from will_map_runs import RunRecorder

    paths, images = load_samples()
    w, h = area_for_tiles(2, 2).area_size_um
    rig = VirtualRig(VirtualChip((w, h), seed=11), center_um=center_um, seed=seed, **rig_kw)
    rec = RunRecorder(base_dir, label)
    return run_process(rig, rec, center_um, paths, images, 2, 2, use_crosses, closed_loop, None, progress)


if __name__ == "__main__":
    res = run_sample_comparison(label="cli")
    print("run folder:", res["recorder"].dir)
    print("\n".join(res["summary"]))

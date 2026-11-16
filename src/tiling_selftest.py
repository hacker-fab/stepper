"""Self-test for the camera map, tile placement and closed-loop alignment, run against the virtual stage
(stage_control/virtual_stage.py, a white chip; no hardware needed).

    python src/tiling_selftest.py                 # all tests (UI tests run if a display exists)
    python src/tiling_selftest.py --visuals OUT   # also write the explanatory pictures to OUT/
    python src/tiling_selftest.py --demo          # the whole flow in a window, against the virtual stage
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from typing import Callable, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib.run_folder as _run_folder  # noqa: E402
from lib.img import (  # noqa: E402
    exposure_mask,
    load_image,
)
from lib.mapping import (  # noqa: E402
    MapConfig,
    MapResult,
    map_area,
    scan_area_um,
    scan_summary,
    snake_order,
)
from lib.tiling import (  # noqa: E402
    CROSS_DIST_BOTTOM_PX,
    CROSS_DIST_LEFT_PX,
    CROSS_DIST_RIGHT_PX,
    CROSS_DIST_TOP_PX,
    CROSS_HEIGHT_PX,
    CROSS_WIDTH_PX,
    PROJ_UM_PER_PX,
    SAMPLE_DIR,
    SAMPLE_SET,
    CrossGeometry,
    TilingTimeModel,
    area_for_tiles,
    budget_report,
    build_placement_map,
    check_seams,
    cross_geometry_for,
    cross_patch,
    measure_cross_geometry,
    render_placement,
    same_layout,
    spec_geometry,
    spec_mismatches,
)
from stage_control.virtual_stage import VirtualChip, VirtualRig  # noqa: E402

# the tests must never read or overwrite this machine's real calibration (src/map_runs/calibration.json)
_run_folder.CALIBRATION_FILE = os.path.join(__import__("tempfile").mkdtemp(), "calibration.json")

HERE = os.path.dirname(os.path.abspath(__file__))
CENTER = (5000.0, 3000.0)  # pretend stage position at the middle of the area


def _load_samples() -> Tuple[List[np.ndarray], CrossGeometry]:
    p = [load_image(os.path.join(SAMPLE_DIR, f"pattern_{i}.png")) for i in (1, 2, 3)]
    cross = load_image(os.path.join(SAMPLE_DIR, "pattern_4_crosses_only.png"))
    geo = measure_cross_geometry(p[2])
    return [cross, p[2], p[1], p[0]], geo  # reading order TL, TR, BL, BR


# ---------------------------------------------------------------- tests
def test_geometry() -> None:
    imgs, geo = _load_samples()
    for img in imgs:
        assert same_layout(measure_cross_geometry(img), geo), "all tiles must share one cross layout"
    assert (geo.frame_w, geo.frame_h) == (3840, 2160)
    assert (geo.cross_w, geo.cross_h) == (314, 198)
    assert geo.tl == (17, 17)
    assert (geo.pitch_x, geo.pitch_y) == (3492, 1928)
    assert (geo.overlap_x, geo.overlap_y) == (348, 232)
    sizes = lambda r, c: (geo.footprint(r, c)[2] - geo.footprint(r, c)[0], geo.footprint(r, c)[3] - geo.footprint(r, c)[1])
    assert sizes(0, 0) == (3806, 2126)  # left+top: both crosses, whole footprint
    assert sizes(0, 1) == (3492, 2126)  # has a left neighbour: left cross dropped
    assert sizes(1, 0) == (3806, 1928)  # has a top neighbour: top cross dropped
    assert sizes(1, 1) == (3492, 1928)  # both dropped, and equals the pitch
    assert sizes(2, 2) == sizes(1, 1)


def test_placement() -> None:
    imgs, geo = _load_samples()
    pm = build_placement_map(geo)
    assert pm.area_size_px == (10790, 5982)
    t0, t1, t2, t3 = pm.tiles
    # 2x2 centred in 3x3: equal margins, each exactly half a pitch
    assert t0.dest[0] - pm.area_box[0] == pm.area_box[2] - t1.dest[2] == geo.pitch_x // 2
    assert t0.dest[1] - pm.area_box[1] == pm.area_box[3] - t2.dest[3] == geo.pitch_y // 2
    # shared crosses coincide: right cross of the left tile == left cross of the right tile
    assert t1.frame_origin[0] + geo.tl[0] == t0.frame_origin[0] + geo.tr[0]
    assert t2.frame_origin[1] + geo.tl[1] == t0.frame_origin[1] + geo.bl[1]
    # footprints tile the 2x2 region with no gap and no overlap
    assert t0.dest[2] == t1.dest[0] and t0.dest[3] == t2.dest[1]
    assert t2.dest[2] == t3.dest[0] and t1.dest[3] == t3.dest[1]
    # stage targets: symmetric about the centre, step = pitch in um
    tg = pm.stage_targets(CENTER)
    assert abs((tg[1][0] - tg[0][0]) - pm.step_um[0]) < 1e-6 and abs((tg[0][1] - tg[2][1]) - pm.step_um[1]) < 1e-6
    assert abs(pm.step_um[0] - 943.0) < 0.5 and abs(pm.step_um[1] - 520.4) < 0.5
    assert abs(np.mean([t[0] for t in tg]) - CENTER[0]) < 1e-6 and abs(np.mean([t[1] for t in tg]) - CENTER[1]) < 1e-6
    # stage +y is up and mosaic y is down: tile row 1 (top) has the HIGHER stage y
    assert tg[0][1] > tg[2][1]
    # nudging +10 um right / +20 um down moves stage x by +10 and stage y by -20
    pn = build_placement_map(geo, nudge_um=(10.0, 20.0))
    tn = pn.stage_targets(CENTER)
    assert abs(tn[0][0] - tg[0][0] - 10) < 0.3 and abs(tn[0][1] - tg[0][1] + 20) < 0.3
    for mode in ("trimmed", "whole"):
        img, _ = render_placement(build_placement_map(geo, mode=mode), imgs)
        assert img.ndim == 3 and img.shape[0] > 100


def test_seams() -> None:
    imgs, geo = _load_samples()
    assert all(s["ok"] for s in check_seams(imgs, 2, 2, geo)), "sample set must line up"
    swapped = check_seams([imgs[0], imgs[2], imgs[1], imgs[3]], 2, 2, geo)
    assert any(not s["ok"] for s in swapped), "a wrong slot order must be flagged"


def _area() -> Tuple[float, float]:
    return build_placement_map(_load_samples()[1]).area_size_um


def _run(chip: VirtualChip, cfg: MapConfig = None, **rig_kw):
    cfg = cfg or MapConfig()
    rig = VirtualRig(chip, center_um=CENTER, seed=5, **rig_kw)
    return rig, cfg, map_area(rig, cfg, _area(), CENTER)


def _max_err_px(rig: VirtualRig, cfg: MapConfig, res: MapResult) -> float:
    cfg = res.cfg  # the grid size is worked out from the area when the config leaves rows/cols empty
    order = snake_order(cfg.rows, cfg.cols)
    a = res.positions_px[order[0]]
    errs = []
    for k, t in enumerate(order):
        tx, ty = rig.true_shift_px(0, k)
        errs.append(max(abs(res.positions_px[t][0] - a[0] - tx), abs(res.positions_px[t][1] - a[1] - ty)))
    return max(errs)


def test_map_happy_path() -> None:
    chip = VirtualChip(_area(), seed=11)
    rig, cfg, res = _run(chip)
    assert res.ok, res.message
    assert res.captured == 9 and len(res.pairs) == 10 and len(res.checks) == 2
    assert all(c.passed for c in res.checks) and all(c.delta_um < 1.5 for c in res.checks)
    assert _max_err_px(rig, cfg, res) < 1.0, "tile positions should land within a pixel of the truth"
    rep = budget_report(res.timing["total_s"])
    assert rep["map_ok"], "map must fit the 5 minute budget"
    # the check is on the 2nd tile of each new row: rows 2 and 3, column 2 (index 1)
    assert [c.tile for c in res.checks] == [(1, 1), (2, 1)]


def test_map_tolerates_a_bad_config_and_rough_stage() -> None:
    chip = VirtualChip(_area(), seed=11)
    rig, cfg, res = _run(chip, true_um_per_px=0.652, rot_deg=0.6)  # config says 0.678, camera is 4% off and rotated
    assert res.ok and _max_err_px(rig, cfg, res) < 1.5, res.message
    rig, cfg, res = _run(chip, backlash_um=30.0, noise_um=4.0)
    assert res.ok and _max_err_px(rig, cfg, res) < 1.5, res.message


def test_map_uses_the_stage_when_it_cannot_trust_the_registration() -> None:
    blank = VirtualChip(_area(), seed=11, features=False)
    periodic = VirtualChip(_area(), seed=11, periodic=True, period_um=50.0)
    for chip, why in ((blank, "weak match"), (periodic, "ambiguous")):
        # default: the map completes, every untrusted overlap is placed from the stage and reported
        rig, cfg, res = _run(chip)
        est = [p for p in res.pairs if p.estimated]
        assert res.ok and est and "stage position" in res.message, res.message
        assert all(why in p.reason and p.measured_px == p.predicted_px for p in est), [p.reason for p in est]
        assert any("stage estimate" in line for line in res.report())
        # strict mode keeps the old behaviour: stop the map
        _, _, res = _run(chip, cfg=MapConfig(stage_fallback=False))
        assert not res.ok and why in res.message, res.message


def test_left_top_check_catches_a_false_lock() -> None:
    import lib.mapping as mapping  # map_area looks register_pair up in its own module

    chip = VirtualChip(_area(), seed=11)
    orig = mapping.register_pair
    state = {"n": 0}

    def make(hits):
        def patched(*a, **k):
            res = orig(*a, **k)
            state["n"] += 1
            if state["n"] in hits:  # call 5 = the 'top' registration of tile (row 2, col 2)
                res.shift = (res.shift[0] + 12.0, res.shift[1])
            return res

        return patched

    try:
        mapping.register_pair = make({5})  # wrong once: caught, re-captured, then fine
        _, _, res = _run(chip)
        assert res.ok, res.message
        state["n"] = 0
        mapping.register_pair = make({5, 7})  # wrong on the retry too: that tile is placed from the stage
        _, _, res = _run(chip)
        failed = [c for c in res.checks if not c.passed]
        assert res.ok and len(failed) == 1 and failed[0].delta_um > 3.0, res.report()
        bad = failed[0].tile
        assert all(p.estimated for p in res.pairs if p.b == bad) and not any(p.estimated for p in res.pairs if p.b != bad)
        state["n"] = 0
        mapping.register_pair = make({5, 7})  # strict mode: the map stops
        _, _, res = _run(chip, cfg=MapConfig(stage_fallback=False))
        assert not res.ok and "left and top disagree" in res.message, res.message
    finally:
        mapping.register_pair = orig


def test_check_every_tile_option() -> None:
    chip = VirtualChip(_area(), seed=11)
    rig, cfg, res = _run(chip, cfg=MapConfig(check_all=True))
    assert res.ok and len(res.checks) == 4  # (2,2) (2,3) (3,2) (3,3)
    assert res.timing["compute_s"] < 10, "registration must stay cheap"


def test_budget() -> None:
    rep = budget_report(map_s=40.0)
    assert rep["map_ok"] and abs(rep["tile_overhead_s"] - 4 * TilingTimeModel().overhead_s) < 1e-9
    assert rep["max_exposure_per_tile_s"] > 100
    full = budget_report(40.0, exposure_s=100.0)
    assert full["tile_ok"] and full["total_ok"]
    assert not budget_report(40.0, exposure_s=200.0)["tile_ok"]


def test_map_measures_the_real_camera_scale_and_rotation() -> None:
    """The stepper's camera turned out 7-14 % off the configured 0.678 um/px and 3-5 deg rotated: the map
    measures that from its own overlaps, and closed-loop patterning places the tiles right anyway."""
    import lib.mapping as mapping

    imgs, geo = _load_samples()
    pm = build_placement_map(geo)
    cfg = MapConfig()  # still says 0.678 um/px, no rotation
    rig = VirtualRig(VirtualChip(pm.area_size_um, seed=11), center_um=CENTER, seed=4, true_um_per_px=0.62, rot_deg=3.5)
    mr = map_area(rig, cfg, pm.area_size_um, CENTER)
    assert mr.ok and mr.stage_px is not None, mr.message
    m = np.array(mr.stage_px).reshape(2, 2)
    for col in (0, 1):
        assert abs(1 / np.linalg.norm(m[:, col]) - 0.62) < 0.01, mapping.describe_stage_px(mr.stage_px)
    assert abs(abs(math.degrees(math.atan2(m[1, 0], m[0, 0]))) - 3.5) < 0.5, mapping.describe_stage_px(mr.stage_px)
    assert any("camera vs stage, measured" in line for line in mr.report())
    from stage_control.virtual_stage import simulate_patterning

    sim = simulate_patterning(rig, mr, pm, imgs, cfg, CENTER, closed_loop=True)
    assert max(sim["spacing_error_um"]) < 3.0, sim["spacing_error_um"]
    # a later map starts from the measured matrix: its predictions are close from the first overlap on
    cfg2 = MapConfig(stage_px=mr.stage_px)
    rig2 = VirtualRig(VirtualChip(pm.area_size_um, seed=11), center_um=CENTER, seed=6, true_um_per_px=0.62, rot_deg=3.5)
    mr2 = map_area(rig2, cfg2, pm.area_size_um, CENTER)
    first = mr2.pairs[0]
    assert math.hypot(*first.error_px) < 15, first.error_px


def test_focus_prediction_and_missed_focus() -> None:
    from lib.mapping import focus_missed, predict_focus_z

    assert predict_focus_z([], 0, 0) is None
    assert predict_focus_z([(0, 0, 10.0)], 500, 0) == 10.0
    # a row of points (first map row): slope along the row, none across it
    row = [(0, 0, 10.0), (500, 0, 15.0), (1000, 0, 20.0)]
    assert abs(predict_focus_z(row, 1500, 0) - 25.0) < 1e-6
    assert abs(predict_focus_z(row, 1000, 400) - 20.0) < 1e-6
    # a tilted plane once the points span an area (clipped to +-40 um from the nearest point)
    plane = row + [(0, 400, 12.0), (500, 400, 17.0)]
    assert abs(predict_focus_z(plane, 1000, 400) - 22.0) < 1e-6
    assert abs(predict_focus_z(plane, 20000, 0) - 60.0) < 1e-6
    usual = [100.0, 104.0, 98.0]
    assert focus_missed(5.0, 101.0, 0.0, 10.0, 0.5, usual) is None
    assert "edge" in focus_missed(9.6, 101.0, 0.0, 10.0, 0.5, usual)
    assert "sharpness" in focus_missed(2.0, 40.0, 0.0, 10.0, 0.5, usual)
    assert focus_missed(2.0, 40.0, 0.0, 10.0, 0.5, []) is None  # nothing to compare with yet


def test_projector_calibration_from_the_autofocus_grid() -> None:
    """The autofocus grid, projected through a turned / scaled / mirrored / shifted optical path, is found in
    the photo and the fitted projector -> camera mapping matches the one used to make the photo."""
    from lib.img import aruco_grid_image, find_aruco_grid
    from lib.mapping import fit_projector_calibration

    proj_size, cam = (1920, 1080), (1360, 929)
    grid = aruco_grid_image(4, proj_size).astype(np.float32)
    for rot, mirror in ((2.8, False), (-5.0, True)):
        th = math.radians(rot)
        r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        a = r @ np.diag([1300.0 * (-1 if mirror else 1), 735.0])  # frame fraction -> photo px
        b = np.array([cam[0] / 2 + 23.0, cam[1] / 2 - 17.0]) - a @ np.array([0.5, 0.5])
        m = np.c_[a @ np.diag([1 / proj_size[0], 1 / proj_size[1]]), b]  # projector px -> photo px
        photo = cv2.warpAffine(grid, m, cam, flags=cv2.INTER_AREA)
        photo = 60 + 0.55 * photo + np.random.default_rng(1).normal(0, 4, photo.shape)  # chip + red light
        photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (0, 0), 1.0)
        cal = fit_projector_calibration(*find_aruco_grid(photo, 4, proj_size), cam)
        assert cal is not None and cal.n_points >= 4 * 12, (rot, None if cal is None else cal.n_points)
        assert np.abs(np.array(cal.a) - a.ravel()).max() < 3.0, (cal.a, a.ravel())
        assert np.abs(np.array(cal.b) - b).max() < 2.0, (cal.b, b)
        off = cal.frame_centre_offset_px()
        assert abs(off[0] - 23.0) < 1.0 and abs(off[1] + 17.0) < 1.0, off


def test_ui() -> None:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:  # no display or no tkinter
        print("    (skipped: no display / tkinter)")
        return
    import gui as ui

    root.withdraw()
    samples = [os.path.join(ui.SAMPLE_DIR, f) for f in ui.SAMPLE_SET]
    d = ui.PatternPickerDialog(root, 2, 2, samples)
    assert d.next_btn.instate(["disabled"])
    d.use_samples()
    assert d.current == 3 and d.next_btn.cget("text") == "Finish"
    d.next()
    assert d.result and len(d.result) == 4
    imgs = [load_image(p) for p in d.result]
    got: list = []
    w = ui.PlacementApprovalWindow(root, imgs, d.result, area_center_um=CENTER, on_approve=got.append)
    w.mode_var.set("whole")
    w.refresh()
    w.mode_var.set("trimmed")
    w.move(1, 0)
    w.approve()
    assert got and len(got[0].stage_targets) == 4 and got[0].placement.nudge_um[0] == 10.0
    root.destroy()


# ---------------------------------------------------------------- cross inputs, alignment, run folders
def test_cross_inputs() -> None:
    imgs, geo = _load_samples()
    spec = spec_geometry()
    assert same_layout(spec, geo), "the CROSS_* inputs must be autofilled with the measured values"
    assert (CROSS_DIST_LEFT_PX, CROSS_WIDTH_PX, CROSS_DIST_TOP_PX) == (17, 314, 17)
    assert (CROSS_DIST_RIGHT_PX, CROSS_HEIGHT_PX, CROSS_DIST_BOTTOM_PX) == (17, 198, 17)
    assert all(spec_mismatches(measure_cross_geometry(i)) == [] for i in imgs)
    shifted = np.roll(imgs[1], 9, axis=1)  # a pattern whose crosses sit 9 px further right
    notes = spec_mismatches(measure_cross_geometry(shifted))
    assert any("left edge" in n for n in notes) and any("right edge" in n for n in notes), notes
    blank = np.zeros_like(imgs[1])
    assert same_layout(cross_geometry_for(blank), geo), "no crosses found -> falls back to the typed-in layout"


def _session(closed: bool, seed: int):
    import lib.alignment as alignment
    from stage_control.virtual_stage import simulate_patterning

    imgs, geo = _load_samples()
    pm = build_placement_map(geo)
    cfg = MapConfig()
    chip = VirtualChip(pm.area_size_um, seed=11)
    rig = VirtualRig(chip, center_um=CENTER, seed=seed)
    mr = map_area(rig, cfg, pm.area_size_um, CENTER)
    assert mr.ok, mr.message
    sim = simulate_patterning(rig, mr, pm, imgs, cfg, CENTER, closed_loop=closed)
    return imgs, geo, pm, cfg, rig, mr, sim, alignment


def test_align_closed_loop_beats_open_loop() -> None:
    o_all, c_all = [], []
    for seed in (1, 2, 3):
        o = _session(False, seed)[6]["spacing_error_um"]
        _, _, _, _, _, _, sim, _ = _session(True, seed)
        o_all += o
        c_all += sim["spacing_error_um"]
        assert sim["photos"] <= 4 * 6, "alignment must converge in a handful of photos per tile"
        assert all(a.ok for a in sim["alignments"]), [a.message for a in sim["alignments"] if not a.ok]
    assert float(np.mean(c_all)) < 2.0 and max(c_all) < 3.5, c_all
    assert float(np.mean(o_all)) > 2.0 * float(np.mean(c_all)), (np.mean(o_all), np.mean(c_all))


def test_align_backlash_hunting_is_prevented() -> None:
    import lib.alignment as alignment

    imgs, geo = _load_samples()
    pm = build_placement_map(geo)
    chip = VirtualChip(pm.area_size_um, seed=11)
    rig = VirtualRig(chip, center_um=CENTER, seed=2, backlash_um=6.0)
    cfg = MapConfig()
    mr = map_area(rig, cfg, pm.area_size_um, CENTER)
    fm = alignment.FeatureMeasurer(mr)
    T = pm.stage_targets(CENTER)[3]
    ok_with = alignment.align_to_target(rig, T, [fm], cfg).ok
    assert ok_with
    # without the back-off approach the loop may need many more photos; it must never be the better choice
    steps_with = len(alignment.align_to_target(rig, T, [fm], cfg).steps)
    steps_without = len(alignment.align_to_target(rig, T, [fm], cfg, approach_um=0.0, max_iter=8).steps)
    assert steps_with <= max(steps_without, 4)


def test_align_fails_loudly_off_the_map() -> None:
    import lib.alignment as alignment

    chip = VirtualChip(_area(), seed=11)
    rig = VirtualRig(chip, center_um=CENTER, seed=5)
    cfg = MapConfig()
    mr = map_area(rig, cfg, _area(), CENTER)
    fm = alignment.FeatureMeasurer(mr)
    far = alignment.align_to_target(rig, (CENTER[0] + 2000.0, CENTER[1]), [fm], cfg)  # beyond the mapped area (+-1457 um)
    assert not far.ok and far.message, "a target outside the mapped area must be reported, not guessed"


def _cross_owner(r: int, c: int, name: str) -> int:
    """Slot of the tile (2x2 pattern, trimmed footprints) that actually exposes corner ``name`` of tile (r, c).
    The same physical cross is the TL of one frame, the TR of its left neighbour, and so on."""
    dr = {"tl": 0, "tr": 0, "bl": 1, "br": 1}[name]
    dc = {"tl": 0, "tr": 1, "bl": 0, "br": 1}[name]
    # the physical cross sits at grid vertex (r + dr, c + dc); candidates are the 4 frames around it
    for rr, cc, nm in ((r + dr, c + dc, "tl"), (r + dr, c + dc - 1, "tr"), (r + dr - 1, c + dc, "bl"), (r + dr - 1, c + dc - 1, "br")):
        if not (0 <= rr < 2 and 0 <= cc < 2):
            continue
        left_ok = nm in ("tr", "br") or cc == 0  # left crosses are dropped by tiles that have a left neighbour
        top_ok = nm in ("bl", "br") or rr == 0  # top crosses are dropped by tiles that have a tile above
        if left_ok and top_ok:
            return rr * 2 + cc
    raise AssertionError("every cross must have exactly one owner")


def test_cross_measurer_and_method_comparison() -> None:
    import lib.alignment as alignment

    imgs, geo, pm, cfg, rig, mr, sim, _ = _session(False, 3)  # layer 1 exposed open-loop (errors of a few um)
    targets = pm.stage_targets(CENTER)
    tmpl = cross_patch(imgs[1], geo, "tl")
    fm = alignment.FeatureMeasurer(mr)
    feature_lost = 0
    for k, T in enumerate(targets):
        cm = alignment.CrossMeasurer(tmpl, geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg)
        rig.move_to(T[0] - 10, T[1] - 10)
        rig.move_to(*T)
        rig.settle(2.5)
        frame = rig.grab()
        m = cm.measure(frame, T)
        assert m.ok and m.n_used >= 3, m.detail
        # truth: each cross was exposed by the tile whose footprint owns it, with that tile's placement error;
        # the camera itself sits cam_err away from where the ideal target would put it
        ideal = [rig.ideal_chip_xy(t) for t in targets]
        r, c = divmod(k, 2)
        errs = []
        for name in ("tl", "tr", "bl", "br"):
            o = _cross_owner(r, c, name)
            errs.append(np.subtract(sim["true_positions_um"][o], ideal[o]))
        cam_err = np.subtract(rig.true_chip_xy(), ideal[k])
        e = np.mean(errs, axis=0) - cam_err
        assert math.hypot(m.correction_um[0] - e[0], m.correction_um[1] + e[1]) < 1.0, (k, m.correction_um, tuple(e))
        feature_lost += 0 if fm.measure(frame, T).ok else 1
    assert feature_lost >= 1, "after patterning the pre-pattern map should stop matching somewhere (remap needed)"
    # the loop can switch: features preferred, crosses take over when features fail
    T = targets[2]
    cm = alignment.CrossMeasurer(tmpl, geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg)
    ar = alignment.align_to_target(rig, T, [fm, cm], cfg)
    assert ar.ok and any(st.used == "crosses" for st in ar.steps), ar.report()
    assert "crosses" in "\n".join(ar.report())


def test_cross_measurer_ml_seed_and_garbage() -> None:
    import lib.alignment as alignment

    imgs, geo, pm, cfg, rig, mr, sim, _ = _session(False, 3)
    T = pm.stage_targets(CENTER)[0]
    tmpl = cross_patch(imgs[1], geo, "tl")
    rig.move_to(T[0] - 10, T[1] - 10)
    rig.move_to(*T)
    rig.settle(2.5)
    frame = rig.grab()
    plain = alignment.CrossMeasurer(tmpl, geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg).measure(frame, T)
    # an ML detector that returns the crosses to within a few px: result must agree with plain template matching
    cw, ch = cfg.cam_size
    s, sg = cfg.um_per_px, cfg.sign
    truth_px = [(cw / 2 + sg[0] * (X - T[0]) / s + plain.correction_um[0] * sg[0] / s * 1.0,
                 ch / 2 + sg[1] * (Y - T[1]) / s + plain.correction_um[1] * sg[1] / s * 1.0)
                for X, Y in alignment.cross_positions_um(geo, T)]
    rng = np.random.default_rng(0)
    seeded = alignment.CrossMeasurer(tmpl, geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg,
                              ml_detector=lambda f: [(x + rng.normal(0, 3), y + rng.normal(0, 3)) for x, y in truth_px])
    ms = seeded.measure(frame, T)
    assert ms.ok and "ML" in ms.detail and math.hypot(ms.correction_um[0] - plain.correction_um[0], ms.correction_um[1] - plain.correction_um[1]) < 1.0
    # garbage detections far from every expected cross are ignored (falls back to template search)
    junk = alignment.CrossMeasurer(tmpl, geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg, ml_detector=lambda f: [(5.0, 5.0), (1800.0, 1000.0)])
    mj = junk.measure(frame, T)
    assert mj.ok and math.hypot(mj.correction_um[0] - plain.correction_um[0], mj.correction_um[1] - plain.correction_um[1]) < 1.0


def test_run_folder_and_finished_piece() -> None:
    import tempfile

    from stage_control.virtual_stage import run_virtual_session

    with tempfile.TemporaryDirectory() as tmp:
        res = run_virtual_session(base_dir=tmp, label="selftest", closed_loop=True, seed=4)
        run = res["recorder"].dir
        assert os.path.basename(run).startswith("run_") and os.path.basename(run).endswith("selftest")
        ls = lambda sub: sorted(os.listdir(os.path.join(run, sub)))
        assert len(ls("1_map/photos")) == 9 and "combined_map.png" in ls("1_map")
        assert {"placement_on_white_3x3.png", "placement_on_camera_map.png", "placement.txt", "patterns_used.txt"} <= set(ls("2_placement"))
        assert any(f.startswith("tile1") for f in ls("3_align")) and len([f for f in ls("3_align") if f.endswith(".txt")]) == 4
        assert len(ls("4_finished_piece/photos")) == 9
        assert {"stitched.png", "stitched_expected_outline.png", "map_report.txt"} <= set(ls("4_finished_piece"))
        assert os.path.exists(os.path.join(run, "run_log.txt"))
        # a second run never overwrites the first
        again = run_virtual_session(base_dir=tmp, label="selftest", closed_loop=False, seed=4)
        assert again["recorder"].dir != run
        # the finished piece: the exposed 2x2 must really be where the placement says (darker inside, not outside)
        fin = res["finished"]
        assert fin.ok, fin.message
        from lib.run_folder import expected_outline_px

        boxes = [b for _, b in expected_outline_px(fin, res["placement"], CENTER)]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        mos = fin.mosaic.astype(np.float32)
        inside = mos[int(y0) + 30 : int(y1) - 30, int(x0) + 30 : int(x1) - 30]
        outside = np.concatenate([mos[: int(y0) - 30, :].ravel(), mos[int(y1) + 30 :, :].ravel()])
        assert inside.mean() < outside.mean() - 5, (inside.mean(), outside.mean())
        # honest accounting: finished-piece scan fits the 5 minute mapping budget too
        assert budget_report(fin.timing["total_s"])["map_ok"]


# ---------------------------------------------------------------- v3: area from tiles, 3/4 step, autofocus, two areas
def test_area_for_tiles_plus_one_frame() -> None:
    for r, c in ((1, 1), (2, 2), (2, 3), (3, 2)):
        pm = area_for_tiles(r, c)
        assert (pm.area_rows, pm.area_cols) == (r + 1, c + 1), "one more complete frame than tiles in each direction"
        assert (pm.rows, pm.cols) == (r, c)
        geo = pm.geo
        w = (geo.footprint(0, 0)[2] - geo.footprint(0, 0)[0]) + c * geo.pitch_x  # first column + c more pitches
        h = (geo.footprint(0, 0)[3] - geo.footprint(0, 0)[1]) + r * geo.pitch_y
        assert pm.area_size_px == (w, h), (r, c, pm.area_size_px, (w, h))
        # the pattern block is centred: half a pitch of margin on every side
        x0 = min(t.dest[0] for t in pm.tiles)
        x1 = max(t.dest[2] for t in pm.tiles)
        assert x0 - pm.area_box[0] == pm.area_box[2] - x1 == geo.pitch_x // 2
    assert area_for_tiles(2, 2).area_size_px == (10790, 5982)


def test_map_grid_is_automatic_and_steps_at_most_three_quarters() -> None:
    cfg = MapConfig()
    fov = (1920 * cfg.um_per_px, 1080 * cfg.um_per_px)
    for r, c in ((1, 1), (2, 2), (3, 3), (2, 4)):
        area = area_for_tiles(r, c).area_size_um
        sc = scan_summary(area, cfg)
        assert max(sc["step_fraction"]) <= 0.75 + 1e-9, (r, c, sc)
        covered = (fov[0] + (sc["cols"] - 1) * sc["step_fraction"][0] * fov[0], fov[1] + (sc["rows"] - 1) * sc["step_fraction"][1] * fov[1])
        assert covered[0] >= area[0] - 1e-6 and covered[1] >= area[1] - 1e-6, "the photos must cover the whole area"
        if sc["cols"] > 1:  # and it never takes more photos than needed
            assert (sc["cols"] - 2) * 0.75 * fov[0] < area[0] - fov[0] + 1e-6
    assert (scan_summary(area_for_tiles(2, 2).area_size_um)["rows"], scan_summary(area_for_tiles(2, 2).area_size_um)["cols"]) == (3, 3)
    # an area that needs exactly a 3/4 step (25% overlap) must still map: the registration window shrinks to fit
    area = (fov[0] * 2.5, fov[1] * 2.5)
    chip = VirtualChip(area, seed=21)
    rig = VirtualRig(chip, center_um=CENTER, seed=2)
    res = map_area(rig, MapConfig(), area, CENTER)
    assert res.ok, res.message
    assert (res.cfg.rows, res.cfg.cols) == (3, 3) and abs(res.stage_um[(0, 1)][0] - res.stage_um[(0, 0)][0] - 0.75 * fov[0]) < 1.0
    assert _max_err_px(rig, res.cfg, res) < 1.5


def test_autofocus_after_moves() -> None:
    import lib.alignment as alignment

    chip = VirtualChip(_area(), seed=11)
    rig = VirtualRig(chip, center_um=CENTER, seed=5)
    mr = map_area(rig, MapConfig(), _area(), CENTER)
    assert mr.ok and rig.af_calls == 9 and mr.timing["autofocus_runs"] == 9, "autofocus after every map move"
    assert mr.timing["total_s"] < 300, "map with autofocus must stay inside the 5 minute target"
    rig2 = VirtualRig(VirtualChip(_area(), seed=11), center_um=CENTER, seed=5)
    assert map_area(rig2, MapConfig(autofocus=False), _area(), CENTER).timing["autofocus_runs"] == 0 and rig2.af_calls == 0
    # alignment: one refocus after the move to the tile; tiny nudges (< af_min_move_um) do not refocus again
    fm = alignment.FeatureMeasurer(mr)
    T = area_for_tiles(2, 2).stage_targets(CENTER)[1]
    before = rig.af_calls
    ar = alignment.align_to_target(rig, T, [fm], MapConfig())
    assert ar.ok and rig.af_calls - before == 1, (rig.af_calls - before, len(ar.steps))
    # focus really matters in the simulation: unfocused photos correlate worse
    rig.z_err = 12.0
    blurred = rig.grab()
    rig.autofocus()
    sharp = rig.grab()
    assert alignment.FeatureMeasurer(mr).measure(sharp, T).confidence >= alignment.FeatureMeasurer(mr).measure(blurred, T).confidence - 0.02


def test_two_areas_cross_on_off_and_both_finished_pieces_mapped() -> None:
    import tempfile

    from stage_control.virtual_stage import run_sample_comparison

    with tempfile.TemporaryDirectory() as tmp:
        r = run_sample_comparison(base_dir=tmp, label="selftest", seed=5)
        w, h = r["area_um"]
        assert not r["areas_overlap"] and r["separation_um"] > w, "the two runs must use different, non-overlapping areas"
        cA, cB = r["runs"]["A"]["center_um"], r["runs"]["B"]["center_um"]
        assert abs(cB[0] - cA[0]) >= w + 100 and cA[1] == cB[1]
        for key in ("A", "B"):
            run = r["runs"][key]
            assert run["map"].ok and run["finished"].ok, f"{key}: map and finished-piece map must both succeed"
            assert run["map"].timing["total_s"] < 300
        A, B = r["runs"]["A"]["patterning"], r["runs"]["B"]["patterning"]
        assert set(A["driven_by"]) == {"features"}, A["driven_by"]  # cross detection off: convolution only
        assert B["driven_by"][0] == "features" and set(B["driven_by"][1:]) == {"crosses"}, B["driven_by"]  # first tile has no crosses yet
        # both ways must stitch well; (over ten seeds neither method was measurably better on seams, ~1.1-1.3 um mean)
        for name, P in (("A", A), ("B", B)):
            assert max(P["seam_error_um"]) < 5.0 and np.mean(P["seam_error_um"]) < 2.5, (name, P["seam_error_um"])
        top = r["recorder"].dir
        for sub in ("A_convolution_only", "B_cross_detection"):
            for d in ("1_map", "2_placement", "3_align", "4_finished_piece"):
                assert os.path.isdir(os.path.join(top, sub, d)), (sub, d)
            assert len(os.listdir(os.path.join(top, sub, "4_finished_piece", "photos"))) == 9
            assert os.path.exists(os.path.join(top, sub, "4_finished_piece", "stitched.png"))
        assert os.path.exists(os.path.join(top, "comparison.txt")) and os.path.exists(os.path.join(top, "both_finished_pieces.png"))
        # the second area really is untouched by the first run's pattern before its own run: A's pattern is inside A only
        rig = r["rig"]
        world = rig.chip.world.astype(np.float32)
        ox, oy = rig.chip.origin_um
        xa = int((cA[0] - CENTER[0] + rig.chip_offset[0] - ox) / rig.chip.res)
        xb = int((cB[0] - CENTER[0] + rig.chip_offset[0] - ox) / rig.chip.res)
        mid = int((rig.chip_offset[1] - oy) / rig.chip.res)
        dark = lambda x: float((world[mid - 300 : mid + 300, x - 700 : x + 700] < 200).mean())
        gap_x = (xa + xb) // 2
        assert dark(gap_x) < dark(xa) and dark(gap_x) < dark(xb), "nothing may be printed in the gap between the areas"


def test_earlier_crosses_logic() -> None:
    import lib.alignment as alignment

    pm = area_for_tiles(2, 2)
    assert alignment.earlier_crosses(pm, 0) == []
    assert alignment.earlier_crosses(pm, 1) == ["tl", "bl"]
    assert alignment.earlier_crosses(pm, 2) == ["tl", "tr"]
    assert alignment.earlier_crosses(pm, 3) == ["tl", "tr", "bl"]
    whole = area_for_tiles(2, 2, mode="whole")  # whole frames: every tile prints all four crosses
    assert alignment.earlier_crosses(whole, 3) == ["tl", "tr", "bl"]
    pm3 = area_for_tiles(2, 3)
    assert alignment.earlier_crosses(pm3, 4) == ["tl", "tr", "bl"] and alignment.earlier_crosses(pm3, 1) == ["tl", "bl"]


def test_yolo_model_as_cross_seed() -> None:
    try:
        from ultralytics import YOLO
    except Exception:  # noqa: BLE001
        print("    (skipped: ultralytics not installed)")
        return
    import lib.alignment as alignment

    model_path = os.path.join(os.path.dirname(HERE), "ckpts", "best.pt")
    if not os.path.exists(model_path):
        print("    (skipped: ckpts/best.pt missing)")
        return
    det = alignment.yolo_detector(YOLO(model_path))
    imgs, geo, pm, cfg, rig, mr, sim, _ = _session(False, 3)
    T = pm.stage_targets(CENTER)[0]
    rig.move_to(T[0] - 10, T[1] - 10)
    rig.move_to(*T)
    rig.autofocus()
    frame = rig.grab()
    pts = det(frame)
    cw, ch = cfg.cam_size
    exp = [(cw / 2 + (X - T[0]) / cfg.um_per_px, ch / 2 - (Y - T[1]) / cfg.um_per_px) for X, Y in alignment.cross_positions_um(geo, T)]
    hits = sum(any(math.hypot(px - ex, py - ey) < 30 for px, py in pts) for ex, ey in exp)
    assert hits >= 2, (hits, pts)  # the model was trained on real photos; it finds most virtual crosses, not all
    cm = alignment.CrossMeasurer(cross_patch(imgs[1], geo, "tl"), geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg, ml_detector=det)
    m = cm.measure(frame, T)
    plain = alignment.CrossMeasurer(cross_patch(imgs[1], geo, "tl"), geo.center_in_cross, alignment.cross_positions_um(geo, T), cfg).measure(frame, T)
    assert m.ok and "ML" in m.detail and math.hypot(m.correction_um[0] - plain.correction_um[0], m.correction_um[1] - plain.correction_um[1]) < 1.0


def test_ui_area_dialog_and_map_before_tiles() -> None:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:  # noqa: BLE001
        print("    (skipped: no display / tkinter)")
        return
    import gui as ui

    root.withdraw()
    try:
        _ui_area_checks(root, ui)
    finally:
        root.destroy()


def test_machine_flow_with_fake_stepper() -> None:
    """Map + Placement -> pattern tiles -> finished piece through ModelRig, the class that drives the real
    stepper, with a fake EventDispatcher over the virtual stage.  The fake camera has the real camera's dark
    edge (outside the part the projector lights); each UV exposure paints the projected pattern onto the chip."""
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:  # noqa: BLE001
        print("    (skipped: no display / tkinter)")
        return
    import tempfile

    import gui as ui
    from lib.img import exposure_mask

    root.withdraw()
    area = area_for_tiles(2, 2).area_size_um
    rig = VirtualRig(VirtualChip(area, seed=11), center_um=CENTER, seed=5)
    log = {"sources_at_grab": [], "patterns": [], "exposed_at": [], "pattern_paths": [], "uv_z": []}

    class Projector:
        def size(self):
            return (1920, 1080)

    class Root:
        def update(self):
            fake.seq += 1  # the camera delivers a frame every time the Tk loop runs
            frame = rig.grab().astype(np.float32)
            h, w = frame.shape[:2]
            ch, cw = round(h / 1.3), round(w / 1.3)
            dark = np.full((h, w), 0.25, np.float32)
            dark[(h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw] = 1.0
            fake.camera_image = (frame * (dark[..., None] if frame.ndim == 3 else dark)).astype(np.uint8)
            log["sources_at_grab"].append(fake.red_focus_source)

        def update_idletasks(self):
            pass

    class FakeStepper:
        camera = True
        model = None  # no YOLO model loaded
        hardware = type("H", (), {"projector": Projector()})()

        def __init__(self):
            self.seq, self.camera_image = 0, None
            self.stage_setpoint = [CENTER[0], CENTER[1], 0.0]
            self.red_focus_source = ui.RedFocusSource.PATTERN
            self.red_focus_image, self.shown_image, self.pattern = None, ui.ShownImage.RED_FOCUS, None
            self.root = Root()

        camera_image_seq = property(lambda s: s.seq)

        def move_absolute(self, d):
            rig.move_to(d["x"], d["y"])
            self.stage_setpoint = [d["x"], d["y"], d.get("z", 0.0)]
            return True

        def move_relative(self, d):
            self.stage_setpoint[2] += d.get("z", 0.0)

        def non_blocking_delay(self, s):
            rig.settle(s)

        def set_red_focus_image(self, img):
            self.red_focus_image = img

        def set_red_focus_source(self, src):
            self.red_focus_source = src

        def set_shown_image(self, s):
            self.shown_image = s

        def enter_red_mode(self, mode_switch_autofocus=True):
            self.shown_image = ui.ShownImage.RED_FOCUS

        def set_autofocus_busy(self, busy):
            pass

        def set_pattern_image(self, img, path):
            # the real UI opens this path for the exposure thumbnail (Chip panel), so it must be a file
            from PIL import Image as PILImage

            PILImage.open(path).close()
            log["pattern_paths"].append(path)
            self.pattern = img

        def expose_pattern(self, red_to_uv_offset, uv_autofocus=True):
            # the map run exposes at red focus + the UV offset, without the old absolute-z UV autofocus
            assert uv_autofocus is False
            z_red = self.stage_setpoint[2]
            self.move_relative({"z": red_to_uv_offset})
            log["uv_z"].append((z_red, self.stage_setpoint[2]))
            pat = np.asarray(self.pattern)
            log["patterns"].append(pat)
            log["exposed_at"].append(rig.expose(exposure_mask(pat), PROJ_UM_PER_PX))
            self.move_relative({"z": -red_to_uv_offset})

    fake = FakeStepper()
    real = (ui.AreaDialog, ui.PatternPickerDialog, ui.messagebox.askyesno, ui.messagebox.showinfo,
            ui.messagebox.showwarning, ui.RunRecorder, ui.ModelRig.autofocus)
    tmp = tempfile.mkdtemp()
    real_proj = ui.ModelRig.projector_calibration

    class FakeArea:
        def __init__(self, *a, **k):
            pass

        def show(self):
            return (2, 2)

    class FakePicker:
        def __init__(self, *a, **k):
            pass

        def show(self):
            return [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]

    class AutoApprove(ui.PlacementApprovalWindow):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.win.after(10, self.approve)

    try:
        ui.AreaDialog, ui.PatternPickerDialog = FakeArea, FakePicker
        ui.messagebox.askyesno = lambda *a, **k: True
        ui.messagebox.showinfo = ui.messagebox.showwarning = lambda *a, **k: None
        ui.RunRecorder = lambda label="": real[5](tmp, label)
        ui.ModelRig.autofocus = lambda self: rig.autofocus()  # the grid search has nothing to focus on here
        # nor a grid to photograph: the projector -> camera mapping the real autofocus would measure
        ui.ModelRig.projector_calibration = lambda self: rig.projector_calibration((1477, 831))
        ui.PlacementApprovalWindow, real_win = AutoApprove, ui.PlacementApprovalWindow
        try:
            approved = ui.run_map_and_placement(root, fake)
        finally:
            ui.PlacementApprovalWindow = real_win
    finally:
        (ui.AreaDialog, ui.PatternPickerDialog, ui.messagebox.askyesno, ui.messagebox.showinfo,
         ui.messagebox.showwarning, ui.RunRecorder, ui.ModelRig.autofocus) = real
        ui.ModelRig.projector_calibration = real_proj
        root.destroy()
    assert approved is not None
    mr = approved.map_result
    # every photo is the lit middle of the frame: no dark edge in the map
    assert mr.cfg.cam_size == (1477, 831), mr.cfg.cam_size
    assert (mr.cfg.rows, mr.cfg.cols) == (3, 3), (mr.cfg.rows, mr.cfg.cols)  # 2 x 2 tiles: a 3 x 3 map
    assert np.percentile(mr.mosaic, 0.5) > 0.6 * np.median(mr.mosaic), "dark edge in the map"
    # photos were taken under plain red, and the projector setting was put back afterwards
    assert ui.RedFocusSource.SOLID in log["sources_at_grab"] and fake.red_focus_source == ui.RedFocusSource.PATTERN
    # four exposures, each the tile's pattern with everything outside its footprint black
    pm = approved.placement
    assert len(log["patterns"]) == 4
    assert all(abs((zu - zr) - (-40.0)) < 1e-9 for zr, zu in log["uv_z"]), log["uv_z"]  # default offset
    for t, pat in zip(pm.tiles, log["patterns"]):
        x0, y0, x1, y1 = t.crop
        outside = exposure_mask(pat).copy()
        outside[y0:y1, x0:x1] = False
        assert not outside.any(), f"tile {t.slot + 1} exposes outside its footprint"
    # the tiles landed where they should relative to each other (closed loop through ModelRig)
    ideal = [rig.ideal_chip_xy(T) for T in approved.stage_targets]
    got = log["exposed_at"]
    err = [
        math.hypot((got[k][0] - got[0][0]) - (ideal[k][0] - ideal[0][0]), (got[k][1] - got[0][1]) - (ideal[k][1] - ideal[0][1]))
        for k in range(1, 4)
    ]
    assert max(err) < 3.0, err
    # and the finished piece was re-mapped into the same run folder, next to each tile's exposed pattern
    run_dir = next(os.path.join(tmp, d) for d in os.listdir(tmp))
    assert [os.path.dirname(p) for p in log["pattern_paths"]] == [os.path.join(run_dir, "3_align")] * 4, log["pattern_paths"]
    assert os.path.exists(os.path.join(run_dir, "4_finished_piece", "stitched_expected_outline.png")), os.listdir(run_dir)


def _ui_area_checks(root, ui) -> None:
    d = ui.AreaDialog(root)
    text = "\n".join(d.info_lines())
    assert "3 x 3 complete frames" in text and "2914 x 1615 um" in text and "(2.91 x 1.61 mm)" in text, text
    assert "3 x 3 (9)" in text and "% x" in text and "max 75%" in text
    d.cols_var.set(3)
    d.rows_var.set(1)
    text = "\n".join(d.info_lines())
    assert "4 x 2 complete frames" in text, text
    d.confirm()
    assert d.result == (1, 3)
    d2 = ui.AreaDialog(root)
    d2.cols_var.set(0)
    assert d2._counts() is None and d2.confirm_btn.instate(["disabled"])
    d2.cancel()
    assert d2.result is None

    # run_map_and_placement: the picker must not open before the map is done, and not at all when the map fails
    chip = VirtualChip(area_for_tiles(2, 2).area_size_um, seed=11)

    def make_model(features=True):
        rig = VirtualRig(VirtualChip(area_for_tiles(2, 2).area_size_um, seed=11, features=features), center_um=CENTER, seed=5)

        class Model:
            stage_setpoint = [CENTER[0], CENTER[1], 0.0]
            af_args: list = []

            def move_absolute(self, d):
                rig.move_to(d["x"], d["y"])
                return True

            def non_blocking_delay(self, t):
                rig.settle(t)

            def autofocus(self, blue_only, log=False):
                self.af_args.append(blue_only)
                rig.autofocus()

            @property
            def camera_image(self):
                return rig.grab()

        return Model(), rig

    events = []
    real = (ui.AreaDialog, ui.PatternPickerDialog, ui.PlacementApprovalWindow, ui.messagebox.showwarning, ui.messagebox.showinfo, ui.RunRecorder)
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        for features in (True, False):
            events.clear()
            model, rig = make_model(features)

            class FakeArea:
                def __init__(self, *a, **k):
                    events.append("area dialog")

                def show(self):
                    return (2, 2)

            class FakePicker:
                def __init__(self, parent, rows, cols, samples):
                    events.append(f"picker opened after {len(rig.log)} photos")
                    self.rows, self.cols = rows, cols

                def show(self):
                    return [os.path.join(ui.SAMPLE_DIR, f) for f in ui.SAMPLE_SET]

            class FakeApproval:
                def __init__(self, parent, images, paths, **k):
                    events.append(f"approval {k['rows']}x{k['cols']} in {k['area_rows']}x{k['area_cols']}, mapped area {k['mapped_area_um'][0]:.0f} um")
                    self.result = None
                    self.win = type("W", (), {"grab_set": lambda s: None, "wait_window": lambda s: None})()

            ui.AreaDialog, ui.PatternPickerDialog, ui.PlacementApprovalWindow = FakeArea, FakePicker, FakeApproval
            ui.messagebox.showwarning = lambda t, m: events.append("warning")
            ui.RunRecorder = lambda label="": real[5](tmp, label)
            if os.path.exists(_run_folder.CALIBRATION_FILE):  # each run starts uncalibrated
                os.remove(_run_folder.CALIBRATION_FILE)
            ui.run_map_and_placement(root, model)
            # the photo run_map_and_placement grabs first to learn the photo size, then a 3 x 3 map (photos are
            # cropped to the lit middle of the frame; the map covers every tile's alignment view), plus any
            # re-takes.  A blank chip maps too: its overlaps are placed from the stage, so tiles can be inserted.
            assert events[0] == "area dialog" and events[1].startswith("picker opened after "), events
            assert 1 + 9 <= int(events[1].split()[3]) <= 1 + 9 + 2, events
            mapped = scan_area_um(area_for_tiles(2, 2), MapConfig(cam_size=(1477, 831)))[0]  # the cropped photo
            assert events[2] == f"approval 2x2 in 3x3, mapped area {mapped:.0f} um", events
            # ModelRig's autofocus projects a grid and needs a camera; this fake model has none, so it is skipped
            # and the old UI's red autofocus is never called
            assert not model.af_args, model.af_args
    finally:
        ui.AreaDialog, ui.PatternPickerDialog, ui.PlacementApprovalWindow, ui.messagebox.showwarning, ui.messagebox.showinfo, ui.RunRecorder = real


# ---------------------------------------------------------------- visuals
def make_visuals(outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(outdir, exist_ok=True)
    imgs, geo = _load_samples()
    pm = build_placement_map(geo)
    W, H = geo.frame_w, geo.frame_h
    px, py = geo.pitch_x, geo.pitch_y
    ux, uy = pm.um_per_px

    # a virtual run for the timing numbers
    chip = VirtualChip(pm.area_size_um, seed=11)
    _, _, res = _run(chip)
    budget = budget_report(res.timing["total_s"])

    prev, _ = render_placement(pm, imgs, scale=0.16)
    cv2.imwrite(os.path.join(outdir, "map_placement_preview.png"), prev)

    fig = plt.figure(figsize=(20, 12.6), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.08], height_ratios=[1, 1], left=0.02, right=0.99, top=0.93, bottom=0.03, wspace=0.05, hspace=0.12)
    fig.suptitle("Camera map: cross geometry, footprints and placement (numbers measured from your three patterns)", fontsize=17, fontweight="bold")
    RED, GREY = "#D80027", "#555555"

    def arrow(ax, p0, p1, text, off=(0, 0), color=GREY, fs=11, ha="center", va="center"):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1.6, shrinkA=0, shrinkB=0))
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, color=color, fontsize=fs, ha=ha, va=va,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    # ---- A: one frame, measured
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("1. One 3840x2160 frame: the four crosses and their constant distances", fontsize=13, loc="left")
    show = np.zeros((540, 960, 3), np.uint8)
    show[cv2.resize(exposure_mask(imgs[3]).astype(np.uint8), (960, 540), interpolation=cv2.INTER_AREA) > 0] = (230, 90, 25)
    ax.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB), extent=(0, W, H, 0))
    ax.add_patch(Rectangle((px, 0), W - px, H, fc="#FFD54F", ec="none", alpha=0.28))
    ax.add_patch(Rectangle((0, py), W, H - py, fc="#FFD54F", ec="none", alpha=0.28))
    for n in ("tl", "tr", "bl", "br"):
        x, y = geo.corner(n)
        ax.add_patch(Rectangle((x, y), geo.cross_w, geo.cross_h, fill=False, ec=RED, lw=2))
    arrow(ax, (geo.tl[0], -150), (geo.tr[0], -150), f"cross-to-cross pitch  {px} px  (left edge to left edge)", color="#0B5394", fs=12)
    arrow(ax, (-190, geo.tl[1]), (-190, geo.bl[1]), f"{py} px", color="#0B5394", fs=12)
    ax.texts[-1].set_rotation(90)
    arrow(ax, (geo.tl[0] + geo.cross_w, 560), (geo.tr[0], 560), f"gap between crosses  {geo.tr[0] - geo.tl[0] - geo.cross_w} px", color=GREY, fs=11)
    arrow(ax, (geo.tl[0], -50), (geo.tl[0] + geo.cross_w, -50), f"{geo.cross_w} px", color=RED, fs=10, off=(0, -45))
    arrow(ax, (-60, geo.tl[1]), (-60, geo.tl[1] + geo.cross_h), f"{geo.cross_h} px", color=RED, fs=10, off=(-95, 0))
    ax.text(geo.tl[0] + geo.cross_w + 60, geo.tl[1] + geo.cross_h / 2, f"cross box, {geo.tl[0]} px in from the frame edge", color="#ffb3b3", fontsize=11, va="center")
    ax.text(px + (W - px) / 2, H * 0.50, f"overlaps\nright neighbour\n{geo.overlap_x} px", color="#8a6d00", fontsize=11, ha="center", va="center")
    ax.text(W / 2, py + (H - py) / 2 + 20, f"overlaps bottom neighbour: {geo.overlap_y} px", color="#8a6d00", fontsize=11, ha="center", va="center")
    ax.set_xlim(-330, W + 80)
    ax.set_ylim(H + 60, -300)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- B: who exposes what
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("2. Footprints: each tile exposes from the neighbour's cross onward (crosses overlap, counted once)", fontsize=13, loc="left")
    cols = ["#FB8C00", "#43A047", "#AB47BC", "#00ACC1"]
    for t in pm.tiles:
        ox, oy = (t.col * px, t.row * py)  # frames shown at pitch spacing, relative
        x0, y0, x1, y1 = t.crop
        ax.add_patch(Rectangle((ox, oy), W, H, fill=False, ec="#999", lw=1, ls="--"))
        ax.add_patch(Rectangle((ox + x0, oy + y0), x1 - x0, y1 - y0, fc=cols[t.slot], ec=cols[t.slot], alpha=0.30, lw=2))
        if t.col > 0:
            ax.add_patch(Rectangle((ox + geo.tl[0], oy + geo.tl[1]), geo.cross_w, y1 - geo.tl[1], fc="none", ec="#777", hatch="///", lw=0))
        if t.row > 0:
            ax.add_patch(Rectangle((ox + geo.tl[0], oy + geo.tl[1]), x1 - geo.tl[0], geo.cross_h, fc="none", ec="#777", hatch="\\\\\\", lw=0))
        ax.text(ox + (x0 + x1) / 2, oy + (y0 + y1) / 2 - 40, f"tile {t.slot + 1}", fontsize=17, fontweight="bold", color=cols[t.slot], ha="center")
        ax.text(ox + (x0 + x1) / 2, oy + (y0 + y1) / 2 + 160, f"exposes x {x0}-{x1}\ny {y0}-{y1}\n= {x1 - x0} x {y1 - y0} px", fontsize=11, ha="center", va="top")
        for n in ("tl", "tr", "bl", "br"):
            cx, cy = geo.corner(n)
            ax.add_patch(Rectangle((ox + cx, oy + cy), geo.cross_w, geo.cross_h, fill=False, ec=RED, lw=1.6))
    ax.text(0.5, -0.02, "red = crosses (neighbouring frames' crosses sit exactly on each other)\nhatched = cross zone the tile does NOT expose because the neighbour already did",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color=GREY)
    ax.set_xlim(-150, px + W + 150)
    ax.set_ylim(py + H + 330, -150)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- C: placement in the white 3x3
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("3. Placement in the white 3x3 area (this is what the Approve window shows)", fontsize=13, loc="left")
    sc = 0.16
    img, (bx, by) = render_placement(pm, imgs, scale=sc)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    to = lambda ax_px, ay_px: ((ax_px - bx) * sc, (ay_px - by) * sc)
    t0 = pm.tiles[0]
    arrow(ax, to(0, 1500), to(t0.dest[0], 1500), f"{t0.dest[0]} px\n= {t0.dest[0] * ux:.0f} um\n(half a pitch)", color="#0B5394", fs=10)
    arrow(ax, to(1000, 0), to(1000, t0.dest[1]), f"{t0.dest[1]} px = {t0.dest[1] * uy:.0f} um", color="#0B5394", fs=10, off=(120, 0), ha="left")
    ax.text(*to(pm.area_box[2] - 150, pm.area_box[3] + 100), f"area {pm.area_size_px[0]} x {pm.area_size_px[1]} px = {pm.area_size_um[0]:.0f} x {pm.area_size_um[1]:.0f} um   (3x3 cells)",
            fontsize=11, ha="right", va="top", color=GREY)
    ax.axis("off")

    # ---- D: numbers + worked example
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    t3 = pm.tiles[3]
    tg = pm.stage_targets(CENTER)
    ovh = TilingTimeModel().overhead_s
    lines = [
        "SIZES",
        f"  projector frame : {W} x {H} px   (old UI tile; {W * ux:.0f} x {H * uy:.0f} um, {ux:.4f} um/px)",
        f"  camera frame    : 1920 x 1080 px (0.678 um/px -> {1920 * 0.678:.0f} x {1080 * 0.678:.0f} um)",
        "",
        "MEASURED FROM YOUR PATTERNS (all three identical)",
        f"  cross box {geo.cross_w} x {geo.cross_h} px, {geo.tl[0]} px in from the edges",
        f"  pitch   x {px} px   y {py} px          gap x {px - geo.cross_w} y {py - geo.cross_h} px",
        f"  overlap with neighbour  x {geo.overlap_x} px   y {geo.overlap_y} px",
        "",
        "FOOTPRINT RULE (what each tile exposes, frame px)",
        f"  first column  x {geo.footprint(0, 0)[0]}-{geo.footprint(0, 0)[2]}  ({geo.footprint(0, 0)[2] - geo.footprint(0, 0)[0]} px, both crosses)",
        f"  other columns x {geo.footprint(0, 1)[0]}-{geo.footprint(0, 1)[2]}  ({geo.footprint(0, 1)[2] - geo.footprint(0, 1)[0]} px = one pitch)",
        f"  first row     y {geo.footprint(0, 0)[1]}-{geo.footprint(0, 0)[3]}  ({geo.footprint(0, 0)[3] - geo.footprint(0, 0)[1]} px)",
        f"  other rows    y {geo.footprint(1, 0)[1]}-{geo.footprint(1, 0)[3]}  ({geo.footprint(1, 0)[3] - geo.footprint(1, 0)[1]} px = one pitch)",
        "",
        "WORKED EXAMPLE: tile 4 (row 2, column 2)",
        f"  frame origin in area = (1.5 x {px} - {geo.tl[0]}, 1.5 x {py} - {geo.tl[1]}) = {t3.frame_origin} px",
        f"  exposes frame x {t3.crop[0]}-{t3.crop[2]}, y {t3.crop[1]}-{t3.crop[3]}",
        f"  lands on area x {t3.dest[0]}-{t3.dest[2]}, y {t3.dest[1]}-{t3.dest[3]}",
        f"  stage target = centre + ({t3.center_um[0]:+.1f}, {-t3.center_um[1]:+.1f}) um",
        f"     centre ({CENTER[0]:.0f}, {CENTER[1]:.0f}) -> ({tg[3][0]:.1f}, {tg[3][1]:.1f})",
        f"  step between tiles: {pm.step_um[0]:.1f} um in x, {pm.step_um[1]:.1f} um in y",
        "",
        "TIME (virtual run, stage times are estimates)",
        f"  map: 9 frames, {res.timing['stage_s']:.0f} s stage + {res.timing['compute_s']:.1f} s compute  (limit {budget['map_limit_s']:.0f} s)",
        f"  tiling overhead ~{ovh:.0f} s/tile assumed -> up to {budget['max_exposure_per_tile_s']:.0f} s exposure per tile in {budget['tile_limit_s']:.0f} s",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=10.6, va="top", ha="left", linespacing=1.42)
    fig.savefig(os.path.join(outdir, "map_understanding.png"))
    plt.close(fig)
    print(f"    wrote {outdir}/map_understanding.png and map_placement_preview.png")


# ---------------------------------------------------------------- virtual demo (UI, no hardware)
def virtual_demo() -> None:
    """The whole flow against the virtual white chip (no hardware): python tiling_selftest.py --demo"""
    import tkinter as tk
    from tkinter import messagebox, ttk

    from gui import AreaDialog, PatternPickerDialog, PlacementApprovalWindow, ProgressWindow
    from lib.run_folder import RunRecorder, map_finished_piece
    from lib.tiling import SAMPLE_SET
    from stage_control.virtual_stage import (
        VirtualChip,
        VirtualRig,
        run_sample_comparison,
        simulate_patterning,
    )

    root = tk.Tk()
    root.title("Camera map - virtual test")
    closed = tk.BooleanVar(value=True)
    crosses = tk.BooleanVar(value=True)

    def go() -> None:
        counts = AreaDialog(root).show()
        if not counts:
            return
        tile_rows, tile_cols = counts
        area_pm = area_for_tiles(tile_rows, tile_cols)
        center = (5000.0, 3000.0)
        rig = VirtualRig(VirtualChip(area_pm.area_size_um), center_um=center)
        rec = RunRecorder(label="virtual")
        prog = ProgressWindow(root)
        res = map_area(rig, MapConfig(), area_pm.area_size_um, center, prog.update, lambda: prog.aborted)
        prog.close()
        print("\n".join(res.report()))
        rec.save_map(res)
        if not res.ok:
            messagebox.showwarning("Camera map", "\n".join(res.report()))
            return
        sample_paths = [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]  # tiles can be inserted only now
        paths = PatternPickerDialog(root, tile_rows, tile_cols, sample_paths if (tile_rows, tile_cols) == (2, 2) else None).show()
        if not paths:
            return
        images = [load_image(p) for p in paths]
        win = PlacementApprovalWindow(
            root, images, paths, rows=tile_rows, cols=tile_cols, area_rows=tile_rows + 1, area_cols=tile_cols + 1,
            area_center_um=center, map_result=res, mapped_area_um=area_pm.area_size_um,
        )
        win.win.wait_window()
        if not win.result:
            return
        print("\n".join(win.result.lines()))
        pm = win.result.placement
        rec.save_placement(pm, images, paths, center, res)
        if messagebox.askyesno(
            "Virtual patterning",
            "Simulate patterning (closed-loop alignment if ticked, cross detection if ticked, then 'expose' each tile "
            "onto the virtual chip) and map the finished piece?",
        ):
            prog = ProgressWindow(root, "Virtual patterning")
            try:
                sim = simulate_patterning(
                    rig, res, pm, images, MapConfig(), center, closed.get(), rec, lambda m: prog.update(0.5, m), crosses.get()
                )
                fin = map_finished_piece(rig, MapConfig(), pm.area_size_um, center, rec, pm, prog.update)
            finally:
                prog.close()
            messagebox.showinfo(
                "Virtual patterning",
                f"tile spacing error vs ideal: {', '.join(f'{v:.1f}' for v in sim['spacing_error_um'])} um "
                f"({sim['photos']} alignment photos, driven by {sim['driven_by']})\n\n"
                + "\n".join(fin.report()) + f"\n\nRun folder:\n{rec.dir}",
            )

    def sample() -> None:
        """Everything twice, on two different areas of one chip: convolution only, then cross detection on."""
        prog = ProgressWindow(root, "Sample run (2 areas)")
        model = None
        try:
            try:  # the old UI's YOLO marker model, when ultralytics is installed
                from ultralytics import YOLO

                model = YOLO(os.path.join(os.path.dirname(HERE), "ckpts", "best.pt"))
            except Exception:  # noqa: BLE001
                model = None
            res = run_sample_comparison(label="sample", ml_model=model, progress=lambda m: prog.update(0.5, m))
        finally:
            prog.close()
        messagebox.showinfo(
            "Sample run",
            "\n".join(res["summary"]) + f"\n\nYOLO seed: {'yes' if model is not None else 'no (ultralytics/model not available)'}"
            f"\n\nRun folder:\n{res['recorder'].dir}",
        )

    ttk.Label(root, text="The chip is white; scratches and dust give the camera something to register on.").grid(row=0, column=0, padx=16, pady=(14, 6))
    ttk.Checkbutton(root, text="Closed-loop alignment before each virtual exposure", variable=closed).grid(row=1, column=0, sticky="w", padx=16)
    ttk.Checkbutton(root, text="Cross detection (use crosses printed by earlier tiles)", variable=crosses).grid(row=2, column=0, sticky="w", padx=16)
    ttk.Button(root, text="Run virtual test", command=go).grid(row=3, column=0, pady=6)
    ttk.Button(root, text="Sample: run twice (convolution only, then cross detection) on two areas", command=sample).grid(row=4, column=0, pady=6, padx=16)
    ttk.Button(root, text="Quit", command=root.destroy).grid(row=5, column=0, pady=(0, 14))
    root.mainloop()


# ---------------------------------------------------------------- runner
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visuals", metavar="OUTDIR", help="write the explanatory pictures here")
    ap.add_argument("--demo", action="store_true", help="open the virtual demo window instead of testing")
    args = ap.parse_args()
    if args.demo:
        virtual_demo()
        return 0
    tests: List[Callable[[], None]] = [
        test_geometry, test_placement, test_seams, test_map_happy_path,
        test_map_tolerates_a_bad_config_and_rough_stage, test_map_uses_the_stage_when_it_cannot_trust_the_registration,
        test_left_top_check_catches_a_false_lock, test_check_every_tile_option, test_budget,
        test_cross_inputs, test_align_closed_loop_beats_open_loop, test_align_backlash_hunting_is_prevented,
        test_align_fails_loudly_off_the_map, test_cross_measurer_and_method_comparison,
        test_cross_measurer_ml_seed_and_garbage, test_run_folder_and_finished_piece,
        test_area_for_tiles_plus_one_frame, test_map_grid_is_automatic_and_steps_at_most_three_quarters,
        test_autofocus_after_moves, test_map_measures_the_real_camera_scale_and_rotation,
        test_focus_prediction_and_missed_focus, test_projector_calibration_from_the_autofocus_grid,
        test_earlier_crosses_logic, test_two_areas_cross_on_off_and_both_finished_pieces_mapped,
        test_yolo_model_as_cross_seed, test_ui_area_dialog_and_map_before_tiles, test_machine_flow_with_fake_stepper,
        test_ui,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    if args.visuals:
        make_visuals(args.visuals)
    print("ALL TESTS PASSED" if not failed else f"{failed} TEST(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

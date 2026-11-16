"""One folder per camera-map test run: the photos, the combined map, the placement and the finished piece.

    map_runs/run_20261104_142300_label/
        run_log.txt                 what happened, in order, with times
        1_map/photos/               every camera photo used for the map, one file per tile
        1_map/combined_map.png      the stitched map
        1_map/map_report.txt, map_data.json
        2_placement/                placement preview on the white 3x3, stage targets, patterns used
        3_align/                    closed-loop alignment reports (and photos, when kept)
        4_finished_piece/photos/    quick re-map of the patterned piece, separate photos
        4_finished_piece/stitched.png and stitched_expected_outline.png
        timing.txt, timing.json     how long every section took (next to the 1_map folder)

The folders live in ``src/map_runs``.  PNGs are git-ignored; the
folder is in .gitignore as well so test runs never end up in a commit.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .img import save_png, save_png_bgr
from .mapping import (
    MapConfig,
    MapResult,
    ProjectorCalibration,
    Rig,
    background_from_map,
    map_area,
    status,
    tile_point_map_px,
)
from .tiling import TILE_COLORS, PlacementMap, check_seams, render_placement

# test runs are written next to the code (src/map_runs); the folder is git-ignored
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "map_runs")


# what this machine has measured about itself (camera vs stage, red -> UV focus offset); kept between runs
CALIBRATION_FILE = os.path.join(RUNS_DIR, "calibration.json")


def load_calibration() -> dict:
    """The stored machine calibration ({} when nothing has been measured yet)."""
    try:
        with open(CALIBRATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_calibration(**values) -> None:
    """Update some calibration values (the others are kept) with the time they were measured."""
    cal = load_calibration()
    stamp = datetime.now().isoformat(timespec="seconds")
    for k, v in values.items():
        cal[k] = v
        cal[f"{k}_measured"] = stamp
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)


def _write_text(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def tile_name(tile: Tuple[int, int], order_index: Optional[int] = None) -> str:
    base = f"tile_r{tile[0] + 1}c{tile[1] + 1}"
    return base if order_index is None else f"{order_index + 1:02d}_{base}"


class RunRecorder:
    """Creates the run folder and writes the pieces of a run into it."""

    def __init__(self, base_dir: Optional[str] = None, label: str = "", exact_dir: Optional[str] = None):
        self.timings: List[dict] = []  # {"section", "seconds", "note", "kind"}; see section()/add_timing()
        self._t_created = time.perf_counter()
        if exact_dir is not None:  # a sub-folder of a run (see sub())
            os.makedirs(exact_dir, exist_ok=True)
            self.dir = exact_dir
            return
        base = base_dir or RUNS_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
        name = f"run_{stamp}" + (f"_{slug}" if slug else "")
        path, n = os.path.join(base, name), 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(base, f"{name}_{n}")
        os.makedirs(path)
        self.dir = path
        self.log(f"run folder created: {path}")

    def sub(self, name: str) -> "RunRecorder":
        """A recorder that writes into a sub-folder of this run (e.g. one per process in a comparison)."""
        return RunRecorder(exact_dir=os.path.join(self.dir, name))

    # ---- timing: one line per section, saved next to the 1_map folder after every section ----
    @contextmanager
    def section(self, name: str, note: str = "", waiting_for_user: bool = False):
        """Time a block of the run.  ``waiting_for_user`` marks sections that are a person clicking through a
        dialog (so they can be left out of the machine time)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_timing(name, time.perf_counter() - t0, note, "user" if waiting_for_user else "machine")

    def add_timing(self, name: str, seconds: float, note: str = "", kind: str = "detail") -> None:
        """kind: 'machine' (a section of the run), 'user' (waiting for a person) or 'detail' (part of a section)."""
        self.timings.append({"section": name, "seconds": round(float(seconds), 2), "note": note, "kind": kind})
        if kind != "detail":
            status(f"timing: {name} took {seconds:.1f} s" + (f" ({note})" if note else ""))
        self.write_timing()

    def add_map_timing(self, name: str, mr: MapResult) -> None:
        """The measured breakdown of a map / finished-piece scan: photos, autofocus, stitching."""
        t = mr.timing
        if "wall_s" not in t:
            return
        n = max(1, len(mr.photo_wall_s))
        self.add_timing(
            f"{name}: photos", t.get("wall_photos_s", 0.0),
            f"{len(mr.photo_wall_s)} photos, {t.get('wall_photos_s', 0.0) / n:.1f} s each incl. move, settle, focus, grab, register",
            "detail",
        )
        self.add_timing(f"{name}: autofocus", t.get("wall_autofocus_s", 0.0), f"{int(t.get('autofocus_runs', 0))} runs", "detail")
        self.add_timing(f"{name}: stitching", t.get("wall_stitch_s", 0.0), "solve positions + blend mosaic", "detail")

    def write_timing(self) -> None:
        lines = ["section timing (seconds, measured on this run)", ""]
        machine = user = 0.0
        for e in self.timings:
            tag = {"machine": "", "user": "  [waiting for you]", "detail": "      "}.get(e["kind"], "")
            if e["kind"] == "detail":
                lines.append(f"      - {e['section']}: {e['seconds']:.1f} s" + (f"  ({e['note']})" if e["note"] else ""))
            else:
                lines.append(f"{e['section']}: {e['seconds']:.1f} s{tag}" + (f"  ({e['note']})" if e["note"] else ""))
                if e["kind"] == "machine":
                    machine += e["seconds"]
                else:
                    user += e["seconds"]
        lines += ["", f"machine sections total: {machine:.0f} s ({machine / 60:.1f} min)", f"waiting for the user: {user:.0f} s"]
        _write_text(os.path.join(self.dir, "timing.txt"), lines)
        with open(os.path.join(self.dir, "timing.json"), "w", encoding="utf-8") as f:
            json.dump({"sections": self.timings, "machine_total_s": machine, "user_wait_s": user}, f, indent=2)

    def log(self, text: str) -> None:
        with open(os.path.join(self.dir, "run_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}  {text}\n")

    # ---- 1. map ----
    def save_map(self, mr: MapResult, subdir: str = "1_map") -> str:
        """Photos used for the map (in scan order), the combined map, a report and the numbers."""
        from .mapping import snake_order

        root = os.path.join(self.dir, subdir)
        order = snake_order(mr.cfg.rows, mr.cfg.cols)
        for k, tile in enumerate(order):
            if tile in mr.frames:
                save_png(os.path.join(root, "photos", tile_name(tile, k) + ".png"), mr.frames[tile])
        if mr.mosaic is not None:
            save_png(os.path.join(root, "combined_map.png"), mr.mosaic)
        _write_text(os.path.join(root, "map_report.txt"), mr.report())
        data = {
            "ok": mr.ok,
            "message": mr.message,
            "um_per_px": mr.cfg.um_per_px,
            "area_center_um": mr.area_center_um,
            "stage_um": {tile_name(t): list(v) for t, v in mr.stage_um.items()},
            "positions_px": {tile_name(t): list(v) for t, v in mr.positions_px.items()},
            "pairs": [
                {"a": tile_name(p.a), "b": tile_name(p.b), "kind": p.kind, "measured_px": list(p.measured_px),
                 "predicted_px": list(p.predicted_px), "peak": p.peak, "second_peak": p.second, "ok": p.ok, "reason": p.reason,
                 "estimated": p.estimated}
                for p in mr.pairs
            ],
            "checks": [
                {"tile": tile_name(c.tile), "delta_um": c.delta_um, "tol_um": c.tol_um, "passed": c.passed} for c in mr.checks
            ],
            "timing_s": mr.timing,
        }
        with open(os.path.join(root, "map_data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.log(f"map saved ({'ok' if mr.ok else 'FAILED'}): {mr.captured} photos -> {root}")
        self.add_map_timing("finished piece" if subdir.startswith("4_") or "finished" in subdir else "map", mr)
        return root

    # ---- 2. placement ----
    def save_placement(
        self,
        pm: PlacementMap,
        images: Sequence[np.ndarray],
        paths: Sequence[str],
        center_um: Tuple[float, float],
        mr: Optional[MapResult] = None,
        subdir: str = "2_placement",
    ) -> str:
        root = os.path.join(self.dir, subdir)
        scale = 0.12
        bg = background_from_map(mr, pm, scale) if mr is not None and mr.ok else None
        white, _ = render_placement(pm, images, scale=scale)
        save_png_bgr(os.path.join(root, "placement_on_white_3x3.png"), white)
        if bg is not None:
            over, _ = render_placement(pm, images, scale=scale, background=bg)
            save_png_bgr(os.path.join(root, "placement_on_camera_map.png"), over)
        tg = pm.stage_targets(center_um)
        lines = list(pm.report()) + [""]
        for t, (x, y) in zip(pm.tiles, tg):
            lines.append(f"tile {t.slot + 1}: stage target ({x:.1f}, {y:.1f}) um")
        try:
            seams = check_seams(list(images), pm.rows, pm.cols, pm.geo)
            lines += [""] + [f"seam {s}" for s in seams]
        except Exception as e:  # noqa: BLE001
            lines.append(f"seam check unavailable: {e}")
        _write_text(os.path.join(root, "placement.txt"), lines)
        _write_text(os.path.join(root, "patterns_used.txt"), [f"{i + 1}: {p}" for i, p in enumerate(paths)])
        self.log(f"placement saved -> {root}")
        return root

    # ---- 3. alignment ----
    def save_alignment(self, result, name: str, subdir: str = "3_align") -> str:
        root = os.path.join(self.dir, subdir)
        _write_text(os.path.join(root, f"{name}.txt"), result.report())
        for i, f in enumerate(result.frames):
            save_png(os.path.join(root, f"{name}_photo{i}.png"), f)
        self.log(f"alignment '{name}': {'ok' if result.ok else 'FAILED'}, error {result.final_error_um:.2f} um")
        return root

    # ---- 4. finished piece ----
    def save_finished(
        self,
        mr: MapResult,
        pm: Optional[PlacementMap] = None,
        center_um: Optional[Tuple[float, float]] = None,
        proj: Optional[ProjectorCalibration] = None,
        subdir: str = "4_finished_piece",
    ) -> str:
        """Separate photos plus the stitched finished piece, with the expected tile outlines drawn on a copy."""
        root = self.save_map(mr, subdir)
        if mr.mosaic is not None:
            os.replace(os.path.join(root, "combined_map.png"), os.path.join(root, "stitched.png"))
            if pm is not None and center_um is not None:
                save_png_bgr(os.path.join(root, "stitched_expected_outline.png"), draw_expected_outline(mr, pm, center_um, proj))
        self.log(f"finished piece saved -> {root}")
        return root


def expected_outline_corners_px(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float], proj: Optional[ProjectorCalibration] = None
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """For each placed tile: (slot, its footprint's four corners in mosaic px: TL, TR, BR, BL).  With a projector
    calibration the outline follows the measured projector -> camera mapping (it may be rotated)."""
    out = []
    for t in pm.tiles:
        x0, y0, x1, y1 = t.crop
        out.append((t.slot, [tile_point_map_px(mr, pm, center_um, proj, t.slot, fx, fy)
                             for fx, fy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]))
    return out


def expected_outline_px(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float], proj: Optional[ProjectorCalibration] = None
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """For each placed tile: (slot, (x0, y0, x1, y1)) of the footprint it should have exposed, in mosaic px."""
    return [(s, (c[0][0], c[0][1], c[2][0], c[2][1])) for s, c in expected_outline_corners_px(mr, pm, center_um, proj)]


def draw_expected_outline(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float], proj: Optional[ProjectorCalibration] = None
) -> np.ndarray:
    base = mr.mosaic if mr.mosaic is not None else np.zeros((10, 10), np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base.copy()
    for slot, corners in expected_outline_corners_px(mr, pm, center_um, proj):
        col = TILE_COLORS[slot % len(TILE_COLORS)]
        cv2.polylines(img, [np.round(np.array(corners)).astype(np.int32)], True, col, 3)
        x0, y0 = corners[0]
        cv2.putText(img, f"tile {slot + 1}", (int(x0) + 14, int(y0) + 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
    return img


def map_finished_piece(
    rig: Rig,
    cfg: MapConfig,
    area_um: Tuple[float, float],
    center_um: Tuple[float, float],
    recorder: RunRecorder,
    pm: Optional[PlacementMap] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> MapResult:
    """Quick re-map of the area after patterning; saves separate photos and the stitch in the run folder."""
    # a patterned chip is full of repeating features (rings, lines); the stage position breaks the tie
    cfg = dataclasses.replace(cfg, ambiguity="stage_prior", label="finished piece")
    t0 = time.perf_counter()
    mr = map_area(rig, cfg, area_um, center_um, progress, should_abort)
    recorder.add_timing("finished piece scan", time.perf_counter() - t0, f"{mr.captured} photos, {'ok' if mr.ok else 'FAILED'}", "machine")
    get_proj = getattr(rig, "projector_calibration", None)
    recorder.save_finished(mr, pm, center_um, get_proj() if callable(get_proj) else None)
    return mr

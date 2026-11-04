"""One folder per Will Map test run: the photos, the combined map, the placement and the finished piece.

    will_map_runs/run_20261104_142300_label/
        run_log.txt                 what happened, in order, with times
        1_map/photos/               every camera photo used for the map, one file per tile
        1_map/combined_map.png      the stitched map
        1_map/map_report.txt, map_data.json
        2_placement/                placement preview on the white 3x3, stage targets, patterns used
        3_align/                    closed-loop alignment reports (and photos, when kept)
        4_finished_piece/photos/    quick re-map of the patterned piece, separate photos
        4_finished_piece/stitched.png and stitched_expected_outline.png

The folders live next to the code that runs them (``src/will_map_runs``).  PNGs are git-ignored; the
folder is in .gitignore as well so test runs never end up in a commit.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from will_map import (
    MapConfig,
    MapResult,
    PlacementMap,
    Rig,
    background_from_map,
    check_seams,
    render_placement,
    will_map,
)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "will_map_runs")

_PALETTE = [(40, 140, 255), (60, 170, 60), (200, 90, 200), (190, 150, 30), (80, 80, 220), (30, 190, 190)]


def _write_png(path: str, img: np.ndarray) -> None:
    """cv2.imwrite that also works with non-ASCII Windows paths.  3-channel input is RGB (camera order)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(path)


def _write_png_bgr(path: str, img_bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(path)


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

    def log(self, text: str) -> None:
        with open(os.path.join(self.dir, "run_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}  {text}\n")

    # ---- 1. map ----
    def save_map(self, mr: MapResult, subdir: str = "1_map") -> str:
        """Photos used for the map (in scan order), the combined map, a report and the numbers."""
        from will_map import snake_order

        root = os.path.join(self.dir, subdir)
        order = snake_order(mr.cfg.rows, mr.cfg.cols)
        for k, tile in enumerate(order):
            if tile in mr.frames:
                _write_png(os.path.join(root, "photos", tile_name(tile, k) + ".png"), mr.frames[tile])
        if mr.mosaic is not None:
            _write_png(os.path.join(root, "combined_map.png"), mr.mosaic)
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
                 "predicted_px": list(p.predicted_px), "peak": p.peak, "second_peak": p.second, "ok": p.ok, "reason": p.reason}
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
        _write_png_bgr(os.path.join(root, "placement_on_white_3x3.png"), white)
        if bg is not None:
            over, _ = render_placement(pm, images, scale=scale, background=bg)
            _write_png_bgr(os.path.join(root, "placement_on_camera_map.png"), over)
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
            _write_png(os.path.join(root, f"{name}_photo{i}.png"), f)
        self.log(f"alignment '{name}': {'ok' if result.ok else 'FAILED'}, error {result.final_error_um:.2f} um")
        return root

    # ---- 4. finished piece ----
    def save_finished(
        self,
        mr: MapResult,
        pm: Optional[PlacementMap] = None,
        center_um: Optional[Tuple[float, float]] = None,
        subdir: str = "4_finished_piece",
    ) -> str:
        """Separate photos plus the stitched finished piece, with the expected tile outlines drawn on a copy."""
        root = self.save_map(mr, subdir)
        if mr.mosaic is not None:
            os.replace(os.path.join(root, "combined_map.png"), os.path.join(root, "stitched.png"))
            if pm is not None and center_um is not None:
                _write_png_bgr(os.path.join(root, "stitched_expected_outline.png"), draw_expected_outline(mr, pm, center_um))
        self.log(f"finished piece saved -> {root}")
        return root


def expected_outline_px(
    mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float]
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """For each placed tile: (slot, (x0, y0, x1, y1)) of the footprint it should have exposed, in mosaic px."""
    out = []
    sg = mr.cfg.sign
    fw, fh = pm.geo.frame_w, pm.geo.frame_h
    for t, (tx, ty) in zip(pm.tiles, pm.stage_targets(center_um, sg)):
        pts = []
        for (fx, fy) in ((t.crop[0], t.crop[1]), (t.crop[2], t.crop[3])):
            dx = (fx - fw / 2.0) * pm.um_per_px[0]
            dy = (fy - fh / 2.0) * pm.um_per_px[1]
            pts.append(mr.stage_to_px(tx + sg[0] * dx, ty + sg[1] * dy))
        out.append((t.slot, (pts[0][0], pts[0][1], pts[1][0], pts[1][1])))
    return out


def draw_expected_outline(mr: MapResult, pm: PlacementMap, center_um: Tuple[float, float]) -> np.ndarray:
    base = mr.mosaic if mr.mosaic is not None else np.zeros((10, 10), np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base.copy()
    for slot, (x0, y0, x1, y1) in expected_outline_px(mr, pm, center_um):
        col = _PALETTE[slot % len(_PALETTE)]
        cv2.rectangle(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col, 3)
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
    cfg = dataclasses.replace(cfg, ambiguity="stage_prior")
    mr = will_map(rig, cfg, area_um, center_um, progress, should_abort)
    recorder.save_finished(mr, pm, center_um)
    return mr

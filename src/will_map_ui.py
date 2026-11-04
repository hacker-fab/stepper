"""tkinter front end for Will Map (matches the old UI: plain tkinter/ttk, no new dependencies).

* ``PatternPickerDialog``      asks for pattern 1, 2, 3 ... with a little grid that highlights the
                               slot currently being picked.
* ``PlacementApprovalWindow``  shows the placement on the white area (optionally over the camera
                               map), seam checks, nudge buttons and an Approve button.
* ``run_will_map``             the whole flow for the real machine (needs the old UI's model).
* ``python will_map_ui.py``    runs the same flow against the virtual white chip.
"""

from __future__ import annotations

import os
import time
import sys
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from will_map import (
    DEFAULT_CAM_UM_PER_PX,
    PROJ_UM_PER_PX,
    CrossGeometry,
    MapConfig,
    MapResult,
    PlacementMap,
    TilingTimeModel,
    area_for_tiles,
    background_from_map,
    budget_report,
    build_placement_map,
    check_seams,
    exposure_mask,
    load_image,
    measure_cross_geometry,
    render_placement,
    same_layout,
    scan_summary,
    spec_mismatches,
    status,
    will_map,
)
from will_map_runs import RunRecorder, map_finished_piece

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "will_map_samples")
# reading order TL, TR, BL, BR.  Patterns 3, 2, 1 are the TR, BL and BR tiles (their crosses only
# line up in that arrangement); the top-left tile was not supplied, so it is crosses-only.
SAMPLE_SET = [
    "pattern_4_crosses_only.png",
    "pattern_3.png",
    "pattern_2.png",
    "pattern_1.png",
]


def slot_name(r: int, c: int, rows: int, cols: int) -> str:
    if rows == 2 and cols == 2:
        return ("top" if r == 0 else "bottom") + "-" + ("left" if c == 0 else "right")
    return f"row {r + 1}, column {c + 1}"


# --------------------------------------------------------------------------------------
# Pattern picker
# --------------------------------------------------------------------------------------
class PatternPickerDialog:
    CELL_W, CELL_H, GAP = 168, 95, 14  # 16:9 cells like the frames themselves

    def __init__(
        self,
        parent: tk.Misc,
        rows: int = 2,
        cols: int = 2,
        sample_paths: Optional[Sequence[str]] = None,
        title: str = "Select pattern images",
    ):
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        self.sample_paths = list(sample_paths) if sample_paths else None
        self.paths: List[Optional[str]] = [None] * self.n
        self.geos: List[Optional[CrossGeometry]] = [None] * self.n
        self.thumbs: List[Optional[ImageTk.PhotoImage]] = [None] * self.n
        self.current = 0
        self.result: Optional[List[str]] = None

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.resizable(False, False)
        try:
            self.win.transient(parent)
        except tk.TclError:
            pass

        self.header = ttk.Label(self.win, font=("TkDefaultFont", 12, "bold"), wraplength=520)
        self.header.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 4), sticky="w")
        w = self.cols * (self.CELL_W + self.GAP) + self.GAP
        h = self.rows * (self.CELL_H + self.GAP) + self.GAP
        self.canvas = tk.Canvas(self.win, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.grid(row=1, column=0, padx=14, pady=6, sticky="n")
        side = ttk.Frame(self.win)
        side.grid(row=1, column=1, padx=(0, 14), pady=6, sticky="nsew")
        side.columnconfigure(0, minsize=300)
        self.path_var = tk.StringVar(value="(no file chosen)")
        ttk.Label(side, textvariable=self.path_var, wraplength=300, justify="left").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar()
        self.status = tk.Label(side, textvariable=self.status_var, wraplength=300, justify="left", fg="#444")
        self.status.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.browse_btn = ttk.Button(side, text="Browse...", command=self.browse)
        self.browse_btn.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.sample_btn = ttk.Button(side, text="Use sample set", command=self.use_samples)
        self.sample_btn.grid(row=3, column=0, sticky="w", pady=(6, 0))
        if not self.sample_paths:
            self.sample_btn.state(["disabled"])

        bar = ttk.Frame(self.win)
        bar.grid(row=2, column=0, columnspan=2, padx=14, pady=(6, 12), sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.back_btn = ttk.Button(bar, text="< Back", command=self.back)
        self.back_btn.grid(row=0, column=1, padx=4)
        self.next_btn = ttk.Button(bar, text="Next >", command=self.next)
        self.next_btn.grid(row=0, column=2, padx=4)
        self._refresh()

    # ---- logic (also callable from tests) ----
    def try_set(self, path: str) -> Optional[str]:
        """Accept ``path`` for the current slot.  Returns an error message, or None on success."""
        try:
            img = load_image(path)
            geo = measure_cross_geometry(img)
        except Exception as e:  # noqa: BLE001 - shown to the user
            return str(e)
        for k, g in enumerate(self.geos):
            if g is not None and k != self.current and not same_layout(g, geo):
                return (
                    f"Cross layout differs from pattern {k + 1} (frame {geo.frame_w}x{geo.frame_h}, "
                    f"crosses {geo.cross_w}x{geo.cross_h}). All tiles must share one layout."
                )
        tw, th = self.CELL_W - 8, self.CELL_H - 8
        small = cv2.resize(img[..., :3] if img.ndim == 3 else img, (tw, th), interpolation=cv2.INTER_AREA)
        small = np.clip(small.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)  # keep thin lines visible
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB) if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        self.thumbs[self.current] = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.paths[self.current] = path
        self.geos[self.current] = geo
        self._ok = (
            f"OK: {geo.frame_w}x{geo.frame_h} px, crosses {geo.cross_w}x{geo.cross_h} px, "
            f"cross pitch {geo.pitch_x} x {geo.pitch_y} px"
        )
        return None

    def browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.win,
            title=f"Image for pattern {self.current + 1}",
            initialdir=SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else None,
            filetypes=[("Pattern images", "*.png *.bmp *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.choose(path)

    def choose(self, path: str) -> bool:
        err = self.try_set(path)
        if err:
            self.status.configure(fg="#B00020")
            self.status_var.set("Not accepted: " + err)
            return False
        self.status.configure(fg="#1B6E2D")
        self.status_var.set(self._ok)
        self._refresh()
        return True

    def use_samples(self) -> None:
        if not self.sample_paths:
            return
        for k, p in enumerate(self.sample_paths[: self.n]):
            self.current = k
            if not self.choose(p):
                break
        else:
            self.current = self.n - 1
        self._refresh()

    def next(self) -> None:
        if self.paths[self.current] is None:
            return
        if self.current < self.n - 1:
            self.current += 1
            self.status_var.set("")
            self._refresh()
        else:
            self.result = [p for p in self.paths if p is not None]
            self.win.destroy()

    def back(self) -> None:
        if self.current > 0:
            self.current -= 1
            self.status_var.set("")
            self._refresh()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> Optional[List[str]]:
        self.win.grab_set()
        self.win.wait_window()
        return self.result

    # ---- drawing ----
    def _refresh(self) -> None:
        r, c = divmod(self.current, self.cols)
        self.header.configure(
            text=f"Pattern {self.current + 1} of {self.n}: choose the image for the "
            f"{slot_name(r, c, self.rows, self.cols)} tile (row {r + 1}, column {c + 1})"
        )
        p = self.paths[self.current]
        self.path_var.set(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" if p else "(no file chosen)")
        self.next_btn.configure(text="Finish" if self.current == self.n - 1 else "Next >")
        self.next_btn.state(["!disabled"] if p else ["disabled"])
        self.back_btn.state(["!disabled"] if self.current > 0 else ["disabled"])
        cv = self.canvas
        cv.delete("all")
        for s in range(self.n):
            rr, cc = divmod(s, self.cols)
            x0 = self.GAP + cc * (self.CELL_W + self.GAP)
            y0 = self.GAP + rr * (self.CELL_H + self.GAP)
            x1, y1 = x0 + self.CELL_W, y0 + self.CELL_H
            if s == self.current:
                cv.create_rectangle(x0 - 5, y0 - 5, x1 + 5, y1 + 5, fill="#FFE2B0", outline="#E07B00", width=4)
            fill = "#F4F4F4" if self.thumbs[s] is None else "#000000"
            cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#888" if s != self.current else "#E07B00", width=2)
            if self.thumbs[s] is not None:
                cv.create_image(x0 + 4, y0 + 4, image=self.thumbs[s], anchor="nw")
                cv.create_rectangle(x0, y0, x0 + 26, y0 + 22, fill="#FFFFFF", outline="")
                cv.create_text(x0 + 13, y0 + 11, text=str(s + 1), font=("TkDefaultFont", 11, "bold"), fill="#1B6E2D")
            else:
                cv.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2 - (8 if s == self.current else 0), text=str(s + 1),
                    font=("TkDefaultFont", 26, "bold"), fill="#E07B00" if s == self.current else "#999",
                )
                if s == self.current:
                    cv.create_text((x0 + x1) / 2, y1 - 14, text="picking this one", fill="#9A5400", font=("TkDefaultFont", 9))


# --------------------------------------------------------------------------------------
# Area dialog: how many tiles -> how much area gets mapped
# --------------------------------------------------------------------------------------
class AreaDialog:
    """Asks for the number of tiles (wide x high) and shows the area that will be mapped: one more complete
    frame than tiles in each direction, centred on the current stage position.  Confirm starts the map."""

    CANVAS_W, CANVAS_H = 400, 240

    def __init__(
        self,
        parent: tk.Misc,
        cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
        cam_size: Tuple[int, int] = (1920, 1080),
        rows: int = 2,
        cols: int = 2,
    ):
        self.cam_um_per_px = cam_um_per_px
        self.cam_size = cam_size
        self.result: Optional[Tuple[int, int]] = None  # (tile rows, tile cols)
        self.win = tk.Toplevel(parent)
        self.win.title("Will Map: area to map")
        self.win.protocol("WM_DELETE_WINDOW", self.cancel)
        self.win.resizable(False, False)
        ttk.Label(self.win, text="Map the area around the current stage position.\nHow many tiles will the pattern have?").grid(
            row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w"
        )
        self.cols_var = tk.IntVar(value=cols)
        self.rows_var = tk.IntVar(value=rows)
        form = ttk.Frame(self.win)
        form.grid(row=1, column=0, columnspan=2, padx=14, sticky="w")
        ttk.Label(form, text="tiles wide").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=8, width=4, textvariable=self.cols_var, command=self.refresh).grid(row=0, column=1, padx=8)
        ttk.Label(form, text="tiles high").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=8, width=4, textvariable=self.rows_var, command=self.refresh).grid(row=1, column=1, padx=8)
        for v in (self.cols_var, self.rows_var):
            v.trace_add("write", lambda *_: self.refresh())
        self.canvas = tk.Canvas(self.win, width=self.CANVAS_W, height=self.CANVAS_H, bg="#e8e8e8", highlightthickness=0)
        self.canvas.grid(row=2, column=0, padx=14, pady=10)
        self.info = tk.Text(self.win, width=58, height=15, font=("Courier", 9), wrap="word", state="disabled")
        self.info.grid(row=2, column=1, padx=(0, 14), pady=10, sticky="n")
        bar = ttk.Frame(self.win)
        bar.grid(row=3, column=0, columnspan=2, pady=(0, 14), padx=14, sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.confirm_btn = ttk.Button(bar, text="Confirm and map", command=self.confirm)
        self.confirm_btn.grid(row=0, column=1, padx=4)
        self.refresh()

    def _counts(self) -> Optional[Tuple[int, int]]:
        try:
            r, c = int(self.rows_var.get()), int(self.cols_var.get())
        except (tk.TclError, ValueError):
            return None
        return (r, c) if 1 <= r <= 8 and 1 <= c <= 8 else None

    def info_lines(self) -> List[str]:
        counts = self._counts()
        if counts is None:
            return ["enter 1 to 8 tiles in each direction"]
        r, c = counts
        pm = area_for_tiles(r, c)
        w_um, h_um = pm.area_size_um
        cfg = MapConfig(um_per_px=self.cam_um_per_px, cam_size=self.cam_size)
        sc = scan_summary((w_um, h_um), cfg)
        x0 = min(t.dest[0] for t in pm.tiles)
        y0 = min(t.dest[1] for t in pm.tiles)
        x1 = max(t.dest[2] for t in pm.tiles)
        y1 = max(t.dest[3] for t in pm.tiles)
        bw, bh = (x1 - x0) * pm.um_per_px[0], (y1 - y0) * pm.um_per_px[1]
        lines = [
            f"pattern: {c} wide x {r} high = {c * r} tiles",
            f"area to map: {c + 1} x {r + 1} complete frames",
            f"space occupied: {w_um:.0f} x {h_um:.0f} um  ({w_um / 1000:.2f} x {h_um / 1000:.2f} mm)",
            f"pattern block inside: {bw:.0f} x {bh:.0f} um, centred",
            "",
            f"map photos: {sc['cols']} x {sc['rows']} ({sc['photos']}), step "
            f"{sc['step_fraction'][0] * 100:.0f}% x / {sc['step_fraction'][1] * 100:.0f}% y of a frame (max 75%)",
            f"estimated mapping time: {sc['est_s']:.0f} s (limit 300 s), with autofocus after each move",
            "",
            "The current stage position becomes the centre of the area.",
            "Tiles can be inserted once the map is done.",
        ]
        if sc["est_s"] > 300:
            lines.insert(7, "WARNING: this is over the 5 minute mapping target")
        return lines

    def refresh(self) -> None:
        lines = self.info_lines()
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        self.canvas.delete("all")
        counts = self._counts()
        self.confirm_btn.state(["!disabled"] if counts else ["disabled"])
        if counts is None:
            return
        r, c = counts
        fr, fc = r + 1, c + 1
        pad = 22
        cell = min((self.CANVAS_W - 2 * pad) / fc, (self.CANVAS_H - 2 * pad) / fr * 16 / 9)  # cell width; frames are 16:9
        cell_h = cell * 9 / 16
        x0 = (self.CANVAS_W - cell * fc) / 2
        y0 = (self.CANVAS_H - cell_h * fr) / 2
        self.canvas.create_rectangle(x0, y0, x0 + cell * fc, y0 + cell_h * fr, fill="white", outline="#888")
        for i in range(1, fc):
            self.canvas.create_line(x0 + i * cell, y0, x0 + i * cell, y0 + cell_h * fr, fill="#d0d0d0")
        for j in range(1, fr):
            self.canvas.create_line(x0, y0 + j * cell_h, x0 + cell * fc, y0 + j * cell_h, fill="#d0d0d0")
        # the pattern block sits half a frame in from every edge, one tile per cell
        for j in range(r):
            for i in range(c):
                tx, ty = x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell_h
                self.canvas.create_rectangle(tx, ty, tx + cell, ty + cell_h, fill="#cfe3ff", outline="#2b6cb0")
                self.canvas.create_text(tx + cell / 2, ty + cell_h / 2, text=str(j * c + i + 1), fill="#2b6cb0", font=("TkDefaultFont", 11, "bold"))
        self.canvas.create_text(self.CANVAS_W / 2, y0 + cell_h * fr + 12, text=f"{fc} x {fr} frames", fill="#555")

    def confirm(self) -> None:
        counts = self._counts()
        if counts is None:
            return
        self.result = counts
        self.win.destroy()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> Optional[Tuple[int, int]]:
        self.win.grab_set()
        self.win.wait_window()
        return self.result


# --------------------------------------------------------------------------------------
# Placement approval
# --------------------------------------------------------------------------------------
@dataclass
class ApprovedPlacement:
    placement: PlacementMap
    stage_targets: List[Tuple[float, float]]
    area_center_um: Tuple[float, float]
    seams: List[dict]
    images: List[np.ndarray] = field(default_factory=list, repr=False)  # the patterns, for the run folder
    paths: List[str] = field(default_factory=list)
    map_result: Optional[MapResult] = field(default=None, repr=False)

    def lines(self) -> List[str]:
        out = self.placement.report()
        for k, (x, y) in enumerate(self.stage_targets):
            out.append(f"stage target tile {k + 1}: x {x:.1f} um, y {y:.1f} um")
        return out


class PlacementApprovalWindow:
    CANVAS_W, CANVAS_H = 880, 500

    def __init__(
        self,
        parent: tk.Misc,
        images: Sequence[np.ndarray],
        paths: Optional[Sequence[str]] = None,
        rows: int = 2,
        cols: int = 2,
        area_rows: int = 3,
        area_cols: int = 3,
        area_center_um: Tuple[float, float] = (0.0, 0.0),
        map_result: Optional[MapResult] = None,
        um_per_px: Tuple[float, float] = PROJ_UM_PER_PX,
        on_approve: Optional[Callable[[ApprovedPlacement], None]] = None,
        sign: Tuple[int, int] = (1, -1),
        mapped_area_um: Optional[Tuple[float, float]] = None,
    ):
        self.mapped_area_um = mapped_area_um
        self.images = list(images)
        self.masks = [exposure_mask(i) for i in self.images]  # cached; used by every redraw
        self.paths = list(paths) if paths else []
        self.rows, self.cols, self.area_rows, self.area_cols = rows, cols, area_rows, area_cols
        self.area_center = area_center_um
        self.map_result = map_result
        self.um_per_px = um_per_px
        self.on_approve = on_approve
        self.sign = sign
        self.geo = measure_cross_geometry(self.images[0])
        self.seams = check_seams(self.images, rows, cols, self.geo)
        self.spec_notes = spec_mismatches(self.geo)  # differences from the CROSS_* inputs in will_map.py
        self.nudge = [0.0, 0.0]
        self.result: Optional[ApprovedPlacement] = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        self.win = tk.Toplevel(parent)
        self.win.title("Placement map: review and approve")
        self.win.protocol("WM_DELETE_WINDOW", self.cancel)
        self.canvas = tk.Canvas(self.win, width=self.CANVAS_W, height=self.CANVAS_H, bg="#e8e8e8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        side = ttk.Frame(self.win)
        side.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        self.mode_var = tk.StringVar(value="trimmed")
        box = ttk.LabelFrame(side, text="Exposure mode")
        box.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(box, text="Split at the crosses (footprints)", value="trimmed", variable=self.mode_var, command=self.refresh).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="Whole frames at half-frame offset", value="whole", variable=self.mode_var, command=self.refresh).grid(row=1, column=0, sticky="w")
        self.bg_var = tk.BooleanVar(value=False)
        self.bg_check = ttk.Checkbutton(side, text="Show camera map underneath", variable=self.bg_var, command=self.refresh)
        self.bg_check.grid(row=1, column=0, sticky="w", pady=(6, 0))
        if map_result is None or map_result.mosaic is None:
            self.bg_check.state(["disabled"])

        nb = ttk.LabelFrame(side, text="Nudge placement (um)")
        nb.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.step_var = tk.StringVar(value="10")
        ttk.Entry(nb, textvariable=self.step_var, width=6).grid(row=0, column=1)
        ttk.Button(nb, text="^", width=3, command=lambda: self.move(0, -1)).grid(row=0, column=3)
        ttk.Button(nb, text="<", width=3, command=lambda: self.move(-1, 0)).grid(row=1, column=2)
        ttk.Button(nb, text="Reset", width=6, command=self.reset_nudge).grid(row=1, column=3)
        ttk.Button(nb, text=">", width=3, command=lambda: self.move(1, 0)).grid(row=1, column=4)
        ttk.Button(nb, text="v", width=3, command=lambda: self.move(0, 1)).grid(row=2, column=3)
        ttk.Label(nb, text="step").grid(row=0, column=0)

        self.info = tk.Text(side, width=64, height=24, font=("Courier", 9), wrap="word", state="disabled")
        self.info.grid(row=3, column=0, pady=(8, 0), sticky="nsew")
        side.rowconfigure(3, weight=1)
        bar = ttk.Frame(side)
        bar.grid(row=4, column=0, pady=(10, 0), sticky="e")
        ttk.Button(bar, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=4)
        self.approve_btn = ttk.Button(bar, text="Approve placement", command=self.approve)
        self.approve_btn.grid(row=0, column=1, padx=4)
        self.refresh()

    # ---- state ----
    def placement(self) -> PlacementMap:
        return build_placement_map(
            self.geo, self.rows, self.cols, self.area_rows, self.area_cols,
            mode=self.mode_var.get(), um_per_px=self.um_per_px, nudge_um=(self.nudge[0], self.nudge[1]),
        )

    def move(self, dx: int, dy: int) -> None:
        try:
            step = float(self.step_var.get())
        except ValueError:
            step = 10.0
        self.nudge[0] += dx * step
        self.nudge[1] += dy * step
        self.refresh()

    def reset_nudge(self) -> None:
        self.nudge = [0.0, 0.0]
        self.refresh()

    # ---- drawing ----
    def refresh(self) -> None:
        pm = self.placement()
        pad = 600
        total_w = pm.area_size_px[0] + 2 * pad
        scale = min(self.CANVAS_W / total_w, self.CANVAS_H / (pm.area_size_px[1] + 2 * pad))
        bg = None
        if self.bg_var.get() and self.map_result is not None and self.map_result.mosaic is not None:
            bg = background_from_map(self.map_result, pm, scale)
        img, _ = render_placement(pm, self.masks, scale=scale, background=bg, pad_px=pad)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.CANVAS_W // 2, self.CANVAS_H // 2, image=self._photo)

        lines = pm.report()
        lines.append(f"nudge: {self.nudge[0]:+.0f} um x, {self.nudge[1]:+.0f} um y")
        lines.append("")
        lines.append("crosses that must coincide between neighbours:")
        for s in self.seams:
            a, b = s["a"] + 1, s["b"] + 1
            where = "left|right" if s["dir"] == "h" else "top|bottom"
            lines.append(f"  tile {a} / tile {b} ({where}): " + ("match" if s["ok"] else f"MISMATCH, {s['mismatch_px']} px differ - wrong order?"))
        lines.append("")
        if self.spec_notes:
            lines.append("cross inputs (CROSS_* in will_map.py) DIFFER from this pattern:")
            lines += [f"  {n}" for n in self.spec_notes]
            lines.append("  keep cross size and distances identical on every tile!")
        else:
            lines.append("cross size/distances match the CROSS_* inputs in will_map.py")
        self.too_big = False
        if self.mapped_area_um is not None:
            over = (pm.area_size_um[0] - self.mapped_area_um[0], pm.area_size_um[1] - self.mapped_area_um[1])
            if max(over) > 25.0:
                self.too_big = True
                lines.append(
                    f"WARNING: this placement needs {pm.area_size_um[0]:.0f} x {pm.area_size_um[1]:.0f} um but only "
                    f"{self.mapped_area_um[0]:.0f} x {self.mapped_area_um[1]:.0f} um was mapped (cross layout differs from CROSS_*)"
                )
        targets = pm.stage_targets(self.area_center, self.sign)
        lines.append("")
        lines.append("stage targets (um):")
        for k, (x, y) in enumerate(targets):
            lines.append(f"  tile {k + 1}: x {x:9.1f}   y {y:9.1f}")
        if self.map_result is not None:
            rep = budget_report(self.map_result.timing.get("total_s", 0.0), n_tiles=len(pm.tiles), model=TilingTimeModel())
            lines.append("")
            lines.append(f"map took {rep['map_s']:.0f} s (limit {rep['map_limit_s']:.0f} s)")
            lines.append(
                f"tiling overhead ~{rep['tile_overhead_s']:.0f} s (assumed), leaves up to "
                f"{rep['max_exposure_per_tile_s']:.0f} s exposure per tile in {rep['tile_limit_s']:.0f} s"
            )
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.configure(state="disabled")
        bad = any(not s["ok"] for s in self.seams)
        self.approve_btn.configure(
            text="Approve placement (outside the mapped area!)" if self.too_big
            else "Approve placement (seam mismatch!)" if bad else "Approve placement"
        )

    # ---- actions ----
    def approve(self) -> None:
        pm = self.placement()
        self.result = ApprovedPlacement(
            pm, pm.stage_targets(self.area_center, self.sign), self.area_center, self.seams,
            self.images, self.paths, self.map_result,
        )
        if self.on_approve:
            self.on_approve(self.result)
        self.win.destroy()

    def cancel(self) -> None:
        self.result = None
        self.win.destroy()


# --------------------------------------------------------------------------------------
# Whole flow
# --------------------------------------------------------------------------------------
class ModelRig:
    """Rig for the old UI: wraps the EventDispatcher ('model') the way TilingCheckFrame uses it."""

    def __init__(self, model, z: Optional[float] = None):
        self.model = model
        self.z = model.stage_setpoint[2] if z is None else z

    def move_to(self, x_um: float, y_um: float) -> None:
        # move_absolute fills missing axes with 0, so always pass z
        if self.model.move_absolute({"x": x_um, "y": y_um, "z": self.z}) is False:
            raise RuntimeError(f"stage refused the move to ({x_um:.0f}, {y_um:.0f}) um (out of bounds?)")

    def settle(self, seconds: float) -> None:
        self.model.non_blocking_delay(seconds)

    def autofocus(self) -> None:
        """Red autofocus after a movement, as the old tiling loop does (model.autofocus(blue_only=False)).
        Toby's autofocus lives in his Qt engine and cannot be called from here; when it is ported, swap it in here."""
        af = getattr(self.model, "autofocus", None)
        if callable(af):
            af(blue_only=False)
        self.z = self.model.stage_setpoint[2]  # autofocus moved z; keep later moves at the focused height

    def grab(self) -> np.ndarray:
        img = self.model.camera_image
        if img is None:
            raise RuntimeError("no camera image available")
        return np.array(img)


class _Progress:
    def __init__(self, parent: tk.Misc, title: str = "Will Map"):
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.resizable(False, False)
        self.label = ttk.Label(self.win, text="starting...", width=60)
        self.label.grid(row=0, column=0, padx=14, pady=(14, 6))
        self.bar = ttk.Progressbar(self.win, length=420, maximum=1.0)
        self.bar.grid(row=1, column=0, padx=14, pady=6)
        self.aborted = False
        ttk.Button(self.win, text="Abort", command=self._abort).grid(row=2, column=0, pady=(6, 14))

    def _abort(self) -> None:
        self.aborted = True

    def update(self, frac: float, msg: str) -> None:
        self.bar["value"] = frac
        self.label.configure(text=msg)
        self.win.update()

    def close(self) -> None:
        self.win.destroy()


DEFAULT_TILE_GRID = (2, 2)  # (rows, cols) used when the tiling panel holds no grid yet (the 2x2 proof of concept)
MAX_TILE_GRID = 8


def default_tile_counts(get_panel_counts: Optional[Callable[[], Tuple[object, object]]] = None) -> Tuple[Tuple[int, int], str]:
    """Tile grid (rows, cols) for the map, taken from the tiling panel instead of being typed in again.

    The tiling panel's X / Y "Amount" fields hold the grid (segmenting a big pattern fills them in).  Negative
    amounts only mean 'go the other way', so the absolute value is used.  If the panel still has its untouched
    1 x 1 default (or cannot be read) the 2 x 2 proof-of-concept grid is used.  Returns (rows, cols), why.
    """
    if get_panel_counts is not None:
        try:
            rows_raw, cols_raw = get_panel_counts()
            rows, cols = abs(int(float(str(rows_raw)))), abs(int(float(str(cols_raw))))
        except (TypeError, ValueError):
            rows = cols = 0
        if rows * cols > 1:
            clamped = (min(rows, MAX_TILE_GRID), min(cols, MAX_TILE_GRID))
            note = "" if clamped == (rows, cols) else f" (limited to {MAX_TILE_GRID} per direction)"
            return clamped, f"{cols} x {rows} tiles from the tiling panel{note}"
    r, c = DEFAULT_TILE_GRID
    return (r, c), f"{c} x {r} tiles (default, the tiling panel holds no grid yet)"


def run_will_map(
    parent: tk.Misc,
    model,
    cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX,
    on_approved: Optional[Callable[[ApprovedPlacement], None]] = None,
    tile_counts: Optional[Callable[[], Tuple[object, object]]] = None,
) -> Optional[ApprovedPlacement]:
    """Tile grid (taken from the tiling panel, shown in the area dialog to confirm) -> camera map around the
    current stage position -> ONLY THEN pick the pattern for each tile -> placement approval."""
    cam = np.asarray(model.camera_image)
    cam_size = (cam.shape[1], cam.shape[0])
    (rows0, cols0), why = default_tile_counts(tile_counts)
    status(f"tile grid: {why}")
    t_dialog = time.perf_counter()
    counts = AreaDialog(parent, cam_um_per_px, cam_size, rows=rows0, cols=cols0).show()
    dialog_s = time.perf_counter() - t_dialog
    if not counts:
        return None
    tile_rows, tile_cols = counts
    area_pm = area_for_tiles(tile_rows, tile_cols)
    cfg = MapConfig(um_per_px=cam_um_per_px, cam_size=cam_size)
    center = (float(model.stage_setpoint[0]), float(model.stage_setpoint[1]))
    status(
        f"mapping {area_pm.area_size_um[0]:.0f} x {area_pm.area_size_um[1]:.0f} um "
        f"({tile_cols + 1} x {tile_rows + 1} complete frames) around stage ({center[0]:.0f}, {center[1]:.0f}) um"
    )
    recorder = RunRecorder(label="map")  # one folder per run, next to this file
    recorder.add_timing("area dialog", dialog_s, "confirm the tile grid", "user")
    try:
        model.will_run = recorder  # the finished-piece scan writes into the same folder
    except Exception:  # noqa: BLE001
        pass
    prog = _Progress(parent)
    t_map = time.perf_counter()
    try:
        result = will_map(ModelRig(model), cfg, area_pm.area_size_um, center, prog.update, lambda: prog.aborted)
    except Exception as e:  # noqa: BLE001
        prog.close()
        recorder.log(f"mapping stopped: {e}")
        messagebox.showerror("Will Map", f"Mapping stopped: {e}")
        return None
    prog.close()
    print("\n".join(result.report()))
    recorder.add_timing("map", time.perf_counter() - t_map, f"{result.captured} photos, {'ok' if result.ok else 'FAILED'}", "machine")
    t_save = time.perf_counter()
    recorder.save_map(result)  # photos + combined map are kept even when the map fails (useful for debugging)
    recorder.add_timing("saving map photos and files", time.perf_counter() - t_save, "", "detail")
    if not result.ok:
        messagebox.showwarning("Will Map", "\n".join(result.report()) + f"\n\nPhotos saved in:\n{recorder.dir}")
        return None

    # the map is good: now (and only now) the tiles can be inserted
    sample_paths = [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]
    sample_paths = sample_paths if (tile_rows, tile_cols) == (2, 2) and all(os.path.exists(p) for p in sample_paths) else None
    t_pick = time.perf_counter()
    paths = PatternPickerDialog(parent, tile_rows, tile_cols, sample_paths).show()
    recorder.add_timing("pattern picking", time.perf_counter() - t_pick, "choose the pattern for each tile", "user")
    if not paths:
        recorder.log("no patterns chosen")
        return None
    images = [load_image(p) for p in paths]
    t_approve = time.perf_counter()
    win = PlacementApprovalWindow(
        parent, images, paths, rows=tile_rows, cols=tile_cols, area_rows=tile_rows + 1, area_cols=tile_cols + 1,
        area_center_um=center, map_result=result, on_approve=on_approved, mapped_area_um=area_pm.area_size_um,
    )
    win.win.grab_set()
    win.win.wait_window()
    recorder.add_timing("placement approval", time.perf_counter() - t_approve, "check and approve the placement", "user")
    if win.result is not None:
        print("\n".join(win.result.lines()))
        t_place = time.perf_counter()
        recorder.save_placement(win.result.placement, images, paths, center, result)
        recorder.add_timing("saving placement preview", time.perf_counter() - t_place, "", "detail")
        try:
            model.will_placement = win.result  # kept for the exposure step that comes next
        except Exception:  # noqa: BLE001
            pass
    else:
        recorder.log("placement not approved")
    return win.result


def run_finished_piece(parent: tk.Misc, model, cam_um_per_px: float = DEFAULT_CAM_UM_PER_PX) -> Optional[MapResult]:
    """After patterning: quick re-map of the same area, saved as separate photos + one stitched image
    in the run folder of the approved placement (4_finished_piece)."""
    placement: Optional[ApprovedPlacement] = getattr(model, "will_placement", None)
    recorder: Optional[RunRecorder] = getattr(model, "will_run", None)
    if placement is None or recorder is None:
        messagebox.showinfo("Will Map", "Run Will Map and approve a placement first.")
        return None
    cam = np.asarray(model.camera_image)
    cfg = MapConfig(um_per_px=cam_um_per_px, cam_size=(cam.shape[1], cam.shape[0]))
    prog = _Progress(parent, "Will Map: finished piece")
    try:
        res = map_finished_piece(
            ModelRig(model), cfg, placement.placement.area_size_um, placement.area_center_um, recorder,
            placement.placement, prog.update, lambda: prog.aborted,
        )
    except Exception as e:  # noqa: BLE001
        prog.close()
        recorder.log(f"finished-piece scan stopped: {e}")
        messagebox.showerror("Will Map", f"Finished-piece scan stopped: {e}")
        return None
    prog.close()
    messagebox.showinfo("Will Map", "\n".join(res.report()) + f"\n\nSaved in:\n{recorder.dir}")
    return res


def _virtual_demo() -> None:
    """The whole flow against the virtual white chip (no hardware)."""
    from will_map_virtual import (
        VirtualChip, VirtualRig, load_samples, run_sample_comparison, simulate_patterning,
    )

    root = tk.Tk()
    root.title("Will Map - virtual test")
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
        prog = _Progress(root)
        res = will_map(rig, MapConfig(), area_pm.area_size_um, center, prog.update, lambda: prog.aborted)
        prog.close()
        print("\n".join(res.report()))
        rec.save_map(res)
        if not res.ok:
            messagebox.showwarning("Will Map", "\n".join(res.report()))
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
            prog = _Progress(root, "Virtual patterning")
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
        prog = _Progress(root, "Sample run (2 areas)")
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


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    _virtual_demo()

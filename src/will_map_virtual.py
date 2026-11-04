"""Virtual environment for testing Will Map without the machine.

A white chip (the "3x3 area" plus a margin) with faint scratches, dust and surface texture so the
camera has something to register on, a stage with backlash / noise / drift, and a camera that
renders 1920x1080 frames of the chip at the true stage position.

Chip coordinates are um with x right, y DOWN, and (0, 0) at the centre of the area.
"""

from __future__ import annotations

import math
import os
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
        drift_um: Tuple[float, float] = (0.35, 0.2),
        sensor_noise: float = 1.5,
        seed: int = 0,
        cam_size: Tuple[int, int] = CAMERA_SIZE_PX,
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
        self.log: List[dict] = []  # one entry per grab: commanded stage and true chip position

    # ---- Rig protocol ----
    def move_to(self, x_um: float, y_um: float) -> None:
        target = [x_um, y_um]
        for ax in (0, 1):
            d = target[ax] - self.cmd[ax]
            if abs(d) > 1e-9:
                self._dir[ax] = 1 if d > 0 else -1
        self._moves += 1
        self.cmd = target
        self.actual = [
            target[ax]
            - self._dir[ax] * self.backlash_um / 2.0
            + self.drift_um[ax] * self._moves
            + float(self.rng.normal(0, self.noise_um))
            for ax in (0, 1)
        ]

    def settle(self, seconds: float) -> None:
        self.sim_settle_s += seconds  # simulated, no real waiting

    def grab(self) -> np.ndarray:
        chip_xy = self.true_chip_xy()
        g = self.chip.render(chip_xy, self.cam_size, self.true_um_per_px, self.rot_deg).astype(np.float32)
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
        return (
            self.sign[0] * (self.actual[0] - self.center[0]) + self.chip_offset[0],
            self.sign[1] * (self.actual[1] - self.center[1]) + self.chip_offset[1],
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
) -> dict:
    """Move to every tile target (optionally with the closed-loop alignment) and 'expose' it onto the
    virtual chip.  Returns the true tile positions and how well the tile spacing came out."""
    from will_align import FeatureMeasurer, align_to_target
    from will_map import exposure_mask

    note = progress or (lambda m: None)
    targets = pm.stage_targets(center_um, cfg.sign)
    fm = FeatureMeasurer(mr) if closed_loop else None
    pos, photos, aligns = [], 0, []
    for t, T in zip(pm.tiles, targets):
        note(f"tile {t.slot + 1}: {'aligning' if closed_loop else 'moving'}")
        if closed_loop:
            ar = align_to_target(rig, T, [fm], cfg)
            photos += len(ar.steps)
            aligns.append(ar)
            if recorder is not None:
                recorder.save_alignment(ar, f"tile{t.slot + 1}")
        else:
            rig.move_to(*T)
            rig.settle(cfg.t_settle_s)
        note(f"tile {t.slot + 1}: exposing (virtual)")
        pos.append(rig.expose(exposure_mask(images[t.slot]), pm.um_per_px, t.crop))
    ideal = [rig.ideal_chip_xy(T) for T in targets]
    spacing = [
        math.hypot(
            (pos[k][0] - pos[0][0]) - (ideal[k][0] - ideal[0][0]),
            (pos[k][1] - pos[0][1]) - (ideal[k][1] - ideal[0][1]),
        )
        for k in range(1, len(pos))
    ]
    return {"true_positions_um": pos, "spacing_error_um": spacing, "photos": photos, "alignments": aligns}


def run_virtual_session(
    base_dir: Optional[str] = None,
    label: str = "virtual",
    closed_loop: bool = True,
    seed: int = 5,
    center_um: Tuple[float, float] = (5000.0, 3000.0),
    progress=None,
    **rig_kw,
) -> dict:
    """Everything the real machine will do, on the white virtual chip, with a run folder at the end."""
    from will_map import (
        MapConfig, build_placement_map, load_image, measure_cross_geometry, will_map,
    )
    from will_map_runs import RunRecorder, map_finished_piece

    paths = [os.path.join(SAMPLE_DIR, f) for f in SAMPLE_SET]
    images = [load_image(p) for p in paths]
    geo = measure_cross_geometry(images[1])
    pm = build_placement_map(geo)
    cfg = MapConfig()
    rec = RunRecorder(base_dir, label)
    rig = VirtualRig(VirtualChip(pm.area_size_um, seed=11), center_um=center_um, seed=seed, **rig_kw)
    note = progress or (lambda m: None)
    note("mapping the white area")
    mr = will_map(rig, cfg, pm.area_size_um, center_um)
    rec.save_map(mr)
    out = {"recorder": rec, "map": mr, "placement": pm, "rig": rig}
    if not mr.ok:
        return out
    rec.save_placement(pm, images, paths, center_um, mr)
    out["patterning"] = simulate_patterning(rig, mr, pm, images, cfg, center_um, closed_loop, rec, progress)
    note("mapping the finished piece")
    out["finished"] = map_finished_piece(rig, cfg, pm.area_size_um, center_um, rec, pm)
    rec.log(
        "tile spacing error vs ideal (um): " + ", ".join(f"{v:.2f}" for v in out["patterning"]["spacing_error_um"])
        + f"  ({'closed-loop' if closed_loop else 'open-loop'})"
    )
    return out


if __name__ == "__main__":
    import sys

    res = run_virtual_session(label="cli")
    print("run folder:", res["recorder"].dir)
    print("\n".join(res["map"].report()))
    if "patterning" in res:
        print("tile spacing error (um):", [round(v, 2) for v in res["patterning"]["spacing_error_um"]])
        print("\n".join(res["finished"].report()))

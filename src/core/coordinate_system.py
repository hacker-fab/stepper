from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def apply_homography_2d(
    points: Union[np.ndarray, Sequence[Tuple[float, float]]],
    matrix_3x3: np.ndarray,
) -> np.ndarray:
    """Applies a 3x3 homogeneous transformation matrix to 2D point(s).

    Args:
        points: Shape (N, 2) or (2,).
        matrix_3x3: Shape (3, 3) float64 homogeneous matrix.

    Returns:
        Transformed points with the same shape as input (N, 2) or (2,).
    """
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 2)

    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    out_h = (matrix_3x3 @ pts_h.T).T
    w = out_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = out_h[:, :2] / w
    return out[0] if single else out


def compute_two_point_rigid_transform(
    src_p1: Tuple[float, float],
    src_p2: Tuple[float, float],
    dst_p1: Tuple[float, float],
    dst_p2: Tuple[float, float],
) -> np.ndarray:
    """Computes a 3x3 2D rigid transformation matrix (rotation + translation)
    mapping (src_p1, src_p2) in corr-stage-space to (dst_p1, dst_p2) in chip-space.

    Aligns the vector direction (p2 - p1) and maps the midpoint of (src_p1, src_p2)
    to the midpoint of (dst_p1, dst_p2).
    """
    s1 = np.array(src_p1, dtype=np.float64)
    s2 = np.array(src_p2, dtype=np.float64)
    d1 = np.array(dst_p1, dtype=np.float64)
    d2 = np.array(dst_p2, dtype=np.float64)

    ds = s2 - s1
    dd = d2 - d1
    len_s = float(np.hypot(ds[0], ds[1]))
    len_d = float(np.hypot(dd[0], dd[1]))

    if len_s < 1e-6 or len_d < 1e-6:
        # Degenerate markers: pure translation from midpoint
        R = np.eye(2, dtype=np.float64)
    else:
        angle_s = float(np.arctan2(ds[1], ds[0]))
        angle_d = float(np.arctan2(dd[1], dd[0]))
        dtheta = angle_d - angle_s
        c = float(np.cos(dtheta))
        s = float(np.sin(dtheta))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)

    mid_s = 0.5 * (s1 + s2)
    mid_d = 0.5 * (d1 + d2)
    t = mid_d - (R @ mid_s)

    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def build_camera_to_corr_stage_matrix(
    stage_xy_corr: Tuple[float, float],
    camera_size: Tuple[int, int],
    camera_pixel_size: float,
) -> np.ndarray:
    """Builds the 3x3 matrix mapping camera pixel (u, v) to corr-stage-space (x_corr, y_corr) in µm.

    Camera center (cam_w / 2, cam_h / 2) corresponds to stage_xy_corr.
    x_corr = stage_x + (u - cam_w / 2) * camera_pixel_size
    y_corr = stage_y + (v - cam_h / 2) * camera_pixel_size
    """
    cam_w, cam_h = float(camera_size[0]), float(camera_size[1])
    s = float(camera_pixel_size)
    sx, sy = float(stage_xy_corr[0]), float(stage_xy_corr[1])

    return np.array(
        [
            [s, 0.0, sx - 0.5 * cam_w * s],
            [0.0, s, sy - 0.5 * cam_h * s],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_corr_stage_to_camera_matrix(
    stage_xy_corr: Tuple[float, float],
    camera_size: Tuple[int, int],
    camera_pixel_size: float,
) -> np.ndarray:
    """Builds the 3x3 matrix mapping corr-stage-space (x_corr, y_corr) in µm to camera pixel (u, v)."""
    M = build_camera_to_corr_stage_matrix(stage_xy_corr, camera_size, camera_pixel_size)
    return np.linalg.inv(M)


def get_proj_to_cam_homography(
    proj_to_cam_matrix: Optional[Sequence[Sequence[float]] | np.ndarray],
    projector_size: Tuple[int, int],
    camera_size: Tuple[int, int],
    projector_pixel_size: float = 1.0,
    camera_pixel_size: float = 1.0,
) -> np.ndarray:
    """Returns the 3x3 homography mapping projector pixels (px, py) to camera pixels (u, v).

    If proj_to_cam_matrix is calibrated and valid, returns it.
    Otherwise constructs a centered scaling matrix using projector_pixel_size / camera_pixel_size.
    """
    if proj_to_cam_matrix is not None:
        arr = np.asarray(proj_to_cam_matrix, dtype=np.float64)
        if arr.shape == (3, 3) and abs(np.linalg.det(arr)) > 1e-12:
            return arr

    pw, ph = float(projector_size[0]), float(projector_size[1])
    cw, ch = float(camera_size[0]), float(camera_size[1])
    scale = float(projector_pixel_size) / max(float(camera_pixel_size), 1e-9)

    return np.array(
        [
            [scale, 0.0, 0.5 * cw - 0.5 * pw * scale],
            [0.0, scale, 0.5 * ch - 0.5 * ph * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_projector_to_chip_matrix(
    stage_xy_corr: Tuple[float, float],
    projector_size: Tuple[int, int],
    camera_size: Tuple[int, int],
    projector_pixel_size: float,
    camera_pixel_size: float,
    proj_to_cam_matrix: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    corr_to_chip_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Builds the composite 3x3 matrix mapping projector pixel (px, py) -> chip-space (x_chip, y_chip) in µm
    when the stage is positioned at stage_xy_corr in corr-stage-space.
    """
    H_proj_to_cam = get_proj_to_cam_homography(
        proj_to_cam_matrix=proj_to_cam_matrix,
        projector_size=projector_size,
        camera_size=camera_size,
        projector_pixel_size=projector_pixel_size,
        camera_pixel_size=camera_pixel_size,
    )
    M_cam_to_corr = build_camera_to_corr_stage_matrix(
        stage_xy_corr=stage_xy_corr,
        camera_size=camera_size,
        camera_pixel_size=camera_pixel_size,
    )
    T_corr_to_chip = (
        np.asarray(corr_to_chip_matrix, dtype=np.float64)
        if corr_to_chip_matrix is not None
        else np.eye(3, dtype=np.float64)
    )
    return T_corr_to_chip @ M_cam_to_corr @ H_proj_to_cam


def compute_fullscreen_pattern_to_chip_matrix(
    pattern_size_um: Tuple[float, float],
    projector_size: Tuple[int, int] = (1920, 1080),
    camera_size: Tuple[int, int] = (1920, 1080),
    projector_pixel_size: float = 1.0,
    camera_pixel_size: float = 1.0,
    proj_to_cam_matrix: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    corr_to_chip_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes the 3x3 matrix mapping local pattern coordinates [-w/2..w/2, -h/2..h/2] (µm)
    to local chip-space offsets (dx_chip, dy_chip) (µm) around the origin when exposed
    as a single fullscreen tile on the projector.
    """
    pw, ph = float(projector_size[0]), float(projector_size[1])
    w_um = max(1e-6, float(pattern_size_um[0]))
    h_um = max(1e-6, float(pattern_size_um[1]))

    M_proj_to_chip_raw = build_projector_to_chip_matrix(
        stage_xy_corr=(0.0, 0.0),
        projector_size=(int(pw), int(ph)),
        camera_size=camera_size,
        projector_pixel_size=projector_pixel_size,
        camera_pixel_size=camera_pixel_size,
        proj_to_cam_matrix=proj_to_cam_matrix,
        corr_to_chip_matrix=corr_to_chip_matrix,
    )
    proj_center_chip = apply_homography_2d((0.5 * pw, 0.5 * ph), M_proj_to_chip_raw)
    T_shift_to_origin = np.array(
        [
            [1.0, 0.0, -float(proj_center_chip[0])],
            [0.0, 1.0, -float(proj_center_chip[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    M_proj_to_chip_0 = T_shift_to_origin @ M_proj_to_chip_raw

    M_pat_to_proj = np.array(
        [
            [pw / w_um, 0.0, 0.5 * pw],
            [0.0, ph / h_um, 0.5 * ph],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return M_proj_to_chip_0 @ M_pat_to_proj


def compute_projector_footprint_in_chip_space(
    stage_xy_corr: Tuple[float, float],
    projector_size: Tuple[int, int],
    camera_size: Tuple[int, int],
    projector_pixel_size: float,
    camera_pixel_size: float,
    proj_to_cam_matrix: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    corr_to_chip_matrix: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    """Returns the 4 corners of the projector frame in chip-space (µm) when stage is at stage_xy_corr."""
    pw, ph = float(projector_size[0]), float(projector_size[1])
    proj_corners = np.array(
        [
            [0.0, 0.0],
            [pw, 0.0],
            [pw, ph],
            [0.0, ph],
        ],
        dtype=np.float64,
    )
    M_proj_to_chip = build_projector_to_chip_matrix(
        stage_xy_corr=stage_xy_corr,
        projector_size=projector_size,
        camera_size=camera_size,
        projector_pixel_size=projector_pixel_size,
        camera_pixel_size=camera_pixel_size,
        proj_to_cam_matrix=proj_to_cam_matrix,
        corr_to_chip_matrix=corr_to_chip_matrix,
    )
    chip_corners = apply_homography_2d(proj_corners, M_proj_to_chip)
    return [(float(pt[0]), float(pt[1])) for pt in chip_corners]


def compute_projector_pixel_size_from_tags(
    detected_corners: Sequence[np.ndarray],
    detected_ids: Optional[np.ndarray],
    known_tag_coords: Dict[int, Dict[str, Any]],
    camera_pixel_size: float,
) -> Optional[float]:
    """Calculates projector pixel size (µm/px) from a detected 2x2 ArUco grid.

    Uses the known camera_pixel_size (µm/cam_px), the known ArUco tag/grid separation
    in projector pixels, and the measured separation in camera pixels.
    """
    if detected_ids is None or len(detected_corners) == 0:
        return None

    proj_pts: List[np.ndarray] = []
    cam_pts: List[np.ndarray] = []

    flat_ids = detected_ids.flatten()
    for idx, marker_id in enumerate(flat_ids):
        mid = int(marker_id)
        if mid not in known_tag_coords:
            continue
        p_corners = np.asarray(known_tag_coords[mid]["corners"], dtype=np.float64).reshape(-1, 2)
        c_corners = np.asarray(detected_corners[idx], dtype=np.float64).reshape(-1, 2)
        for p_pt, c_pt in zip(p_corners, c_corners):
            proj_pts.append(p_pt)
            cam_pts.append(c_pt)

    n = len(proj_pts)
    if n < 2:
        return None

    ratios: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_proj = float(np.linalg.norm(proj_pts[j] - proj_pts[i]))
            dist_cam = float(np.linalg.norm(cam_pts[j] - cam_pts[i]))
            if dist_proj > 10.0 and dist_cam > 1.0:
                ratios.append(dist_cam / dist_proj)

    if not ratios:
        return None

    cam_px_per_proj_px = float(np.median(ratios))
    return cam_px_per_proj_px * float(camera_pixel_size)


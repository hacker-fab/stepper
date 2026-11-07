from typing import Dict, List, Optional, Tuple, Set
import numpy as np


def estimate_chip_rotation(
    corners: List[np.ndarray],
    ids: np.ndarray,
    nominal_dict: Dict[int, Dict],
) -> Optional[float]:
    """Estimates the wafer/chip rotation angle theta (radians) from detected tag corners."""
    angles = []
    for idx_det, mid_val in enumerate(ids.flatten()):
        mid_int = int(mid_val)
        if mid_int in nominal_dict:
            tag_corners = corners[idx_det].reshape(-1, 2)
            edge_top = tag_corners[1] - tag_corners[0]
            angles.append(np.arctan2(edge_top[1], edge_top[0]))

    if not angles:
        return None
    return float(np.median(angles))


def extract_pairwise_measurements(
    corners: List[np.ndarray],
    ids: np.ndarray,
    nominal_dict: Dict[int, Dict],
    pixel_scale: float,
    R_inv: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, float, float]]:
    """Extracts all relative pairwise distance measurements (id_a, id_b, dx_wafer, dy_wafer)

    from detected tags in a single camera frame by projecting displacements onto the
    intrinsic physical basis vectors (top-edge and left-edge) of the detected tags.
    """
    centers_in_frame: Dict[int, np.ndarray] = {}
    u_x_list: List[np.ndarray] = []
    u_y_list: List[np.ndarray] = []

    for idx_m, mid_val in enumerate(ids.flatten()):
        m_int = int(mid_val)
        if m_int in nominal_dict:
            pts = corners[idx_m].reshape(-1, 2)
            centers_in_frame[m_int] = np.mean(pts, axis=0)

            # Tag physical axes in image coordinates:
            # pts[0] -> Top-Left, pts[1] -> Top-Right, pts[3] -> Bottom-Left
            vx = pts[1] - pts[0]
            vy = pts[3] - pts[0]
            lx = float(np.linalg.norm(vx))
            ly = float(np.linalg.norm(vy))
            if lx > 1e-6:
                u_x_list.append(vx / lx)
            if ly > 1e-6:
                u_y_list.append(vy / ly)

    if u_x_list and u_y_list:
        mean_ux = np.mean(u_x_list, axis=0)
        mean_uy = np.mean(u_y_list, axis=0)
        u_x_frame = mean_ux / max(float(np.linalg.norm(mean_ux)), 1e-6)
        u_y_frame = mean_uy / max(float(np.linalg.norm(mean_uy)), 1e-6)
    elif R_inv is not None:
        u_x_frame = R_inv[0, :]
        u_y_frame = R_inv[1, :]
    else:
        u_x_frame = np.array([1.0, 0.0], dtype=np.float64)
        u_y_frame = np.array([0.0, 1.0], dtype=np.float64)

    measurements: List[Tuple[int, int, float, float]] = []
    marker_ids_found = sorted(list(centers_in_frame.keys()))
    for i in range(len(marker_ids_found)):
        for j in range(i + 1, len(marker_ids_found)):
            id_a = marker_ids_found[i]
            id_b = marker_ids_found[j]
            disp_cam_px = centers_in_frame[id_b] - centers_in_frame[id_a]
            disp_cam_um = disp_cam_px * pixel_scale
            dx = float(np.dot(disp_cam_um, u_x_frame))
            dy = float(np.dot(disp_cam_um, u_y_frame))
            measurements.append((id_a, id_b, dx, dy))

    return measurements


def estimate_camera_to_stage_transform(
    multi_frame_observations: List[Tuple[int, Tuple[float, float], Tuple[float, float]]],
    default_rotation: float = 0.0,
) -> np.ndarray:
    """Estimates the 2x2 linear transformation matrix mapping camera displacements (um) to stage displacements (um).

    Args:
        multi_frame_observations: List of (marker_id, stage_xy, cam_offset_um) across all scanned images.
        default_rotation: Fallback rotation theta (radians) if multi-image overlap is insufficient.

    Returns:
        2x2 transformation matrix T_cam_to_stage.
    """
    # Group observations by marker_id
    by_marker: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    for mid, s_xy, c_off in multi_frame_observations:
        by_marker.setdefault(mid, []).append((s_xy, c_off))

    A_rows: List[np.ndarray] = []
    b_x: List[float] = []
    b_y: List[float] = []

    for mid, obs_list in by_marker.items():
        if len(obs_list) < 2:
            continue
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                s1, c1 = obs_list[i]
                s2, c2 = obs_list[j]
                ds = (s2[0] - s1[0], s2[1] - s1[1])
                dc = (c2[0] - c1[0], c2[1] - c1[1])
                # Only use if stage moved and camera offset changed
                if (ds[0] ** 2 + ds[1] ** 2) > 1e-3 and (dc[0] ** 2 + dc[1] ** 2) > 1e-3:
                    A_rows.append(np.array([dc[0], dc[1]], dtype=np.float64))
                    b_x.append(-ds[0])
                    b_y.append(-ds[1])

    # Constrain T_cam_to_stage to orthogonal axes (zero skew) while allowing independent
    # axis scales (s_x, s_y) for X and Y stage motor axes.
    # This prevents T from absorbing mechanical stage skew while preserving equal movement distance.
    if len(A_rows) >= 2:
        DC = np.array(A_rows, dtype=np.float64)  # (N, 2) camera displacements
        V = np.column_stack([b_x, b_y])         # (N, 2) -ds stage displacements
        H = V.T @ DC                             # (2, 2) covariance matrix
        if np.linalg.matrix_rank(H) >= 2:
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            r0 = R[0, :]
            r1 = R[1, :]

            denom_x = float(np.sum((DC @ r0) ** 2))
            denom_y = float(np.sum((DC @ r1) ** 2))
            if denom_x > 1e-9 and denom_y > 1e-9:
                s_x = float(np.sum(V[:, 0] * (DC @ r0)) / denom_x)
                s_y = float(np.sum(V[:, 1] * (DC @ r1)) / denom_y)
                if s_x < 0:
                    s_x = -s_x
                    r0 = -r0
                if s_y < 0:
                    s_y = -s_y
                    r1 = -r1
                if s_x > 1e-4 and s_y > 1e-4:
                    return np.array([s_x * r0, s_y * r1], dtype=np.float64)

    # Fallback to default rotation in un-mirrored camera frame
    cos_t = np.cos(default_rotation)
    sin_t = np.sin(default_rotation)
    R_rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return R_rot


def solve_detected_stage_positions(
    detected_ids: Set[int],
    observed_positions: Dict[int, Tuple[float, float]],
    pairwise_measurements: Optional[List[Tuple[int, int, float, float]]] = None,
    anchor_id: int = 0,
    nominal_positions: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Solves stage-space positions of detected markers directly using direct stage observations
    and pairwise stage measurements between detected markers, WITHOUT adding any generated
    neighbor constraints.

    Args:
        detected_ids: Set of marker IDs detected across images.
        observed_positions: Dict mapping marker_id -> observed stage position (mean of s_k + T c_{k, m}).
        pairwise_measurements: List of (id_a, id_b, dx_stage, dy_stage) in stage space.
        anchor_id: Marker ID used as translation gauge anchor if no direct stage observations exist.
        nominal_positions: Optional nominal positions for gauge anchoring.

    Returns:
        Dict mapping marker_id -> solved stage position (X, Y) for detected markers.
    """
    if not detected_ids:
        return {}

    markers = sorted(list(detected_ids))
    M = len(markers)
    if M == 1:
        m0 = markers[0]
        if m0 in observed_positions:
            return {m0: observed_positions[m0]}
        if nominal_positions and m0 in nominal_positions:
            return {m0: nominal_positions[m0]}
        return {m0: (0.0, 0.0)}

    m_to_idx = {m: i for i, m in enumerate(markers)}

    # Filter pairwise measurements to those where both markers are in detected_ids
    valid_meas: List[Tuple[int, int, float, float]] = []
    if pairwise_measurements:
        for id_a, id_b, dx, dy in pairwise_measurements:
            if id_a in m_to_idx and id_b in m_to_idx:
                # If both have direct observed positions, check if measurement is in the same coordinate frame
                if id_a in observed_positions and id_b in observed_positions:
                    obs_a = observed_positions[id_a]
                    obs_b = observed_positions[id_b]
                    obs_disp = float(np.hypot(obs_b[0] - obs_a[0], obs_b[1] - obs_a[1]))
                    meas_disp = float(np.hypot(dx, dy))
                    # Length discrepancy check
                    if abs(obs_disp - meas_disp) > max(100.0, 0.2 * max(obs_disp, 1.0)):
                        continue
                    # Rotation/coordinate discrepancy check: if measurement is rotated relative to observed, skip
                    diff_x = (obs_b[0] - obs_a[0]) - dx
                    diff_y = (obs_b[1] - obs_a[1]) - dy
                    if float(np.hypot(diff_x, diff_y)) > max(50.0, 0.2 * max(obs_disp, 1.0)):
                        continue
                valid_meas.append((id_a, id_b, float(dx), float(dy)))

    # Direct observations count
    obs_count = sum(1 for m in markers if m in observed_positions)

    # Build linear system A x = bx, A y = by
    rows = []
    bx_list = []
    by_list = []

    for m in markers:
        if m in observed_positions:
            row = np.zeros(M, dtype=np.float64)
            row[m_to_idx[m]] = 1.0
            rows.append(row)
            bx_list.append(observed_positions[m][0])
            by_list.append(observed_positions[m][1])

    for id_a, id_b, dx, dy in valid_meas:
        row = np.zeros(M, dtype=np.float64)
        row[m_to_idx[id_a]] = -1.0
        row[m_to_idx[id_b]] = 1.0
        rows.append(row)
        bx_list.append(dx)
        by_list.append(dy)

    if obs_count == 0:
        ref_id = anchor_id if anchor_id in m_to_idx else markers[0]
        row = np.zeros(M, dtype=np.float64)
        row[m_to_idx[ref_id]] = 1.0
        rows.append(row)
        ref_pos = nominal_positions.get(ref_id, (0.0, 0.0)) if nominal_positions else (0.0, 0.0)
        bx_list.append(ref_pos[0])
        by_list.append(ref_pos[1])

    if not rows:
        return {m: nominal_positions.get(m, (0.0, 0.0)) if nominal_positions else (0.0, 0.0) for m in markers}

    A = np.array(rows, dtype=np.float64)
    bx = np.array(bx_list, dtype=np.float64)
    by = np.array(by_list, dtype=np.float64)

    sol_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    result = {}
    for m in markers:
        idx = m_to_idx[m]
        result[m] = (float(sol_x[idx]), float(sol_y[idx]))

    return result


def match_global_transform_and_assign_undetected(
    detected_positions: Dict[int, Tuple[float, float]],
    nominal_positions: Dict[int, Tuple[float, float]],
    default_rotation: float = 0.0,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[int, Tuple[float, float]], np.ndarray, np.ndarray]:
    """Matches a global rotation & translation aligning nominal coordinates to calibrated
    detected tags, and assigns not-detected tags using opposite-side neighbor averaging
    if available (+-x or +-y or both), falling back to the fitted rigid transform.

    Args:
        detected_positions: Reconstructed positions of detected markers in stage coordinates.
        nominal_positions: Nominal coordinates of all grid markers.
        default_rotation: Fallback rotation in radians if < 2 markers are detected.
        grid_shape: Optional (ny, nx) grid dimensions.

    Returns:
        (full_stage_positions, R, t)
    """
    common_ids = [m for m in nominal_positions if m in detected_positions]

    if len(common_ids) >= 2:
        src = np.array([nominal_positions[m] for m in common_ids], dtype=np.float64)
        dst = np.array([detected_positions[m] for m in common_ids], dtype=np.float64)

        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_mean - R @ src_mean

    elif len(common_ids) == 1:
        m0 = common_ids[0]
        cos_t, sin_t = np.cos(default_rotation), np.sin(default_rotation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        t = np.array(detected_positions[m0], dtype=np.float64) - R @ np.array(nominal_positions[m0], dtype=np.float64)
    else:
        cos_t, sin_t = np.cos(default_rotation), np.sin(default_rotation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        t = np.zeros(2, dtype=np.float64)

    # Build grid mapping for neighbor lookup
    id_to_grid: Dict[int, Tuple[int, int]] = {}
    grid_to_id: Dict[Tuple[int, int], int] = {}

    if grid_shape is not None:
        ny, nx = grid_shape
        for r in range(ny):
            for c in range(nx):
                mid = r * nx + c
                id_to_grid[mid] = (r, c)
                grid_to_id[(r, c)] = mid
    elif nominal_positions:
        unique_xs = sorted(list({round(float(p[0]), 2) for p in nominal_positions.values()}))
        unique_ys = sorted(list({round(float(p[1]), 2) for p in nominal_positions.values()}))
        if len(unique_xs) * len(unique_ys) == len(nominal_positions) and len(unique_xs) > 0 and len(unique_ys) > 0:
            x_to_c = {x: c for c, x in enumerate(unique_xs)}
            y_to_r = {y: r for r, y in enumerate(unique_ys)}
            for mid, pos in nominal_positions.items():
                r = y_to_r[round(float(pos[1]), 2)]
                c = x_to_c[round(float(pos[0]), 2)]
                id_to_grid[mid] = (r, c)
                grid_to_id[(r, c)] = mid

    full_positions: Dict[int, Tuple[float, float]] = {}
    for m, p_nom in nominal_positions.items():
        if m in detected_positions:
            full_positions[m] = detected_positions[m]
        else:
            # Check if opposite-side neighbors are detected (+-x or +-y or both)
            assigned = False
            if m in id_to_grid:
                r, c = id_to_grid[m]
                left_id = grid_to_id.get((r, c - 1))
                right_id = grid_to_id.get((r, c + 1))
                up_id = grid_to_id.get((r - 1, c))
                down_id = grid_to_id.get((r + 1, c))

                has_x = (
                    left_id is not None and left_id in detected_positions and
                    right_id is not None and right_id in detected_positions
                )
                has_y = (
                    up_id is not None and up_id in detected_positions and
                    down_id is not None and down_id in detected_positions
                )

                if has_x and has_y:
                    mid_x = 0.5 * (np.array(detected_positions[left_id], dtype=np.float64) + np.array(detected_positions[right_id], dtype=np.float64))
                    mid_y = 0.5 * (np.array(detected_positions[up_id], dtype=np.float64) + np.array(detected_positions[down_id], dtype=np.float64))
                    p_est = 0.5 * (mid_x + mid_y)
                    full_positions[m] = (float(p_est[0]), float(p_est[1]))
                    assigned = True
                elif has_x:
                    mid_x = 0.5 * (np.array(detected_positions[left_id], dtype=np.float64) + np.array(detected_positions[right_id], dtype=np.float64))
                    full_positions[m] = (float(mid_x[0]), float(mid_x[1]))
                    assigned = True
                elif has_y:
                    mid_y = 0.5 * (np.array(detected_positions[up_id], dtype=np.float64) + np.array(detected_positions[down_id], dtype=np.float64))
                    full_positions[m] = (float(mid_y[0]), float(mid_y[1]))
                    assigned = True

            if not assigned:
                p_pred = R @ np.array(p_nom, dtype=np.float64) + t
                full_positions[m] = (float(p_pred[0]), float(p_pred[1]))

    return full_positions, R, t


def solve_grid_relative_positions(
    grid_shape: Tuple[int, int],
    measurements: List[Tuple[int, int, float, float]],
    step_distance: float,
    nominal_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    anchor_id: int = 0,
    observed_marker_ids: Optional[Set[int]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Step 1: Uses relative position between each anchor to estimate the position of the full grid.

    If an anchor is not detected, adds default constraints that its relative distance to neighbor
    anchors matches the grid size (step_distance).

    This step solves the relative position of the entire grid, but does not match the grid to
    the actual stage/camera position.

    Args:
        grid_shape: (ny, nx) for the grid. Total markers M = ny * nx.
        measurements: List of (id_A, id_B, dx, dy) where x_B - x_A = dx and y_B - y_A = dy.
        step_distance: Nominal distance between neighbor nodes in the grid (um).
        nominal_positions: Optional dict mapping marker_id -> nominal (X, Y).
        anchor_id: Marker ID to anchor at (0, 0) to fix translation gauge freedom.
        observed_marker_ids: Set of marker IDs detected in camera frames. If None, derived from measurements.

    Returns:
        Dict mapping marker_id -> relative (X, Y) coordinate in the unaligned grid frame.
    """
    ny, nx = grid_shape
    total_markers = ny * nx
    if total_markers <= 0:
        return {}

    # Determine which markers have pairwise measurement connections
    meas_connected: Set[int] = set()
    for id_a, id_b, _, _ in measurements:
        if 0 <= id_a < total_markers and 0 <= id_b < total_markers:
            meas_connected.add(id_a)
            meas_connected.add(id_b)

    detected_set: Set[int] = observed_marker_ids if observed_marker_ids is not None else meas_connected

    # Build constraint list: (id_a, id_b, dx, dy, weight)
    constraints: List[Tuple[int, int, float, float, float]] = []

    # 1. Add measured pairwise relative positions with full weight
    for id_a, id_b, dx, dy in measurements:
        if 0 <= id_a < total_markers and 0 <= id_b < total_markers:
            constraints.append((id_a, id_b, float(dx), float(dy), 1.0))

    # 2. For anchors that are not detected (or not connected by pairwise measurements),
    # add default constraints connecting them to their adjacent neighbors in the grid.
    # We use a soft prior weight (1e-3) so default constraints do not distort actual optical measurements.
    added_neighbor_pairs: Set[Tuple[int, int]] = set()
    default_prior_weight = 1e-3 if measurements else 1.0

    for r in range(ny):
        for c in range(nx):
            m_id = r * nx + c
            # If this anchor is not detected, or has no relative measurements connecting it
            is_unconstrained = (m_id not in detected_set) or (m_id not in meas_connected)

            # Define neighbors: (neighbor_id, nominal_dx, nominal_dy)
            neighbors: List[Tuple[int, float, float]] = []
            if c + 1 < nx:
                n_id_r = r * nx + (c + 1)
                if nominal_positions and m_id in nominal_positions and n_id_r in nominal_positions:
                    nom_dx = float(nominal_positions[n_id_r][0] - nominal_positions[m_id][0])
                    nom_dy = float(nominal_positions[n_id_r][1] - nominal_positions[m_id][1])
                else:
                    nom_dx, nom_dy = float(step_distance), 0.0
                neighbors.append((n_id_r, nom_dx, nom_dy))

            if r + 1 < ny:
                n_id_d = (r + 1) * nx + c
                if nominal_positions and m_id in nominal_positions and n_id_d in nominal_positions:
                    nom_dx = float(nominal_positions[n_id_d][0] - nominal_positions[m_id][0])
                    nom_dy = float(nominal_positions[n_id_d][1] - nominal_positions[m_id][1])
                else:
                    nom_dx, nom_dy = 0.0, float(step_distance)
                neighbors.append((n_id_d, nom_dx, nom_dy))

            for n_id, nom_dx, nom_dy in neighbors:
                pair_key = (min(m_id, n_id), max(m_id, n_id))
                # Add default constraint if either anchor is unconstrained
                n_unconstrained = (n_id not in detected_set) or (n_id not in meas_connected)
                if (is_unconstrained or n_unconstrained) and pair_key not in added_neighbor_pairs:
                    added_neighbor_pairs.add(pair_key)
                    # Constraint weight: default constraints ensure the grid geometry is preserved
                    constraints.append((m_id, n_id, nom_dx, nom_dy, default_prior_weight))

    # If measurements are completely empty, add default grid constraints for the entire grid
    if not constraints:
        for r in range(ny):
            for c in range(nx):
                m_id = r * nx + c
                if c + 1 < nx:
                    n_id_r = r * nx + (c + 1)
                    if nominal_positions and m_id in nominal_positions and n_id_r in nominal_positions:
                        nom_dx = float(nominal_positions[n_id_r][0] - nominal_positions[m_id][0])
                        nom_dy = float(nominal_positions[n_id_r][1] - nominal_positions[m_id][1])
                    else:
                        nom_dx, nom_dy = float(step_distance), 0.0
                    constraints.append((m_id, n_id_r, nom_dx, nom_dy, 1.0))
                if r + 1 < ny:
                    n_id_d = (r + 1) * nx + c
                    if nominal_positions and m_id in nominal_positions and n_id_d in nominal_positions:
                        nom_dx = float(nominal_positions[n_id_d][0] - nominal_positions[m_id][0])
                        nom_dy = float(nominal_positions[n_id_d][1] - nominal_positions[m_id][1])
                    else:
                        nom_dx, nom_dy = 0.0, float(step_distance)
                    constraints.append((m_id, n_id_d, nom_dx, nom_dy, 1.0))

    K = len(constraints)
    A = np.zeros((K + 1, total_markers), dtype=np.float64)
    bx = np.zeros(K + 1, dtype=np.float64)
    by = np.zeros(K + 1, dtype=np.float64)

    for k, (id_a, id_b, dx, dy, w) in enumerate(constraints):
        A[k, id_a] = -w
        A[k, id_b] = w
        bx[k] = dx * w
        by[k] = dy * w

    # Fix gauge freedom by anchoring anchor_id to (0, 0)
    ref_id = anchor_id if (0 <= anchor_id < total_markers) else 0
    A[K, ref_id] = 1.0
    bx[K] = 0.0
    by[K] = 0.0

    sol_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    results: Dict[int, Tuple[float, float]] = {}
    for m in range(total_markers):
        results[m] = (float(sol_x[m]), float(sol_y[m]))

    return results


def estimate_global_transform(
    relative_positions: Dict[int, Tuple[float, float]],
    observed_positions: Dict[int, Tuple[float, float]],
    default_rotation: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2: Estimates a global transform matrix such that the grid matches what the camera sees.

    Finds 2x2 rotation R and 2x1 translation t aligning relative_positions to observed_positions
    using 2D Kabsch / Umeyama rigid alignment.

    Args:
        relative_positions: Marker positions in the unaligned grid frame from Step 1.
        observed_positions: Positions seen by the camera in stage coordinates.
        default_rotation: Fallback rotation angle (radians) if fewer than 2 markers are observed.

    Returns:
        (R, t, T_3x3) where R is 2x2 rotation matrix, t is 2D translation vector, and T_3x3 is homogeneous.
    """
    common_ids = [m for m in relative_positions if m in observed_positions]

    if len(common_ids) >= 2:
        src = np.array([relative_positions[m] for m in common_ids], dtype=np.float64)
        dst = np.array([observed_positions[m] for m in common_ids], dtype=np.float64)

        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_mean - R @ src_mean

    elif len(common_ids) == 1:
        m0 = common_ids[0]
        cos_t, sin_t = np.cos(default_rotation), np.sin(default_rotation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        t = np.array(observed_positions[m0], dtype=np.float64) - R @ np.array(relative_positions[m0], dtype=np.float64)
    else:
        cos_t, sin_t = np.cos(default_rotation), np.sin(default_rotation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        t = np.zeros(2, dtype=np.float64)

    T_3x3 = np.eye(3, dtype=np.float64)
    T_3x3[:2, :2] = R
    T_3x3[:2, 2] = t

    return R, t, T_3x3


def align_calibrated_origin_to_stage_origin(
    relative_positions: Dict[int, Tuple[float, float]],
    R: np.ndarray,
    t: Optional[np.ndarray] = None,
    nominal_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    raw_grid_center: Optional[Tuple[float, float]] = None,
    origin_marker_id: int = 0,
) -> Tuple[Dict[int, Tuple[float, float]], np.ndarray]:
    """Step 3: Un-rotates wafer mounting rotation R, and matches calibrated location
    of grid center to raw grid center.

    R represents the rigid rotation of the physical wafer mounted on the stage relative to
    nominal stage axes. To eliminate the arbitrary mounting angle and align the calibrated stage
    coordinates with the nominal stage axes (0 deg):
      p_unrot = R.T @ (p - c_cal) + c_raw

    This ensures that:
    1. Grid rows are parallel to the stage motor X axis (0 deg).
    2. Any stage distortion, scale error, or rail non-orthogonality is preserved.
    3. Commanded stage motion along stage axes will not suffer from wafer mounting tilt.

    Args:
        relative_positions: Relative or stage positions for grid markers.
        R: 2x2 rotation matrix from Step 2 representing chip rotation on stage.
        t: Optional 2D translation vector from Step 2. If provided, applies (R @ p + t) before alignment.
        nominal_positions: Optional dict of nominal/raw positions to compute the raw grid center.
        raw_grid_center: Optional explicit (X, Y) coordinate of the raw grid center.
        origin_marker_id: Kept for backwards compatibility.

    Returns:
        (calibrated_positions, chip_to_stage_matrix)
    """
    # 1. Determine raw grid center C_raw
    if raw_grid_center is not None:
        c_raw = np.array(raw_grid_center, dtype=np.float64)
    elif nominal_positions:
        c_raw = np.mean(
            np.array([nominal_positions[m] for m in nominal_positions], dtype=np.float64),
            axis=0,
        )
    else:
        c_raw = np.mean(
            np.array([relative_positions[m] for m in relative_positions], dtype=np.float64),
            axis=0,
        )

    # 2. Determine calibrated/stage positions before center shift
    if t is not None:
        cal_stage: Dict[int, np.ndarray] = {}
        for m, p_rel in relative_positions.items():
            cal_stage[m] = R @ np.array(p_rel, dtype=np.float64) + t
    else:
        cal_stage = {m: np.array(p, dtype=np.float64) for m, p in relative_positions.items()}

    # 3. Determine calibrated location of grid center C_cal
    c_cal = np.mean(
        np.array([cal_stage[m] for m in cal_stage], dtype=np.float64),
        axis=0,
    )

    # 4. Un-rotate wafer mounting angle R around grid center and shift to match raw grid center
    R_inv = R.T
    calibrated_positions: Dict[int, Tuple[float, float]] = {}
    for m, p in cal_stage.items():
        diff = p - c_cal
        p_unrot = R_inv @ diff + c_raw
        calibrated_positions[m] = (float(p_unrot[0]), float(p_unrot[1]))

    cam_to_stage_matrix = R.copy()
    return calibrated_positions, cam_to_stage_matrix


def solve_global_positions(
    grid_shape: Tuple[int, int],
    measurements: List[Tuple[int, int, float, float]],
    anchor_id: int,
    anchor_pos: Tuple[float, float],
    nominal_positions: Dict[int, Tuple[float, float]],
    observed_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    step_distance: Optional[float] = None,
    default_rotation: float = 0.0,
) -> Tuple[Dict[int, Tuple[float, float]], np.ndarray]:
    """Executes the full calibration solver logic.

    Step 1: Solves detected marker positions in stage space directly (without adding generated constraints).
    Step 2: Fits global rotation and translation from nominal positions to detected stage positions,
            and assigns undetected markers using the fitted rigid transform.
    Step 3: Matches calibrated location of grid center to raw grid center.

    Returns:
        (calibrated_positions, cam_to_stage_matrix)
    """
    ny, nx = grid_shape

    # Determine step distance if not explicitly provided
    if step_distance is None or step_distance <= 0:
        if nx > 1 and 0 in nominal_positions and 1 in nominal_positions:
            step_distance = float(nominal_positions[1][0] - nominal_positions[0][0])
        elif ny > 1 and 0 in nominal_positions and nx in nominal_positions:
            step_distance = float(nominal_positions[nx][1] - nominal_positions[0][1])
        else:
            step_distance = 1000.0

    # Ensure nominal_positions covers all grid markers
    full_nominal = dict(nominal_positions)
    for r in range(ny):
        for c in range(nx):
            mid = r * nx + c
            if mid not in full_nominal:
                full_nominal[mid] = (c * step_distance, r * step_distance)

    # Observed marker IDs
    obs_ids: Set[int] = set()
    if observed_positions:
        obs_ids.update(observed_positions.keys())
    for id_a, id_b, _, _ in measurements:
        obs_ids.add(id_a)
        obs_ids.add(id_b)

    obs_pos = observed_positions if observed_positions is not None else {}

    # If no markers were observed at all, fall back to nominal grid (aligned to stage axes)
    if not obs_ids:
        cos_t, sin_t = np.cos(default_rotation), np.sin(default_rotation)
        R_rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        calibrated: Dict[int, Tuple[float, float]] = {m: (float(p[0]), float(p[1])) for m, p in full_nominal.items()}
        return calibrated, R_rot

    # 1. Step 1: Solve detected markers in stage space directly (NO generated constraints!)
    detected_stage_pos = solve_detected_stage_positions(
        detected_ids=obs_ids,
        observed_positions=obs_pos,
        pairwise_measurements=measurements,
        anchor_id=anchor_id,
        nominal_positions=full_nominal,
    )

    # 2. Step 2: Match global rotation & transform and assign undetected tags
    full_stage_pos, R, t = match_global_transform_and_assign_undetected(
        detected_positions=detected_stage_pos,
        nominal_positions=full_nominal,
        default_rotation=default_rotation,
        grid_shape=grid_shape,
    )

    # 3. Step 3: Match calibrated location of grid center to raw grid center
    cal_positions, cam_to_stage_mat = align_calibrated_origin_to_stage_origin(
        relative_positions=full_stage_pos,
        R=R,
        t=None,
        nominal_positions=full_nominal,
        origin_marker_id=anchor_id,
    )

    return cal_positions, cam_to_stage_mat

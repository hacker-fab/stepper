"""
EdgeAruco: Edge-Only Lithography ArUco Tag Detector
===================================================

A self-contained computer vision utility library designed to detect, validate,
decode, and subpixel-refine ArUco markers when surface black/white color fills
are lost and only boundary/etched step edges are visible (e.g., semiconductor
lithography on silicon wafers, etched quartz masks, laser marking, phase contrast).

Interface mimics OpenCV's `cv2.aruco.detectMarkers` / `cv2.aruco.ArucoDetector`.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


@dataclass
class EdgeDetectorParameters:
    """
    Parameters for Edge-Only ArUco Detection and Subpixel Refinement.

    Rules of thumb:
    ----------------
    1. Marker scale & cell size:
       - For an NxN dictionary (e.g. DICT_4X4 has 4 payload cells + 2 border cells = 6 cells wide).
       - If the marker side length in the image is S pixels:
         * cell_size ≈ S / (marker_dimension + 2)  [e.g., S / 6 for 4x4]
         * boundary stroke width w is typically 2 to 6 pixels in microscope optics.

    2. Adaptive threshold fine window (adaptive_thresh_fine_block_size):
       - RULE OF THUMB: Should be ~0.5x to 0.8x of cell_size (e.g. 11 to 17 px for 21 px cell).
       - Acts as a high-pass filter tuned to the 3-5 px etched stroke width, cleanly
         suppressing wide-period silicon wafer interference fringes.

    3. Adaptive threshold wide window (adaptive_thresh_wide_block_size):
       - RULE OF THUMB: Should be ~2.5x to 3.5x of cell_size (e.g. 51 to 65 px for 21 px cell).
       - Establishes the broad baseline background illumination across the marker,
         preventing bright phase-contrast halos around isolated 1x1 cells from
         producing false dark spots at cell centers.

    4. Grid-center edge threshold (max_center_edge_density):
       - Percentage of pixels in the inner 50% grid-center box allowed to be edges.
       - RULE OF THUMB: 0.10 to 0.18. Since true tag centers are clear, they typically
         have 0.00 to 0.05 density. Outer decorative frames or misaligned squares
         typically have 0.30 to 0.60 density.

    5. Neighbor edge decision threshold (neighbor_edge_thresh):
       - Mean edge response along cell shared boundaries to infer color change.
       - RULE OF THUMB: 25.0 to 45.0 (for 0-255 binary edge maps).
    """

    # --- Stage 0: Contour & Preprocessing ---
    adaptive_thresh_fine_block_size: int = 15
    adaptive_thresh_fine_C: float = 7.0
    adaptive_thresh_wide_block_size: int = 61
    adaptive_thresh_wide_C: float = 14.0
    contour_thresh_block_size: int = 25
    contour_thresh_C: float = 8.0

    min_marker_area: float = 1500.0
    max_marker_area: float = 200000.0
    min_aspect_ratio: float = 0.75
    poly_approx_epsilon: float = 0.03

    # --- Stage 1: Inner Grid Check (50% Center Box) ---
    grid_center_ratio: float = 0.50
    max_center_edge_density: float = 0.16
    warp_canonical_size: int = 180

    # --- Stage 2: Neighbor Edge Inference ---
    neighbor_edge_thresh: float = 35.0
    max_correction_rate: float = 0.0

    # --- Stage 4: Subpixel Optimization ---
    enable_subpixel_refinement: bool = True
    subpixel_iterations: int = 5
    subpixel_search_radius_cells: float = 0.22
    subpixel_n_probe_points: int = 19


class EdgeArucoDetector:
    """
    Edge-Only ArUco Detector for lithography / etched markers.

    Usage:
        detector = EdgeArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
    """

    def __init__(
        self,
        dictionary=None,
        parameters: Optional[EdgeDetectorParameters] = None,
    ):
        if dictionary is None:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.dictionary = dictionary
        self.params = parameters if parameters is not None else EdgeDetectorParameters()

    def detectMarkers(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """
        Detect ArUco markers in an edge-only image.

        Args:
            image: BGR or Grayscale input image (np.ndarray).

        Returns:
            corners: List of detected marker corners, each of shape (1, 4, 2) in float32.
                     Corners are ordered [TL, TR, BR, BL] in original image coordinates.
            ids: (N, 1) numpy array of int32 tag IDs, or None if no tags found.
            rejectedImgPoints: List of rejected candidate quads, each shape (1, 4, 2).
        """
        corners, ids, rejected, _ = self.detectMarkersWithDetails(image)
        return corners, ids, rejected

    def detectMarkersWithDetails(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        Detect markers and return intermediate inspection structures for debugging.
        """
        # Run on native image orientation
        res_native = self._run_pipeline(image)
        n_valid = len(res_native["step1_valid"])
        n_dec = len(res_native["decoded_tags"])

        res = res_native



        corners_out = []
        ids_out = []
        w_img = image.shape[1]

        for item in res["decoded_tags"]:
            c = item["refined_corners"].copy() if self.params.enable_subpixel_refinement else item["snapped_corners"].copy()

            corners_out.append(c.reshape(1, 4, 2).astype(np.float32))
            ids_out.append(item["tag_id"])

        rejected_out = []
        for sq in res["unique_squares"]:
            if not sq.get("step1_valid", False):
                c = sq["snapped_corners"].copy()
                rejected_out.append(c.reshape(1, 4, 2).astype(np.float32))

        ids_array = np.array(ids_out, dtype=np.int32).reshape(-1, 1) if ids_out else None

        return corners_out, ids_array, rejected_out, res

    # -------------------------------------------------------------------------
    # Internal Pipeline Stages
    # -------------------------------------------------------------------------

    def _run_pipeline(self, img: np.ndarray) -> Dict[str, Any]:
        p = self.params
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray_f = gray.astype(np.float32)

        # Stage 0: Dual-scale edge filtering
        e_fine = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            p.adaptive_thresh_fine_block_size, p.adaptive_thresh_fine_C
        )
        e_wide = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            p.adaptive_thresh_wide_block_size, p.adaptive_thresh_wide_C
        )
        edges_decode = cv2.bitwise_and(e_fine, e_wide)
        edges_clean = cv2.erode(edges_decode, np.ones((3, 3), np.uint8), iterations=1)

        edges_cnt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            p.contour_thresh_block_size, p.contour_thresh_C
        )
        contours, _ = cv2.findContours(edges_cnt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        raw_squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < p.min_marker_area or area > p.max_marker_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, p.poly_approx_epsilon * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = self._order_corners(approx.reshape(4, 2).astype(np.float32))
                d01 = np.linalg.norm(pts[0] - pts[1])
                d12 = np.linalg.norm(pts[1] - pts[2])
                if max(d01, d12) > 0 and (min(d01, d12) / max(d01, d12)) < p.min_aspect_ratio:
                    continue
                snapped = self._snap_quad_to_dark_line(gray_f, pts)
                raw_squares.append({
                    "area": area,
                    "raw_corners": pts,
                    "snapped_corners": snapped,
                })

        # Deduplicate inner/outer stroke contours
        unique_squares = []
        for sq in raw_squares:
            c = sq["snapped_corners"].mean(axis=0)
            side = np.linalg.norm(sq["snapped_corners"][0] - sq["snapped_corners"][1])
            is_dup = False
            for u_sq in unique_squares:
                uc = u_sq["snapped_corners"].mean(axis=0)
                uside = np.linalg.norm(u_sq["snapped_corners"][0] - u_sq["snapped_corners"][1])
                if np.linalg.norm(c - uc) < 8.0 and abs(side - uside) < 8.0:
                    is_dup = True
                    break
            if not is_dup:
                unique_squares.append(sq)

        # Stage 1: Inner grid center check
        step1_valid = []
        grid_dim = self.dictionary.markerSize
        total_cells = grid_dim + 2  # Payload + 2 border cells

        for sq in unique_squares:
            is_valid, max_density, center_densities, center_boxes, w_clean = (
                self._check_inner_grid_centers(sq["snapped_corners"], edges_clean, total_cells)
            )
            sq["step1_valid"] = is_valid
            sq["max_center_density"] = max_density
            sq["center_densities"] = center_densities
            sq["center_boxes"] = center_boxes
            sq["w_clean"] = w_clean
            if is_valid:
                step1_valid.append(sq)

        # Stage 2 & 3: Neighbor edge inference & Dictionary matching
        decoded_tags = []
        for item in step1_valid:
            (
                inner_bits, full_grid, h_edges, v_edges,
                h_vals, v_vals, h_boxes, v_boxes, w_dec
            ) = self._infer_grid_colors_from_edges(item["snapped_corners"], edges_decode, total_cells)

            ret, tag_id, rot = self.dictionary.identify(inner_bits, p.max_correction_rate)
            item.update({
                "inner_bits": inner_bits,
                "full_grid": full_grid,
                "h_edges": h_edges,
                "v_edges": v_edges,
                "h_vals": h_vals,
                "v_vals": v_vals,
                "h_boxes": h_boxes,
                "v_boxes": v_boxes,
                "w_dec": w_dec,
                "dict_matched": bool(ret),
                "tag_id": int(tag_id) if ret else -1,
                "rotation": int(rot) if ret else -1,
            })

            if ret:
                # Stage 4: Subpixel refinement
                if p.enable_subpixel_refinement:
                    refined_corners, H_subpixel, median_res_px, subpixel_edge_pts = (
                        self._refine_homography_subpixel_direct(
                            gray_f, item["snapped_corners"], int(tag_id), int(rot), total_cells
                        )
                    )
                else:
                    refined_corners = item["snapped_corners"]
                    H_subpixel = None
                    median_res_px = 0.0
                    subpixel_edge_pts = None

                item.update({
                    "refined_corners": refined_corners,
                    "H_subpixel": H_subpixel,
                    "median_res_px": median_res_px,
                    "subpixel_edge_pts": subpixel_edge_pts,
                })
                decoded_tags.append(item)

        return {
            "img": img,
            "gray": gray,
            "edges_cnt": edges_cnt,
            "edges_decode": edges_decode,
            "edges_clean": edges_clean,
            "raw_squares": raw_squares,
            "unique_squares": unique_squares,
            "step1_valid": step1_valid,
            "decoded_tags": decoded_tags,
        }

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _snap_quad_to_dark_line(
        self, gray_f: np.ndarray, pts: np.ndarray, n_samples: int = 40, search_radius: int = 6
    ) -> np.ndarray:
        curr_pts = pts.copy().astype(np.float32)
        t_vals = np.linspace(0.20, 0.80, n_samples, dtype=np.float32)
        ks = np.arange(-search_radius, search_radius + 1, dtype=np.float32)
        lines = []

        for i in range(4):
            p0, p1 = curr_pts[i], curr_pts[(i + 1) % 4]
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L < 1e-3:
                return curr_pts
            u = v / L
            n = np.array([-u[1], u[0]], dtype=np.float32)
            cs = p0[None, :] + t_vals[:, None] * v[None, :]
            xs = (cs[:, 0:1] + ks[None, :] * n[0]).astype(np.float32)
            ys = (cs[:, 1:2] + ks[None, :] * n[1]).astype(np.float32)
            profs = cv2.remap(gray_f, xs, ys, cv2.INTER_LINEAR)
            k_idxs = 1 + np.argmin(profs[:, 1:-1], axis=1)
            edge_pts = (cs + ks[k_idxs, None] * n[None, :]).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(edge_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            lines.append((-vy, vx, -vy * x0 + vx * y0))

        new_pts = []
        for i in range(4):
            a1, b1, c1 = lines[i - 1]
            a2, b2, c2 = lines[i]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                return curr_pts
            new_pts.append([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det])
        return np.array(new_pts, dtype=np.float32)

    def _check_inner_grid_centers(
        self, snapped_corners: np.ndarray, edges_clean: np.ndarray, total_cells: int
    ):
        p = self.params
        ws = p.warp_canonical_size
        dst = np.array([[0, 0], [ws, 0], [ws, ws], [0, ws]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(snapped_corners, dst)
        w_clean = cv2.warpPerspective(edges_clean, M, (ws, ws))
        step = ws / float(total_cells)

        c_ratio = p.grid_center_ratio
        c_low = (1.0 - c_ratio) / 2.0
        c_high = 1.0 - c_low

        center_densities = np.zeros((total_cells, total_cells), dtype=np.float32)
        center_boxes = []
        for r in range(total_cells):
            for c in range(total_cells):
                y0 = int(round((r + c_low) * step))
                y1 = int(round((r + c_high) * step))
                x0 = int(round((c + c_low) * step))
                x1 = int(round((c + c_high) * step))
                density = float((w_clean[y0:y1, x0:x1] > 0).mean())
                center_densities[r, c] = density
                center_boxes.append((r, c, x0, y0, x1, y1, density))

        max_density = float(center_densities.max())
        is_valid = max_density < p.max_center_edge_density
        return is_valid, max_density, center_densities, center_boxes, w_clean

    def _infer_grid_colors_from_edges(
        self, snapped_corners: np.ndarray, edges_decode: np.ndarray, total_cells: int
    ):
        p = self.params
        ws = p.warp_canonical_size
        dst = np.array([[0, 0], [ws, 0], [ws, ws], [0, ws]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(snapped_corners, dst)
        w_dec = cv2.warpPerspective(edges_decode, M, (ws, ws))
        step = ws / float(total_cells)
        k_inner = total_cells - 2  # Payload bit dimension (e.g. 4 for 4x4)

        # Vertical edges between (r, c) and (r, c+1)
        v_vals = np.zeros((total_cells, total_cells - 1), dtype=np.float32)
        v_boxes = []
        for r in range(total_cells):
            for c in range(total_cells - 1):
                x = int(round((c + 1) * step))
                y0 = int(round(r * step + 8))
                y1 = int(round((r + 1) * step - 8))
                x0, x1 = max(0, x - 4), min(ws, x + 5)
                v_vals[r, c] = float(w_dec[y0:y1, x0:x1].mean())
                v_boxes.append((r, c, x0, y0, x1, y1, v_vals[r, c]))

        # Horizontal edges between (r, c) and (r+1, c)
        h_vals = np.zeros((total_cells - 1, total_cells), dtype=np.float32)
        h_boxes = []
        for r in range(total_cells - 1):
            for c in range(total_cells):
                y = int(round((r + 1) * step))
                x0 = int(round(c * step + 8))
                x1 = int(round((c + 1) * step - 8))
                y0, y1 = max(0, y - 4), min(ws, y + 5)
                h_vals[r, c] = float(w_dec[y0:y1, x0:x1].mean())
                h_boxes.append((r, c, x0, y0, x1, y1, h_vals[r, c]))

        h_edges = (h_vals > p.neighbor_edge_thresh).astype(np.uint8)
        v_edges = (v_vals > p.neighbor_edge_thresh).astype(np.uint8)

        n_vars = k_inner * k_inner
        A = np.zeros((n_vars, n_vars), dtype=np.float64)
        b = np.zeros(n_vars, dtype=np.float64)

        def idx(r, c):
            return (r - 1) * k_inner + (c - 1)

        # Horizontal edge constraints on interior cells
        for c in range(1, k_inner + 1):
            for r in range(k_inner + 1):
                val = float(h_vals[r, c])
                t = -1.0 if val > p.neighbor_edge_thresh else 1.0
                w = 0.1 + abs(val - p.neighbor_edge_thresh) / 40.0
                if r == 0:
                    i = idx(1, c)
                    A[i, i] += w
                    b[i] += w * t
                elif r == k_inner:
                    i = idx(k_inner, c)
                    A[i, i] += w
                    b[i] += w * t
                else:
                    i, j = idx(r, c), idx(r + 1, c)
                    A[i, i] += w
                    A[j, j] += w
                    A[i, j] -= w * t
                    A[j, i] -= w * t

        # Vertical edge constraints on interior cells
        for r in range(1, k_inner + 1):
            for c in range(k_inner + 1):
                val = float(v_vals[r, c])
                t = -1.0 if val > p.neighbor_edge_thresh else 1.0
                w = 0.1 + abs(val - p.neighbor_edge_thresh) / 40.0
                if c == 0:
                    i = idx(r, 1)
                    A[i, i] += w
                    b[i] += w * t
                elif c == k_inner:
                    i = idx(r, k_inner)
                    A[i, i] += w
                    b[i] += w * t
                else:
                    i, j = idx(r, c), idx(r, c + 1)
                    A[i, i] += w
                    A[j, j] += w
                    A[i, j] -= w * t
                    A[j, i] -= w * t

        s = np.linalg.solve(A, b)
        inner_bits = (s < 0).astype(np.uint8).reshape(k_inner, k_inner)
        full_grid = np.zeros((total_cells, total_cells), dtype=np.uint8)
        full_grid[1 : total_cells - 1, 1 : total_cells - 1] = inner_bits

        return inner_bits, full_grid, h_edges, v_edges, h_vals, v_vals, h_boxes, v_boxes, w_dec

    def _refine_homography_subpixel_direct(
        self, gray_f: np.ndarray, pts_init: np.ndarray, tag_id: int, rot: int, total_cells: int
    ):
        p = self.params
        TC = float(total_cells)
        k_inner = total_cells - 2
        src_canon = np.array([[0, 0], [TC, 0], [TC, TC], [0, TC]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_canon, pts_init.astype(np.float32)).astype(np.float64)

        marker = (cv2.aruco.generateImageMarker(self.dictionary, tag_id, total_cells) > 0).astype(np.uint8)
        inner = np.rot90(marker[1 : total_cells - 1, 1 : total_cells - 1], -rot)
        grid = np.zeros((total_cells, total_cells), dtype=np.uint8)
        grid[1 : total_cells - 1, 1 : total_cells - 1] = inner

        # Outer square borders + internal bit transitions
        v_segs = [(0.0, 0.9, TC - 0.9, 2.0), (TC, 0.9, TC - 0.9, 2.0)]
        for r in range(1, total_cells - 1):
            for c in range(total_cells - 1):
                if grid[r, c] != grid[r, c + 1]:
                    v_segs.append((float(c + 1), r + 0.28, r + 0.72, 1.0))

        h_segs = [(0.0, 0.9, TC - 0.9, 2.0), (TC, 0.9, TC - 0.9, 2.0)]
        for r in range(total_cells - 1):
            for c in range(1, total_cells - 1):
                if grid[r, c] != grid[r + 1, c]:
                    h_segs.append((float(r + 1), c + 0.28, c + 0.72, 1.0))

        samples = []
        for u0, v0_s, v1_s, w_s in v_segs:
            n_pts = max(8, int(round((v1_s - v0_s) * 15)))
            for v in np.linspace(v0_s, v1_s, n_pts):
                samples.append((u0, float(v), True, w_s))
        for v0, u0_s, u1_s, w_s in h_segs:
            n_pts = max(8, int(round((u1_s - u0_s) * 15)))
            for u in np.linspace(u0_s, u1_s, n_pts):
                samples.append((float(u), v0, False, w_s))

        ks = np.linspace(-p.subpixel_search_radius_cells, p.subpixel_search_radius_cells, p.subpixel_n_probe_points, dtype=np.float64)
        step_k = ks[1] - ks[0]

        N = len(samples)
        u_base = np.array([s[0] for s in samples], dtype=np.float64)
        v_base = np.array([s[1] for s in samples], dtype=np.float64)
        is_vert = np.array([s[2] for s in samples], dtype=bool)
        w_base = np.array([s[3] for s in samples], dtype=np.float64)

        U = u_base[:, None] + np.where(is_vert[:, None], ks[None, :], 0.0)
        V = v_base[:, None] + np.where(~is_vert[:, None], ks[None, :], 0.0)
        uv_flat = np.stack([U.ravel(), V.ravel()], axis=-1).reshape(1, -1, 2).astype(np.float64)

        A = np.zeros((N, 8), dtype=np.float64)
        A[is_vert, 0] = u_base[is_vert]
        A[is_vert, 1] = v_base[is_vert]
        A[is_vert, 2] = 1.0
        A[is_vert, 6] = -(u_base[is_vert] ** 2)
        A[is_vert, 7] = -u_base[is_vert] * v_base[is_vert]

        A[~is_vert, 3] = u_base[~is_vert]
        A[~is_vert, 4] = v_base[~is_vert]
        A[~is_vert, 5] = 1.0
        A[~is_vert, 6] = -u_base[~is_vert] * v_base[~is_vert]
        A[~is_vert, 7] = -(v_base[~is_vert] ** 2)

        idx_r = np.arange(N)
        median_res_px = 0.0
        subpixel_edge_pts = None

        for _ in range(p.subpixel_iterations):
            xy_flat = cv2.perspectiveTransform(uv_flat, H).reshape(N, len(ks), 2).astype(np.float32)
            profs = cv2.remap(gray_f, xy_flat[:, :, 0], xy_flat[:, :, 1], cv2.INTER_CUBIC)
            k_idxs = 1 + np.argmin(profs[:, 1:-1], axis=1)
            ym1 = profs[idx_r, k_idxs - 1]
            y0 = profs[idx_r, k_idxs]
            yp1 = profs[idx_r, k_idxs + 1]
            denom = 2.0 * (ym1 - 2.0 * y0 + yp1)
            delta = np.where(np.abs(denom) > 1e-4, (ym1 - yp1) / denom, 0.0)
            delta = np.clip(delta, -0.5, 0.5)
            d_norm = ks[k_idxs] + delta * step_k

            u_det = u_base + np.where(is_vert, d_norm, 0.0)
            v_det = v_base + np.where(~is_vert, d_norm, 0.0)
            uv_det = np.stack([u_det, v_det], axis=-1).reshape(1, -1, 2).astype(np.float64)
            subpixel_edge_pts = cv2.perspectiveTransform(uv_det, H).reshape(-1, 2)

            b = d_norm
            w = w_base.copy()
            for _ in range(3):
                W = np.sqrt(w)[:, None]
                g, _, _, _ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
                res = np.abs(A @ g - b)
                w = w_base * np.where(res < 0.02, 1.0, 0.02 / (res + 1e-6))

            dG = np.array([
                [1.0 + g[0], g[1], g[2]],
                [g[3], 1.0 + g[4], g[5]],
                [g[6], g[7], 1.0]
            ], dtype=np.float64)
            H = H @ dG
            H /= H[2, 2]
            cell_px = float(np.linalg.norm(pts_init[0] - pts_init[1]) / TC)
            median_res_px = float(np.median(res) * cell_px)

        refined_pts = cv2.perspectiveTransform(src_canon.reshape(1, 4, 2), H).reshape(4, 2)
        return refined_pts.astype(np.float32), H, median_res_px, subpixel_edge_pts


def detect_edge_markers(
    image: np.ndarray,
    dictionary=None,
    parameters: Optional[EdgeDetectorParameters] = None,
) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
    """
    Functional wrapper identical to OpenCV's cv2.aruco.detectMarkers:

        corners, ids, rejected = detect_edge_markers(image, dictionary, parameters)

    Args:
        image: BGR or grayscale numpy image.
        dictionary: cv2.aruco Dictionary (defaults to DICT_4X4_1000).
        parameters: Optional EdgeDetectorParameters instance.

    Returns:
        corners: List of (1, 4, 2) float32 arrays in original image coordinates.
        ids: (N, 1) int32 array of tag IDs, or None.
        rejected: List of (1, 4, 2) float32 arrays of rejected candidates.
    """
    detector = EdgeArucoDetector(dictionary=dictionary, parameters=parameters)
    return detector.detectMarkers(image)


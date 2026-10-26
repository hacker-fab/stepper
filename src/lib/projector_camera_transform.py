from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from lib.ar_tag import get_aruco_dict
from lib.cross_marker import MarkerSpec


def generate_calibration_tag_grid_with_coords(
    grid_n: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    dict_id: int = cv2.aruco.DICT_4X4_250,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """Generates an N x N ArUco grid pattern and returns known projector coordinates.
    
    Args:
        grid_n: Grid dimension (default 2 for 2x2 grid).
        canvas_size: (width, height) of projector canvas.
        dict_id: OpenCV ArUco dictionary identifier.
        
    Returns:
        bgr_image: (height, width, 3) uint8 image.
        tag_coords: Dict mapping marker_id -> {
            "corners": np.ndarray shape (4, 2) [top-left, top-right, bottom-right, bottom-left],
            "center": (cx, cy)
        }
    """
    width, height = canvas_size
    grid_n = max(1, int(grid_n))
    dictionary = get_aruco_dict(dict_id)

    # 5% border margin
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y

    cell_w = usable_w / grid_n
    cell_h = usable_h / grid_n
    marker_sz = max(12, int(min(cell_w, cell_h) * 0.65))
    quiet = max(4, int(marker_sz * 0.2))
    box_sz = marker_sz + 2 * quiet

    canvas = np.zeros((height, width), dtype=np.uint8)
    tag_coords: Dict[int, Dict[str, Any]] = {}

    max_markers = 250
    for r in range(grid_n):
        for c in range(grid_n):
            idx = (r * grid_n + c) % max_markers
            cx = int(margin_x + (c + 0.5) * cell_w)
            cy = int(margin_y + (r + 0.5) * cell_h)

            x1 = cx - box_sz // 2
            y1 = cy - box_sz // 2
            x2 = x1 + box_sz
            y2 = y1 + box_sz

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue

            canvas[y1:y2, x1:x2] = 255

            mx = x1 + quiet
            my = y1 + quiet
            marker_img = cv2.aruco.generateImageMarker(dictionary, idx, marker_sz)
            canvas[my : my + marker_sz, mx : mx + marker_sz] = marker_img

            # Order: top-left, top-right, bottom-right, bottom-left (matching OpenCV ArUco order)
            corners = np.array(
                [
                    [float(mx), float(my)],
                    [float(mx + marker_sz), float(my)],
                    [float(mx + marker_sz), float(my + marker_sz)],
                    [float(mx), float(my + marker_sz)],
                ],
                dtype=np.float64,
            )
            marker_center = (float(mx + marker_sz / 2.0), float(my + marker_sz / 2.0))

            tag_coords[idx] = {
                "corners": corners,
                "center": marker_center,
            }

    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = canvas
    bgr[:, :, 1] = canvas
    bgr[:, :, 2] = canvas
    return bgr, tag_coords


def compute_projector_to_camera_homography(
    detected_corners: List[np.ndarray],
    detected_ids: Optional[np.ndarray],
    known_tag_coords: Dict[int, Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], int]:
    """Computes the 3x3 homography matrix H mapping projector coordinates to camera coordinates.
    
    Args:
        detected_corners: List of (1, 4, 2) corner arrays detected in the camera frame.
        detected_ids: Array of detected marker IDs.
        known_tag_coords: Dictionary of known projector coordinates from tag generation.
        
    Returns:
        (H, matched_points_count) where H is (3, 3) float64 ndarray or None if < 4 points.
    """
    if detected_ids is None or len(detected_corners) == 0:
        return None, 0

    src_pts: List[Tuple[float, float]] = []
    dst_pts: List[Tuple[float, float]] = []

    flat_ids = detected_ids.flatten()
    for idx, marker_id in enumerate(flat_ids):
        marker_id_int = int(marker_id)
        if marker_id_int not in known_tag_coords:
            continue

        proj_corners = known_tag_coords[marker_id_int]["corners"]  # shape (4, 2)
        cam_corners = detected_corners[idx].reshape(-1, 2)  # shape (4, 2)

        for p_pt, c_pt in zip(proj_corners, cam_corners):
            src_pts.append((float(p_pt[0]), float(p_pt[1])))
            dst_pts.append((float(c_pt[0]), float(c_pt[1])))

    if len(src_pts) < 4:
        return None, len(src_pts)

    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)

    homography, inliers = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
    matched_count = int(np.sum(inliers)) if inliers is not None else len(src_pts)

    return homography, matched_count


def transform_points_projector_to_camera(
    points: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """Transforms an N x 2 array of points from projector coordinates to camera image coordinates using H."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    # Convert to homogeneous coords
    pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])  # (N, 3)
    transformed_homo = (homography @ pts_homo.T).T  # (N, 3)

    # Normalize by w
    w = transformed_homo[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    transformed = transformed_homo[:, :2] / w
    return transformed


def transform_marker_to_camera_space(
    marker_spec: MarkerSpec,
    homography: np.ndarray,
) -> Dict[str, Any]:
    """Transforms a MarkerSpec from projector space into camera image space using homography H.
    
    Returns a dictionary containing:
        - id
        - center_proj: [x, y] in projector pixels
        - center_img: [x, y] in camera pixels
        - scale_proj_px
        - scale_proj_pct
        - rotation_proj_deg
        - rotation_img_deg: transformed rotation angle in camera space
        - longer_arm_tip_img: [x, y] in camera pixels
        - vertices_img: list of [x, y] in camera pixels
    """
    pts_to_transform = np.array(
        [
            marker_spec.center_proj,
            marker_spec.longer_arm_tip_proj,
        ]
        + marker_spec.vertices_proj,
        dtype=np.float64,
    )

    transformed = transform_points_projector_to_camera(pts_to_transform, homography)

    center_img = (float(transformed[0, 0]), float(transformed[0, 1]))
    longer_tip_img = (float(transformed[1, 0]), float(transformed[1, 1]))
    vertices_img = [(float(pt[0]), float(pt[1])) for pt in transformed[2:]]

    # Transformed direction vector from center to longer arm tip
    dx = longer_tip_img[0] - center_img[0]
    dy = longer_tip_img[1] - center_img[1]
    rot_rad = np.arctan2(dy, dx)
    rotation_img_deg = float(np.degrees(rot_rad) % 360.0)

    # The longer arm in base units extends 2.5b. Total horizontal width is 4b.
    # Therefore, scale in camera image space = 4/2.5 * length(longer_tip - center)
    arm_len = float(np.hypot(dx, dy))
    scale_img_px = arm_len * (4.0 / 2.5)

    return {
        "id": marker_spec.id,
        "center_proj": [marker_spec.center_proj[0], marker_spec.center_proj[1]],
        "center_img": [center_img[0], center_img[1]],
        "scale_proj_px": marker_spec.scale_px,
        "scale_proj_pct": marker_spec.scale_pct,
        "scale_img_px": scale_img_px,
        "rotation_proj_deg": marker_spec.rotation_deg,
        "rotation_img_deg": rotation_img_deg,
        "thickness": getattr(marker_spec, "thickness", 1.0),
        "longer_arm_tip_proj": [marker_spec.longer_arm_tip_proj[0], marker_spec.longer_arm_tip_proj[1]],
        "longer_arm_tip_img": [longer_tip_img[0], longer_tip_img[1]],
        "vertices_proj": [[v[0], v[1]] for v in marker_spec.vertices_proj],
        "vertices_img": vertices_img,
    }


def warp_projector_image_to_camera(
    projector_img: np.ndarray,
    homography: np.ndarray,
    camera_size: Tuple[int, int],
) -> np.ndarray:
    """Warps a projector ground-truth image into camera image coordinates.
    
    Args:
        projector_img: BGR or grayscale image in projector dimensions.
        homography: 3x3 homography matrix mapping projector coordinates to camera coordinates.
        camera_size: (width, height) of destination camera frame.
        
    Returns:
        warped: BGR image of shape (camera_height, camera_width, 3).
    """
    cam_w, cam_h = camera_size
    if projector_img.ndim == 2:
        img_bgr = cv2.cvtColor(projector_img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = projector_img

    warped = cv2.warpPerspective(img_bgr, homography, (cam_w, cam_h), flags=cv2.INTER_LINEAR)
    return warped

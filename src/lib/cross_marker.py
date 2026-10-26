from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class MarkerSpec:
    id: int
    center_proj: Tuple[float, float]
    scale_pct: float
    scale_px: float
    rotation_deg: float
    longer_arm_tip_proj: Tuple[float, float]
    vertices_proj: List[Tuple[float, float]]
    thickness: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center_proj": [self.center_proj[0], self.center_proj[1]],
            "scale_pct": self.scale_pct,
            "scale_px": self.scale_px,
            "rotation_deg": self.rotation_deg,
            "thickness": self.thickness,
            "longer_arm_tip_proj": [self.longer_arm_tip_proj[0], self.longer_arm_tip_proj[1]],
            "vertices_proj": [[v[0], v[1]] for v in self.vertices_proj],
        }


def get_base_cross_vertices(block_size: float, thickness: float = 1.0) -> np.ndarray:
    """Returns the 12 vertices of the asymmetric cross relative to the intersection center.
    
    The bounding box remains unchanged:
      Left arm: -1.5 * b
      Right arm: +2.5 * b
      Top arm: -1.5 * b
      Bottom arm: +1.5 * b

    Thickness factor (0 to 1) scales the bar width/height (nominal thickness = 1.0 corresponds to 1 block width):
      Horizontal bar half-height: h = 0.5 * b * thickness (spans y = -h to +h)
      Vertical bar half-width: h = 0.5 * b * thickness (spans x = -h to +h)
    """
    b = float(block_size)
    t = float(np.clip(thickness, 0.0, 1.0))
    h = 0.5 * b * t
    vertices = np.array(
        [
            [-h, -1.5 * b],
            [h, -1.5 * b],
            [h, -h],
            [2.5 * b, -h],
            [2.5 * b, h],
            [h, h],
            [h, 1.5 * b],
            [-h, 1.5 * b],
            [-h, h],
            [-1.5 * b, h],
            [-1.5 * b, -h],
            [-h, -h],
        ],
        dtype=np.float64,
    )
    return vertices


def generate_cross_pattern(
    canvas_size: Tuple[int, int],
    target_scale_pct: float = 8.0,
    scale_jitter_pct: float = 2.0,
    marker_count: int = 20,
    min_thickness: float = 0.2,
    max_thickness: float = 1.0,
    margin_px: int = 10,
    min_spacing_px: float = 5.0,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[MarkerSpec]]:
    """Generates a binary BGR canvas filled with non-overlapping, randomly scaled and rotated cross markers.
    
    Args:
        canvas_size: (width, height) of the projector canvas.
        target_scale_pct: Nominal scale of marker width as percentage of projector's longer edge.
        scale_jitter_pct: Uniform random variation (+/-) around target_scale_pct.
        marker_count: Desired number of markers.
        min_thickness: Minimum thickness factor [0.0, 1.0].
        max_thickness: Maximum thickness factor [0.0, 1.0].
        margin_px: Boundary margin to keep markers inside canvas.
        min_spacing_px: Minimum clearance between enclosing circles of adjacent markers.
        random_seed: Optional seed for reproducible generation.
        
    Returns:
        canvas: (height, width, 3) uint8 BGR image with white cross markers on black background.
        markers: List of MarkerSpec containing ground truth projector coordinates and attributes.
    """
    width, height = canvas_size
    rng = np.random.default_rng(random_seed)
    longer_edge = float(max(width, height))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    placed_markers: List[MarkerSpec] = []
    # Store (cx, cy, enclosing_radius) for quick non-overlapping collision checks
    placed_circles: List[Tuple[float, float, float]] = []

    max_attempts_per_marker = 300

    for marker_idx in range(marker_count):
        placed = False
        for _ in range(max_attempts_per_marker):
            # 1. Sample scale
            scale_min = max(0.5, target_scale_pct - scale_jitter_pct)
            scale_max = max(scale_min, target_scale_pct + scale_jitter_pct)
            scale_pct = float(rng.uniform(scale_min, scale_max))

            # Total width of cross is 4 blocks
            scale_px = (scale_pct / 100.0) * longer_edge
            block_size = max(1.0, scale_px / 4.0)

            # Sample thickness factor
            t_min = max(0.0, min(min_thickness, max_thickness))
            t_max = min(1.0, max(min_thickness, max_thickness))
            thickness = float(rng.uniform(t_min, t_max))

            # Enclosing radius around intersection center (1.5b, 1.5b)
            # Max corner is at (2.5b, 0.5b * thickness) => sqrt(2.5^2 + (0.5*t)^2) * b <= 2.55 * b
            enclosing_radius = 2.6 * block_size

            # 2. Sample rotation angle in [0, 2*pi)
            theta_rad = float(rng.uniform(0.0, 2.0 * np.pi))
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

            # 3. Sample center inside usable bounds
            min_x = margin_px + enclosing_radius
            max_x = width - margin_px - enclosing_radius
            min_y = margin_px + enclosing_radius
            max_y = height - margin_px - enclosing_radius

            if min_x >= max_x or min_y >= max_y:
                # If marker is too large for canvas, skip
                continue

            cx = float(rng.uniform(min_x, max_x))
            cy = float(rng.uniform(min_y, max_y))

            # 4. Check collision with existing markers
            collision = False
            for prev_cx, prev_cy, prev_r in placed_circles:
                dist_sq = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
                min_dist = enclosing_radius + prev_r + min_spacing_px
                if dist_sq < (min_dist * min_dist):
                    collision = True
                    break

            if collision:
                continue

            # 5. Transform vertices
            base_vertices = get_base_cross_vertices(block_size, thickness=thickness)
            rotated_vertices = (base_vertices @ rot_mat.T) + np.array([cx, cy])

            # Verify all vertices are strictly inside canvas
            if (
                np.any(rotated_vertices[:, 0] < 0)
                or np.any(rotated_vertices[:, 0] >= width)
                or np.any(rotated_vertices[:, 1] < 0)
                or np.any(rotated_vertices[:, 1] >= height)
            ):
                continue

            # Longer arm tip is at (+2.5b, 0) relative to intersection center
            longer_tip_base = np.array([2.5 * block_size, 0.0])
            longer_tip = (longer_tip_base @ rot_mat.T) + np.array([cx, cy])

            rotation_deg = float(np.degrees(theta_rad) % 360.0)

            spec = MarkerSpec(
                id=marker_idx,
                center_proj=(cx, cy),
                scale_pct=scale_pct,
                scale_px=scale_px,
                rotation_deg=rotation_deg,
                longer_arm_tip_proj=(float(longer_tip[0]), float(longer_tip[1])),
                vertices_proj=[(float(pt[0]), float(pt[1])) for pt in rotated_vertices],
                thickness=thickness,
            )

            # Draw polygon onto canvas
            pts_int = np.int32(np.round(rotated_vertices))
            cv2.fillPoly(canvas, [pts_int], (255, 255, 255))

            placed_markers.append(spec)
            placed_circles.append((cx, cy, enclosing_radius))
            placed = True
            break

    return canvas, placed_markers

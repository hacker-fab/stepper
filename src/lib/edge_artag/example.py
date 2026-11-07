"""
Example: Using EdgeAruco with OpenCV-style calling interface
============================================================
"""

import cv2
from . import edge_artag as ea


def main():
    # 1. Load an edge-only sample image
    image_path = "manual_snapshot_2026-10-20_09-42-09.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    image = cv2.flip(image, 1)

    # 2. Get the ArUco dictionary (default DICT_4X4_1000)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

    # 3. (Optional) Customize detector parameters
    params = ea.EdgeDetectorParameters(
        adaptive_thresh_fine_block_size=15,  # ~0.7x cell size (tuned to stroke width)
        adaptive_thresh_wide_block_size=61,  # ~3.0x cell size (broad background level)
        max_center_edge_density=0.16,        # Max edge density allowed in 50% cell center
        neighbor_edge_thresh=35.0,           # Threshold to detect color transition
        enable_subpixel_refinement=True,     # Subpixel Gauss-Newton homography fit
    )

    # 4. Detect markers — identical calling signature to cv2.aruco.detectMarkers:
    corners, ids, rejected = ea.detect_edge_markers(image, dictionary, parameters=params)

    print(f"Found {len(corners)} markers:")
    if ids is not None:
        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = c.mean(axis=0)
            print(f"  Tag ID {tag_id:3d}: Center=({center[0]:.2f}, {center[1]:.2f}), "
                  f"Corners=TL({c[0,0]:.1f},{c[0,1]:.1f}) -> BR({c[2,0]:.1f},{c[2,1]:.1f})")

    print(f"Rejected quads (e.g. outer frames): {len(rejected)}")

    # 5. Draw detected markers using standard OpenCV draw function or custom overlay
    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    output_path = "detected_markers_example.png"
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()


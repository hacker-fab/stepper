from PIL import Image
import cv2 as cv
import os
import json
import math
import numpy as np
import onnxruntime as rt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

RF_DETR_IMPUT_SIZE = 704
CONFIDENCE_THRESHOLD = 0.75

px_to_step_x = 1.0/1.576
px_to_step_y = 1.0/1.668
digital_to_cam_view = 0.5

################## Capture and Stitch Utility Functions ##################
def stitching_preprocess(img):
    """
    pre-processing helper for all captured images
    before capture nd stitch stage in layer 2+ tiling

    Returns the image that has been pre-processed
    """
    def crop_image(img, h_start, h_end, w_start, w_end):
        return img[h_start:h_end, w_start:w_end]
    
    preprocessed = img.copy()

    gaussian_kernel_size = (5, 5)
    preprocessed = cv.cvtColor(preprocessed, cv.COLOR_BGR2GRAY)
    preprocessed = cv.GaussianBlur(preprocessed, gaussian_kernel_size, 0)
    h, w = preprocessed.shape[:2]
    margin_h = 150
    margin_w = margin_h
    preprocessed = crop_image(preprocessed, margin_h, h - margin_h, margin_w, w - margin_w)
    return preprocessed

def read_info_file(directory, info_file="info.txt"):
    info_file_path = os.path.join(directory, info_file)
    file = open(info_file_path, 'r')
    content = file.read()
    return json.loads(content)

def read_tiles(directory, info):
    rows = info['rows']
    cols = info['cols']

    imgs = []
    img_paths = []
    for row in range(rows):
        row_imgs = []
        row_paths = []
        for col in range(cols):
            img_path = os.path.join(directory, f"tile_{row}_{col}.png")
            row_paths.append(f"tile_{row}_{col}.png")

            img = cv.imread(img_path)
            preprocessed = stitching_preprocess(img)
            row_imgs.append(preprocessed)
        imgs.append(row_imgs)
        img_paths.append(row_paths)
    return imgs, img_paths

################## Alignment and Detection Utility Functions ##################
def rf_detr_preprocess(img, layer: int = 1):
    """
    pre-processes any type of image and prepares it
    for alignment marker detection.

    Returns the pre-processed image, the original image
    width and the original image height in pixels
    """
    def clahe(img_cleaned):
        clahe_obj = cv.createCLAHE(clipLimit=30.0, tileGridSize=(15, 18))

        if len(img_cleaned.shape) == 3:
            gray = cv.cvtColor(img_cleaned, cv.COLOR_RGB2GRAY)
        else:
            gray = img_cleaned

        return clahe_obj.apply(gray)

    if img is None:
        raise Exception("Error: image is None")

    # Convert PIL to numpy
    if isinstance(img, Image.Image):
        img = img.convert('RGB')
        img = np.array(img)

    processed = img.copy()
    if processed.ndim == 4:
        processed = processed[0].transpose(1, 2, 0)
    elif processed.ndim == 2:
        processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)

    # Strip alpha channel if present (RGBA -> RGB)
    if processed.ndim == 3 and processed.shape[2] == 4:
        processed = processed[:, :, :3]

    if layer == 1:
        processed = clahe(processed)                                    # -> grayscale (H, W)
        processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)         # -> (H, W, 3)
        processed = np.array(processed)

    orig_h, orig_w = processed.shape[:2]
    img_resized = cv.resize(processed, (RF_DETR_IMPUT_SIZE, RF_DETR_IMPUT_SIZE))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0
    print(img_input.shape, "with orig_h", orig_h, "and orig_w", orig_w)
    return img_input, orig_h, orig_w

def estimate_transform(dest: np.ndarray, src: np.ndarray) -> tuple[float, float, float]:
    """
    Given matched point pairs, estimate (dx, dy, rotation_degrees)
    using least-squares rigid body fit.

    Precondition: dest and src are of same size
    Assumption made: user does not tilt the chip by more than 90 degrees
    because sth is wrong with chip placement if that's the case and
    user should benefit from reloading the chip on the stage

    dst_pts: (K, 2) — camera-detected positions (after step shift)
    src_pts: (K, 2) — pattern expected positions
    """

    # --- Translation: avg x,y offset ---
    assert dest.shape == src.shape, "dest and src sized differently"

    delta = dest - src
    dx = float(np.mean(delta[:, 0]))
    dy = float(np.mean(delta[:, 1]))

    dest_centroid = dest - dest.mean(axis=0)
    src_centroid = src - src.mean(axis=0)

    angles = []
    for s, d in zip(dest_centroid, src_centroid):
        cross = s[0]*d[1] - s[1]*d[0]
        dot   = s[0]*d[0] + s[1]*d[1]
        angle = np.arctan2(cross, dot)
        angles.append(angle)

    rotation_deg = float(np.degrees(np.mean(angles)))
    return (dx, dy, rotation_deg)

def detect_marks_for_slam(img, session, orig_h, orig_w, threshold=0.77) -> list[dict]:
    """
    Detects alignment markers for tiling SLAM algorithm
    Returns an array of dictionary coordinates such that
    - `arr[i] = {"center": (x,y), "left":(x,y), "right":(x,y), "top":(x,y), "bottom":(x,y)}`
    """
    vis = img.copy()
    print("detecting image of shape: ", (vis.shape[1]) <= 3)
    assert ((vis.shape[1]) <= 3), "bad image color dimensions"

    # Run inference with our weights
    input_name = session.get_inputs()[0].name
    boxes, scores = session.run(None, {input_name: img})

    # collect good matches
    boxes = boxes[0]
    scores = scores[0]

    markers = []
    for i in range(len(boxes)):
        class_id = np.argmax(scores[i])
        confidence = 1 / (1 + np.exp(-scores[i][class_id]))
        if confidence > threshold:
            x, y, w, h = boxes[i]

            # fetch centers, and scale back to orig_img size
            cx = int(x * orig_w)
            cy = int(y * orig_h)
            bw = int(w * orig_w)
            bh = int(h * orig_h)

            marker = {
                "center": (cx, cy),
                "left":   (cx - bw // 2, cy),
                "right":  (cx + bw // 2, cy),
                "top":    (cx, cy - bh // 2),
                "bottom": (cx, cy + bh // 2)
            }
            markers.append(marker)
    return markers

################## Snake Pattern Tiling Functions ##################
def get_next_tile_vector(row:int, col:int, width:int, height:int, num_rows:int, num_cols:int, num_steps:int, error_x:int=0, error_y:int = 0):
    """
    determine next step direction and size (in steps)
    Arguments:
        - row: current row
        - col: current column
        - width: total amount of steps to take horizontally (steps)
        - height: total amount of steps to take vertically (steps)
        - num_rows, num_cols: number of columns
        - num_steps: how many steps of movement
        - error_x=0: x step-errors from previous step (steps)
        - error_y=0: y step-errors from previous step (steps)

    Returns: tuple(h_direction, v_direction, step_x, step_y) --> units: (steps)
    """           
    is_row_transition = ((col == 0 and row % 2 == 1) or (col == num_cols-1 and row % 2 == 0))
    if is_row_transition == True: # moving to different row
        v_direction = 'down'
        h_direction = None
    else: # moving to different column
        h_direction = 'right' if (row % 2 == 0) else 'left'
        v_direction = None
    print(f"width={width}, error_x={error_x}, num_steps={num_steps}, height={height}, error_y={error_y}")
    step_x = math.floor(width - error_x) if h_direction is not None else 0
    step_y = math.floor(height + error_y) if v_direction is not None else 0
    print(f"h_direction={h_direction}, v_direction={v_direction}, step_x={step_x}, step_y={step_y}")
    return (h_direction, v_direction, step_x, step_y)

def match_alignment_markers_by_coordinates(dest_marks, src_marks, src_marks_shifted, img_h, img_w):

    img_diagonal = np.sqrt(img_h**2 + img_w**2)
    match_threshold = 0.08 * img_diagonal
    dists = cdist(dest_marks, src_marks_shifted)
    row_ind, col_ind = linear_sum_assignment(dists)

    matched_dest = []
    matched_src_shifted = []   # shifted — for transform calculation
    matched_src_original = []  # original — for graphing/visualization purposes
    for r, c in zip(row_ind, col_ind):
        dist = dists[r, c]
        status = "✓" if dist < match_threshold else "✗ REJECTED"
        print(f"  cam[{r}]={dest_marks[r]} → pat[{c}]={src_marks_shifted[c]}  dist={dist:.1f}px {status}")
        if dist < match_threshold:
            matched_dest.append(dest_marks[r])
            matched_src_shifted.append(src_marks_shifted[c])
            matched_src_original.append(src_marks[c])
    
    return (matched_dest, matched_src_original, matched_src_shifted)


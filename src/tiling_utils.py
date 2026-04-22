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

# digital pattern pixels to step size scalar
px_to_step_x = 1.0/1.576
px_to_step_y = 1.0/1.668
digital_to_cam_view = 0.5

# image set position scale (steps to pixels scalar)
step_to_projection_pixels_x = 100.0/220.0
step_to_projection_pixels_y = 100.0/160.0

# auto_align scale (digital to projection scalar)
digital_to_cam_scale_w = 2.0469
digital_to_cam_scale_h = 1.8

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
        print("Latent image! Must be pre-processed")
        processed = clahe(processed)                                    # -> grayscale (H, W)
        processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)         # -> (H, W, 3)
        processed = np.array(processed)

    orig_h, orig_w = processed.shape[:2]
    img_resized = cv.resize(processed, (RF_DETR_IMPUT_SIZE, RF_DETR_IMPUT_SIZE))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0).astype(np.float32) / 255.0
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
    # for some reason moving in this direction causes overstep?
    if h_direction == 'left':
        step_x -= 10
    step_y = math.floor(height + error_y) if v_direction is not None else 0
    print(f"h_direction={h_direction}, v_direction={v_direction}, step_x={step_x}, step_y={step_y}")
    return (h_direction, v_direction, step_x, step_y)

def match_alignment_markers_by_coordinates(dest_marks, src_marks, src_marks_shifted, img_h, img_w):

    img_diagonal = np.sqrt(img_h**2 + img_w**2)
    match_threshold = 0.1 * img_diagonal
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

######################## Auto-Alignment Feature Functions ########################
def fetch_alignemnt_errors(model, camera: Image, pattern: Image, layer=1) -> tuple[float, float, float]:
    """
    Takes in 2 images:
    - `camera`: image of current state
    - `pattern`: image of digital pattern projected onto chip

    Preconditions:
    - imgs must be of same size (or scaled to same size for convenience)
    - step_size is given in stepper steps

    Match camera detections to pattern detections by nearest-neighbor
    after normalizing for scale/translation.Then calculates transform
    Returns list of tuple(dx, dy, rotation degree) pairs (pixels)
    """
    assert camera is not None and pattern is not None, "Error: one or both images are None"
    # pre-processing
    dest_processed, d_h, d_w = rf_detr_preprocess(camera, layer)
    src_processed, s_h, s_w = rf_detr_preprocess(pattern, layer+1)

    # intermediate check
    assert dest_processed.shape == src_processed.shape, "Error: images are differently sized, exiting function"

    # detect raw alignment marks -> raw marks are in pixels
    src_marks_raw = detect_marks_for_slam(src_processed, model, d_h, d_w, CONFIDENCE_THRESHOLD) # assume returning [dict{}]
    dest_marks_raw = detect_marks_for_slam(dest_processed, model, s_h, s_w, CONFIDENCE_THRESHOLD) # assume returning [dict{}]

    # detection failed: no correction needed, just default to trusting stage steps
    if len(dest_marks_raw) == 0 or len(src_marks_raw) == 0:
        print("Early exit: No markers detected in one or both images")
        return (None, None, None)

    # take centroids of each set of marks
    dest_marks = [mark["center"] for mark in dest_marks_raw]
    src_marks = [mark["center"] for mark in src_marks_raw]

    # match coordinates to closest coordinates -> match-finder alg
    img_h, img_w = camera.shape[1], camera.shape[0]  
    matched_dest, _, matched_src = match_alignment_markers_by_coordinates(dest_marks, src_marks, src_marks, img_h, img_w)
    print("\nmatched_dest_marks", matched_dest)
    print("matched_src_shifted_marks", matched_src)

    # check if we should proceed with error correction
    if len(matched_dest) < 1:
        print(f"Warning: {len(matched_dest)} valid match(es) found, need at least 2 for rotation. Skipping correction.")
        return (0, 0, 0)

    # calculate transform in pixels and rotation
    dx, dy, d0 = estimate_transform(np.array(matched_dest), np.array(matched_src))
    print(f"calculated estimate_transform, {dx}, {dy}, {d0}")
    if len(matched_dest) == 1:
        d0 = 0.0
    if abs(d0) >= 90.0:
        model.create_warning(f"Error: chip is rotated from previous layer. Please correct rotation and re-pattern. Skipping errors")
        return (0, 0, 0)
    
    print(f"error transform result (px): {round(dx)}, {round(dy)}, {d0}")
    return (dx, dy, d0)   

"""    
def do_align() function
    x_amount = self.x_settings.amount_var
    x_offset = int(self.x_settings.offset_var.get())
    x_dir = 1 if x_amount > 0 else -1
    x_amount = abs(x_amount)

    y_amount = self.y_settings.amount_var
    y_offset = int(self.y_settings.offset_var.get())
    y_dir = 1 if y_amount > 0 else -1
    y_amount = abs(y_amount)

    x_start, y_start = self.model.stage_setpoint[0], self.model.stage_setpoint[1]
    print(f"x_start {x_start}, y_start = {y_start}")

    # Move in Snake pattern with left to right on even rows and right to left on odd rows
    for y_idx in range(y_amount):
        if(y_idx %2 == 0):
          for x_idx in range(x_amount):
              pattern_for_tile(self, model, x_start, -x_dir, x_idx, x_offset, y_start, -y_dir, y_idx, y_offset, y_idx_max=y_amount, x_idx_max=x_amount)
              print("Patterned x_idx:" + str(x_idx) + " y_idx: "+str(y_idx))
        else:
            for x_idx in range(x_amount - 1, -1, -1):
              pattern_for_tile(self, model, x_start, -x_dir, x_idx, x_offset, y_start, -y_dir, y_idx, y_offset, y_idx_max=y_amount, x_idx_max=x_amount)
              print("Patterned x_idx:" + str(x_idx) + " y_idx: "+str(y_idx))

def detect_alignment_markers_yolo(yolo_model, image, draw_rectangle=False, edge=None, edge_fraction=0.25):
    #Detects alignment markers and optionally filters detections by image edge(s).
    #yolo_model: YOLO model
    #image: image to detect on 
    #draw_rectangle: If True, draw rectangles
    #edge: 'left', 'right', 'top', or a list like ['left', 'right'] where markers are expected
                                            #none means that markers are expect on all edges
    #edge_fraction: Fraction of width/height considered as edge region
    
    detections = []
    display_image = image.copy()
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (640, 640))
        results = yolo_model(resized)
        boxes = results[0].boxes

        if isinstance(edge, str):
            edge = [edge]  # allow single string or list

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * original_width / 640)
            x2 = int(x2 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            y2 = int(y2 * original_height / 640)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # If edge filtering is enabled
            if edge is not None:
                if 'left' in edge and x_center > original_width * edge_fraction:
                    continue
                if 'right' in edge and x_center < original_width * (1 - edge_fraction):
                    continue
                if 'top' in edge and y_center > original_height * edge_fraction:
                    continue

            detections.append(((x1, y1), (x2, y2)))
            if draw_rectangle:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        print(f"Detected {len(detections)} marker(s)")
    except Exception as e:
        print(f"Detection failed: {e}")

    return detections, display_image
    
def do_align_tiling(edge):
    #edge = ['left', 'right', 'top']
    h, w, _ = model.camera_image.shape

    # Detect markers on the left, right, and top edges
    markers, _ = detect_alignment_markers_yolo(model.model, model.camera_image, edge)
    if len(markers) == 0:
        print("No markers detected.")
        return

    alignment = model.config.alignment
    dx, dy = 0.0, 0.0
    count_x, count_y = 0, 0

    for m in markers:
        xy0, xy1 = m
        x0, y0 = xy0
        x1, y1 = xy1
        x = (x0 + x1) / 2 / w
        y = (y0 + y1) / 2 / h

        # Horizontal alignment (left/right markers)
        if x < 0.5:
            dx += alignment.x_scale_factor * (alignment.left_marker_x / w - x)
            count_x += 1
        elif x > 0.5:
            dx += alignment.x_scale_factor * (alignment.right_marker_x / w - x)
            count_x += 1

        # Vertical alignment (top markers only)
        if y < 0.3:  # top region
            dy += alignment.y_scale_factor * (alignment.top_marker_y / h - y)
            count_y += 1

    # Average corrections based on detected edges
    if count_x > 0:
        dx /= count_x
    if count_y > 0:
        dy /= count_y

    # Move accordingly (if no top markers, dy=0)
    #If a small amount of alignment is needed move the image otherwise move the stage since we have far more percision in moving the image than the stage
    #The con of this is that large movements of the image result in cropping of the image
    #TODO calibrate the stage move threshold
    if(dx < 10 or dy < 10):
        #move the image instead of the stage
        model.set_image_position(dx, dy, t=0)
    else:
        model.move_relative({'x': dx, 'y': dy})
        print(f"Alignment correction: dx={dx:.5f}, dy={dy:.5f} using {len(markers)} markers.")
"""

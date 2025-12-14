#Tiling verisons of alignment
def detect_alignment_markers_tiling(model, image, draw_rectangle=False, edge=None, edge_fraction=0.25):
    #Detects alignment markers and optionally filters detections by image edge(s).
    #model: YOLO model
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
        results = model(resized)
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
    markers, _ = detect_alignment_markers_tiling(model.model, model.camera_image, edge)
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
    model.move_relative({'x': dx, 'y': dy})
    print(f"Alignment correction: dx={dx:.5f}, dy={dy:.5f} using {len(markers)} markers.")
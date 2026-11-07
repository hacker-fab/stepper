import cv2
import numpy as np


def generate_litho_target(
    size: int = 1600,
    main_text: str = "RETICLE CAL #02",
    min_size: float = 1.0,
    scale_ratio: float = 1.14,
) -> np.ndarray:
    """Generates a photolithography calibration mask with geometric feature scaling

    growing outward from the smallest resolution limit.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)

    # -------------------------------------------------------------
    # 1. Header (1/4 of total height, maximized title)
    # -------------------------------------------------------------
    header_h = int(size * 0.25)

    # Frame and divider lines
    cv2.rectangle(canvas, (0, 0), (size, size), 0, 4)
    cv2.line(canvas, (0, header_h), (size, header_h), 0, 3)

    mid_x = size // 2
    cv2.line(canvas, (mid_x, header_h), (mid_x, size), 0, 3)

    # Negative tone right half
    canvas[header_h:size, mid_x:size] = 0

    # Auto-fit main title text (supports multiple lines)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = main_text.split("\n") if main_text else [""]

    target_line_h = int(header_h * 0.5)
    target_total_h = int(header_h * 0.95)
    max_title_w = size - 70

    # Iteratively fit font scale
    scale = 1.0
    thickness = 2
    for _ in range(4):
        sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
        max_tw = max(max((sz[0][0] for sz in sizes), default=1), 1)
        max_th = max(max((sz[0][1] for sz in sizes), default=1), 1)
        max_baseline = max((sz[1] for sz in sizes), default=1)
        line_height = int(max_th * 1.3)
        total_h = (len(lines) - 1) * line_height + max_th + max_baseline

        scale_factors = []
        if max_th > 0:
            scale_factors.append(target_line_h / max_th)
        if max_tw > 0:
            scale_factors.append(max_title_w / max_tw)
        if total_h > 0:
            scale_factors.append(target_total_h / total_h)

        scale *= min(scale_factors)
        thickness = max(1, int(scale * 2.8))

    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    max_th = max(max((sz[0][1] for sz in sizes), default=1), 1)
    max_baseline = max((sz[1] for sz in sizes), default=1)
    line_height = int(max_th * 1.3)
    total_block_h = (len(lines) - 1) * line_height + max_th + max_baseline

    start_top = (header_h - total_block_h) // 2
    title_y_start = start_top + max_th
    title_x = 35

    for i, line in enumerate(lines):
        if line:
            title_y = title_y_start + i * line_height
            cv2.putText(
                canvas,
                line,
                (title_x, title_y),
                font,
                scale,
                0,
                thickness,
                cv2.LINE_AA,
            )

    # -------------------------------------------------------------
    # Vertical Budgeting (remaining 75% height)
    # -------------------------------------------------------------
    work_h = size - header_h
    h_strip_h = int(size * 0.25)
    circle_h = int(size * 0.25)
    pad_h = work_h - h_strip_h - circle_h

    y_strip_start = header_h
    y_circle_start = y_strip_start + h_strip_h
    y_pad_start = y_circle_start + circle_h

    # -------------------------------------------------------------
    # Section A: Horizontal & Vertical Gratings
    # -------------------------------------------------------------
    def draw_gratings(x_offset: int, is_negative: bool):
        fg = 255 if is_negative else 0
        avail_h = h_strip_h
        sub_w = mid_x // 2
        # Draw from bottom upward so smallest feature is at bottom edge
        curr_w = min_size
        curr_y = y_strip_start + avail_h
        
        while curr_y - curr_w > y_strip_start:
            cv2.rectangle(
                canvas,
                (x_offset, curr_y),
                (x_offset + sub_w, curr_y - (int(curr_w - 1))),
                fg,
                -1,
            )
            curr_y -= 2 * int(curr_w)
            curr_w *= scale_ratio


        curr_x = x_offset + sub_w + sub_w
        curr_w = min_size
        while curr_x - 2 * int(curr_w) > x_offset + sub_w:
            cv2.rectangle(
                canvas,
                (curr_x, y_strip_start),
                (curr_x - int(curr_w) + 1, y_strip_start + avail_h),
                fg,
                -1,
            )
            curr_x -= 2 * int(curr_w)
            curr_w *= scale_ratio

    draw_gratings(0, is_negative=False)
    draw_gratings(mid_x, is_negative=True)

    # -------------------------------------------------------------
    # Section B: Concentric Rings & Siemens Star
    # -------------------------------------------------------------
    def draw_radials(x_offset, is_negative):
        fg = 255 if is_negative else 0
        bg = 0 if is_negative else 255

        sub_w = mid_x // 2
        cy = y_circle_start + (circle_h // 2)
        max_r = min(sub_w, circle_h) // 2 - 12

        # 1. Concentric rings expanding outward from center (smallest to largest)
        cx_ring = x_offset + sub_w // 2
        curr_w = min_size
        curr_r = 0
        rings = []  # list of (r_inner, r_outer)

        while True:
            w = int(curr_w)
            r_outer = curr_r + w
            if r_outer > max_r:
                break
            rings.append((curr_r, r_outer))
            curr_r = r_outer + w
            curr_w *= scale_ratio

        # Fill circles from largest down to smallest so they nest cleanly
        for r_inner, r_outer in reversed(rings):
            cv2.circle(canvas, (cx_ring, cy), r_outer, fg, -1)
            if r_inner > 0:
                cv2.circle(canvas, (cx_ring, cy), r_inner, bg, -1)

        # 2. Siemens Star
        cx_star = x_offset + sub_w + (sub_w // 2)
        num_spokes = 32
        d_angle = 360.0 / num_spokes

        for i in range(0, num_spokes, 2):
            cv2.ellipse(
                canvas,
                (cx_star, cy),
                (max_r, max_r),
                0,
                i * d_angle,
                (i + 1) * d_angle,
                fg,
                -1,
            )

    draw_radials(0, is_negative=False)
    draw_radials(mid_x, is_negative=True)

    # -------------------------------------------------------------
    # Section C: Multi-row Centered Contact Squares & Circular Dots
    # -------------------------------------------------------------
    def draw_pad_array(x_offset, is_negative):
        fg = 255 if is_negative else 0
        left_bound = x_offset + 20
        right_bound = x_offset + mid_x - 20

        # Grow pads geometrically until right boundary is reached (smallest -> largest)
        pad_sizes = []
        x_positions = []
        curr_pad = float(min_size)
        curr_x = left_bound

        while True:
            p = int(curr_pad)
            spacing = p
            if curr_x + p > right_bound:
                break
            pad_sizes.append(p)
            x_positions.append(curr_x)
            curr_x += p + spacing
            curr_pad *= scale_ratio

        max_pad = max(pad_sizes) if pad_sizes else 10
        half_pad_h = pad_h // 2

        # Square contact rows (centered in upper sub-band)
        row_step = max(max_pad + 6, 16)
        num_rows = max(1, (half_pad_h - 12) // row_step)
        block_h = (num_rows - 1) * row_step + max_pad
        start_sq_y = y_pad_start + (half_pad_h - block_h) // 2

        for r in range(num_rows):
            y_base = start_sq_y + r * row_step
            for p, xp in zip(pad_sizes, x_positions):
                # Center each square vertically on row centerline
                y_pos = y_base + (max_pad - p) // 2
                cv2.rectangle(canvas, (xp, y_pos), (xp + p - 1, y_pos + p - 1), fg, -1)

        # Circular dot rows (centered in lower sub-band)
        start_circ_y = (
            y_pad_start + half_pad_h + (half_pad_h - block_h) // 2
        )
        for r in range(num_rows):
            row_cy = start_circ_y + r * row_step + (max_pad // 2)
            for p, xp in zip(pad_sizes, x_positions):
                if p == 1:
                    cv2.rectangle(canvas, (xp, row_cy), (xp, row_cy), fg, -1)
                elif p == 2:
                    cv2.rectangle(canvas, (xp, row_cy - 1), (xp + 1, row_cy), fg, -1)
                else:
                    rad = p // 2
                    cx = xp + rad
                    cv2.circle(canvas, (cx, row_cy), rad, fg, -1)

    draw_pad_array(0, is_negative=False)
    draw_pad_array(mid_x, is_negative=True)

    return canvas


if __name__ == "__main__":
    # Test execution: min_size=1 px, scale_ratio=1.14 (grows smoothly until fit)
    target = generate_litho_target(
        size=1600,
        main_text="16.12S | 0.1um",
        min_size=1.0,
        scale_ratio=1.1,
    )
    cv2.imwrite("litho_target_v4.png", target)

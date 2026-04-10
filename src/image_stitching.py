from gui import EventDispatcher
from tkinter import ttk
from PIL import Image, ImageTk
from dataclasses import dataclass
from datetime import datetime
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class ImageCaptureSettings:
    stride_x_um: int
    stride_y_um: int
    capture_folder: str

@dataclass
class ImageStitchSettings:
    num_rows: int
    num_cols: int
    output_folder: str
    resize: float
    debug: bool

@dataclass
class TilePreprocessSettings:
    crop_margin_x_px: int
    crop_margin_y_px: int
    gaussian_kernel_size: tuple[int, int]

class ImageStitchingFrame:
    def __init__(self, parent, event_dispatcher: EventDispatcher):
        self.frame = ttk.Frame(parent)
        self.event_dispatcher = event_dispatcher

        self.capture_button = ttk.Button(
            self.frame, 
            text="Capture & Stitch", 
            command=self.capture_and_stitch
        )
        self.capture_button.grid(row=0, column=0)

        self.preview_label = ttk.Label(self.frame)
        self.preview_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)

    def capture_and_stitch(self):
        self.img = self.event_dispatcher.pattern_image

        self.img_w_px, self.img_h_px = self.img.size
        print(f"image width: {self.img_w}, image height: {self.img_h}")

        # projection size
        self.projection_width_um, self.projection_height_um = 1037, 583

        # tile size
        self.tile_width_px, self.tile_height_px = 3840, 2160

        # camera capture size
        self.snapshot_width_px, self.snapshot_height_px = 1920, 1080

        # overlay ratio that is used for alignment
        # currently we need to move half-sized width and height
        # so that there are enough features for alignment purposes
        self.overlay_ratio = 0.5
        self.stride_x_um = self.projection_width_um * (1 - self.overlay_ratio)
        self.stride_y_um = self.projection_height_um * (1 - self.overlay_ratio)

        # used for cropping out dark edges in camera capture
        self.crop_margin_x_px, self.crop_margin_y_px = 150, 150

        self.capture_button.config(state='disabled', text="Capturing...")
        self.frame.update()
        
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_folder = f"data_collection_{curr_time}/"

        captures, num_rows, num_cols = self.capture_helper(
            settings=ImageCaptureSettings(
                stride_x_um=self.stride_x_um,
                stride_y_um=self.stride_y_um,
                capture_folder=capture_folder
            )
        )

        preprocessed_imgs = self.preprocess_images(captures, settings=TilePreprocessSettings(
            crop_margin_x_px=self.crop_margin_x_px,
            crop_margin_y_px=self.crop_margin_y_px,
            gaussian_kernel_size=(5,5)
        ))

        stitched_image = self.stitch_helper(
            preprocessed_imgs,
            settings=ImageStitchSettings(
                num_rows=num_rows,
                num_cols=num_cols,
                output_folder=capture_folder,
                resize=0.2,
                debug=False
            )
        )
        
        if stitched_image:
            self.display_image(stitched_image)
            print("stitching complete!")
        else:
            print("failed to stitch images")
            
        self.capture_button.config(state='normal', text="Capture & Stitch Chip Imges")

    def display_image(self, pil_image):
        display_img = pil_image.copy()
        photo = ImageTk.PhotoImage(display_img)
        self.preview_label.config(image=photo)
        self.preview_label.image = photo

    def capture_current_image(self):
        # Get the camera view from the event dispatcher
        if hasattr(self.event_dispatcher, 'camera_image') and self.event_dispatcher.camera_image is not None:
            camera_image = self.event_dispatcher.camera_image
            pil_image = Image.fromarray(camera_image)
            return pil_image
        else:
            print("No camera image available")
            return None

    def capture_helper(self, settings: ImageCaptureSettings):
        """
        Take snapshots num_cols * num_rows times, move in snake pattern
        Move in stride_x and y um in distance
        Crop margins off to account for dark margins in camera snapshot
        """

        captured_imgs = []

        total_x_um = (self.img_w_px  / self.tile_width_px ) * self.projection_width_um
        total_y_um = (self.img_h_px / self.tile_height_px) * self.projection_height_um
    
        num_cols = int(total_x_um // settings.stride_x_um)
        num_rows = int(total_y_um // settings.stride_y_um)
        if num_cols * settings.stride_x_um < total_x_um:
            num_cols += 1
        if num_rows * settings.stride_y_um < total_y_um:
            num_rows += 1
        print("starting capture...")
        print("num_cols, num_rows: ", num_cols, num_rows)
        
        # get stage positions (um)
        orig_x, orig_y, orig_z = self.event_dispatcher.stage_setpoint
        start_x = orig_x
        start_y = orig_y

        # create folder to save captures, info and log file
        os.mkdir(settings.capture_folder)
        log_file_path = os.path.join(settings.capture_folder, "log.txt")
        log_file = open(log_file_path, "w")
        log_file.write(json.dumps({"rows": num_rows, "cols": num_cols}))

        # move in snake pattern with left to right on even rows
        # and right to left on odd rows
        for row in range(num_rows):
            row_imgs = []
            current_y = start_y + row * settings.stride_y_um

            if row % 2 == 0:
                col_range = range(num_cols)
                first_x = start_x
            else:
                col_range = range(num_cols - 1, -1, -1)
                first_x = start_x + (num_cols - 1) * settings.stride_x_um

            # move to the next row
            self.event_dispatcher.move_absolute({
                "x": first_x,
                "y": current_y,
                "z": orig_z
            })
            self.event_dispatcher.non_blocking_delay(2)
            
            # columns in this row
            for idx, col in enumerate(col_range):
                current_x = start_x + col * settings.stride_x_um
                if idx > 0: # idx = 0 first one don't need to move
                    self.event_dispatcher.move_absolute({
                        "x": current_x,
                        "y": current_y,
                        "z": orig_z
                    })
                    self.event_dispatcher.non_blocking_delay(2.5)
                
                captured_img = self.capture_current_image()
                row_imgs.append(captured_img)

                # save capture and write to log file
                tile_file = f"tile_{row}_{col}.png"
                tile_path = os.path.join(settings.capture_folder, tile_file)
                captured_img.save(tile_path)
                log_file.write(f"{tile_file}: x={current_x}, y={current_y}\n")
                
                self.event_dispatcher.non_blocking_delay(0.5)
                self.frame.update()

            captured_imgs.append(row_imgs)
        
        # Return to starting position
        self.event_dispatcher.move_absolute({
            "x": orig_x,
            "y": orig_y,
            "z": orig_z
        })
        self.event_dispatcher.non_blocking_delay(0.5)

        return captured_imgs, num_rows, num_cols

    def preprocess_image(self, img, settings: TilePreprocessSettings):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, settings.gaussian_kernel_size, 0)
        
        # remove margins in camera view
        h, w = img.shape[:2]
        margin_h = settings.crop_margin_y_px
        margin_w = settings.crop_margin_x_px
        img = self.crop_image(img, margin_h, h - margin_h, margin_w, w - margin_w)
        return img

    def preprocess_images(self, imgs, settings: TilePreprocessSettings):
        result = []
        for row in len(imgs):
            row_imgs = []
            for col in len(imgs[row]):
                row_imgs.append(self.preprocess_image(imgs[row][col]))
            result.append(row_imgs)
        return result
        
    def crop_image(img, h_start, h_end, w_start, w_end):
        return img[h_start:h_end, w_start:w_end]

    def image_alignment(dst_img, src_img, display=False):
        sift = cv2.SIFT_create(
            contrastThreshold=0.04,
            edgeThreshold=10
        )
        src_keypoints, src_descriptors = sift.detectAndCompute(src_img, None)
        dst_keypoints, dst_descriptors = sift.detectAndCompute(dst_img, None)

        # Use kdtrees to find nearest neighbors
        # trees: neighborhood size
        # checks: more checks → searches more of the trees → more accurate matches
        FLANN_INDEX_KDTREE = 1
        NUM_TREES=100
        NUM_CHECKS=100
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
        search_params = dict(checks=NUM_CHECKS)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # gets the two best matches
        matches = flann.knnMatch(src_descriptors, dst_descriptors, k=2)

        good = []
        for m,n in matches:
            # only keep the best match if it is significantly better than the 2nd best match
            if m.distance < 0.8 * n.distance:
                good.append(m)

        # If we get lower than MIN_MATCH_COUNT matches
        # it is probably not a good match -> abort
        MIN_MATCH_COUNT = 4
        if len(good) < MIN_MATCH_COUNT:
            print(f"not enough matches were found: {len(good)} < {MIN_MATCH_COUNT}")
            matchesMask = None
            return

        src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Find homography M that transforms src_pts to dst_pts
        # dst_pts = M * src_pts
        # mask: Nx1 array --> 1: inlier, 0: outlier
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=3, maxIters=2000, confidence=0.99, refineIters=10)

        matchesMask = mask.ravel().tolist()

        ################## Evaluation ##################

        M = np.vstack([M, [0, 0, 1]])
        print(M)

        # inlier ratio
        num_inliers = sum(matchesMask)
        inlier_ratio = num_inliers / len(matchesMask)
        print(f"Inlier ratio: {inlier_ratio}")

        # apply homography on src points and calculate distance to dst points
        # only considering inliers
        src_pts = cv2.perspectiveTransform(src_pts, M)
        error = 0
        for i in range(0, len(matchesMask)):
            if matchesMask[i] == 1: # inlier
                [delta_x, delta_y] = src_pts[i][0] - dst_pts[i][0]
                error += (np.pow(delta_x, 2) + np.pow(delta_y, 2))
        error /= num_inliers
        error = np.sqrt(error)
        print(f"RMS error between inliers={error}")

        if display:
            # show boundary of src on dst after homography
            h,w = src_img.shape[:2]
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            dst = np.int32(dst).reshape((-1, 1, 2))
            dst_img = cv2.polylines(dst_img, [np.int32(dst)], True, 255, 8, cv2.LINE_AA)

            # draw matches
            draw_params = dict(matchColor = None, # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            img3 = cv2.drawMatches(src_img, src_keypoints, dst_img, dst_keypoints, good, None, **draw_params)
            plt.imshow(img3, 'gray')
            plt.show()

        return (M, error)

    def stitch_helper(self, imgs, settings: ImageStitchSettings):
        # stich images in a snake like pattern
        rows = settings.num_rows
        cols = settings.num_cols
        curr_tile = None
        next_tile = None
        positions = [[0] * cols for _ in range(rows)]
        curr_pos = (0, 0)

        # snake pattern stitching
        for row in range(0, rows):
            col_range = range(0, cols) if row % 2 == 0 else range(cols-1, -1, -1)
            
            for col in col_range:
                if row == 0 and col == 0:
                    curr_tile_info = (row, col)
                    curr_tile = imgs[0][0]
                    positions[row][col] = curr_pos
                    continue

                next_tile = imgs[row][col]
                next_tile_info = (row, col)

                print(f"curr_tile_info: {curr_tile_info}, next_tile_info: {next_tile_info}")
                M, error = self.image_alignment(curr_tile, next_tile)
                dx = M[0][2]
                dy = M[1][2]
                x, y = curr_pos
                curr_pos = (x + dx, y + dy)
                positions[row][col] = curr_pos
                print(f"should move dx={dx}, dy={dy}")

                curr_tile = next_tile

        if settings.debug:
            for row in range(0, rows):
                for col in range(0, cols):
                    print(f"row: {row}, col: {col}, position:{positions[row][col]}")

        xs = []
        ys = []

        for row in range(0, rows):
            for col in range(0, cols):
                x, y = positions[row][col]
                h, w = imgs[row][col].shape[:2]

                xs.append(x + w)
                xs.append(x)
                ys.append(y + h)
                ys.append(y)

        canvas_w = int(max(xs) - min(xs))
        canvas_h = int(max(ys) - min(ys))

        print(f"canvas_w={canvas_w}, canvas_h={canvas_h}")

        shift_w = int(min(xs))
        shift_h = int(min(ys))

        print(f"shift_w={shift_w}, shift_h={shift_h}")
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                img = imgs[row][col]
                h, w = img.shape[:2]
                x, y = positions[row][col]
                x = int(x) - shift_w
                y = int(y) - shift_h
                # shift into canvas coords
                canvas[y:y+h, x:x+w] = img

        # resize and output
        canvas = cv2.resize(canvas, None, fx=settings.resize, fy=settings.resize)
        output_path = os.path.join(settings.output_folder, "output.png")
        cv2.imwrite(output_path, canvas)

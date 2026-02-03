import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def get_overlays(img, overlay_width=1000):
    h, w = img.shape
    print(f"size of image: H={h}, W={w}")
    top_overlay = img[0:overlay_width, :]
    left_overlay = img[:, 0:overlay_width]
    right_overlay = img[:, w-overlay_width:]
    return {"top": top_overlay, "left": left_overlay, "right": right_overlay}

def read_images(img_file, src_file):
    src_path = os.path.join("./src/litho_captures/", src_file)
    img_path = os.path.join("./src/litho_captures/", img_file)

    src_img = cv.imread(src_path)
    dst_img = cv.imread(img_path)
    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    dst_img = cv.cvtColor(dst_img, cv.COLOR_BGR2GRAY)
    return (src_img, dst_img)

def image_alignment(dst_img, src_img, display=False):

    # SIFT feature detection and descriptor calculation
    sift = cv.SIFT_create()
    src_keypoints, src_descriptors = sift.detectAndCompute(src_img, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dst_img, None)

    # Use kdtrees to find nearest neighbors
    # trees: neighborhood size
    # checks: more checks → searches more of the trees → more accurate matches
    FLANN_INDEX_KDTREE = 1
    NUM_TREES=20
    NUM_CHECKS=50
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
    search_params = dict(checks=NUM_CHECKS)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # gets the two best matches
    matches = flann.knnMatch(src_descriptors, dst_descriptors, k=2)

    good = []
    for m,n in matches:
        # only keep the best match if it is significantly better than the 2nd best match
        if m.distance < 0.4 * n.distance:
            good.append(m)

    # If we get lower than MIN_MATCH_COUNT matches
    # it is probably not a good match -> abort
    MIN_MATCH_COUNT = 10
    if len(good) < MIN_MATCH_COUNT:
        print(f"not enough matches were found: {len(good)} < {MIN_MATCH_COUNT}")
        matchesMask = None
        return

    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Find homography M that transforms src_pts to dst_pts
    # dst_pts = M * src_pts
    # mask: Nx1 array --> 1: inlier, 0: outlier
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    ################## Evaluation ##################

    # inlier ratio
    num_inliers = sum(matchesMask)
    inlier_ratio = num_inliers / len(matchesMask)
    print(f"Inlier ratio: {inlier_ratio}")

    # apply homography on src points and calculate distance to dst points
    # only considering inliers

    src_pts = cv.perspectiveTransform(src_pts, M)
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
        h,w = src_img.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)
        dst_img = cv.polylines(dst_img, [np.int32(dst)], True, 255,3, cv.LINE_AA)

        # draw matches
        draw_params = dict(matchColor = None, # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        img3 = cv.drawMatches(src_img, src_keypoints, dst_img, dst_keypoints, good, None, **draw_params)
        plt.imshow(img3, 'gray')
        plt.show()

if __name__ == "__main__":
    dst_file = "output_2025-02-06_20-29-57.png"
    src_file = "output_2025-02-06_20-30-00.png"
    dst_img, src_img = read_images(dst_file, src_file)
    overlays = get_overlays(src_img)
    rotated_overlay = cv.rotate(overlays["top"], cv.ROTATE_90_CLOCKWISE)
    image_alignment(dst_img, rotated_overlay, True)
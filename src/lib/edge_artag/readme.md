

### 1. OpenCV-Compatible Calling Interface

The library provides both a functional interface matching OpenCV's `cv2.aruco.detectMarkers` and an object-oriented class `EdgeArucoDetector`:

```python
import cv2
import edge_artag as ea

# 1. Load image and dictionary
image = cv2.imread("manual_snapshot_2026-10-20_09-42-09.png")
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

# 2. (Optional) Custom detector parameters
parameters = ea.EdgeDetectorParameters()

# 3. Detect markers (same return signature as OpenCV):
corners, ids, rejected = ea.detect_edge_markers(image, dictionary, parameters)

# 4. Directly compatible with standard OpenCV drawing utilities:
cv2.aruco.drawDetectedMarkers(image, corners, ids)
```

**Returned Data Shapes:**
- `corners`: List of $N$ float32 numpy arrays of shape `(1, 4, 2)` representing `[Top-Left, Top-Right, Bottom-Right, Bottom-Left]` in the **original input image coordinate system**.
- `ids`: `(N, 1)` int32 numpy array of detected tag IDs (or `None` if none found).
- `rejected`: List of candidate quad corner arrays `(1, 4, 2)` that were rejected (e.g., outer $8 \times 8$ decorative frames or misaligned contours).

---

### 2. Parameters & Rules of Thumb

All parameters are configured via [`EdgeDetectorParameters`](file:///home/toby/Downloads/lith-artag/edge_artag.py#L17-L80):

#### Geometry & Scale Reference
For an ArUco marker with dimension $N \times N$ (e.g. $4 \times 4$ payload has $4 + 2 = 6$ cells per side including the 1-cell black border):
$$\text{cell\_size} = \frac{\text{marker\_side\_length}}{N + 2}$$
In these microscope images, $\text{marker\_side\_length} \approx 126\text{ px}$, so $\text{cell\_size} \approx 21\text{ px}$, and the etched boundary stroke width $w \approx 4\text{--}5\text{ px}$.

| Parameter | Default | Meaning & Rule of Thumb |
| :--- | :---: | :--- |
| `adaptive_thresh_fine_block_size` | `15` | **Fine scale adaptive filter window.**<br>• *Rule of thumb*: **$0.5\times$ to $0.8\times$ of `cell_size`** (e.g., $11\text{--}17\text{ px}$ for a $21\text{ px}$ cell).<br>• Acts as a narrow high-pass filter tuned to the $3\text{--}5\text{ px}$ dark boundary stroke, completely suppressing broad silicon wafer interference fringes. |
| `adaptive_thresh_fine_C` | `7.0` | **Contrast threshold for fine edges.**<br>• Controls the minimum intensity drop below the local Gaussian mean required to count as an edge ($6\text{--}9$ typical). |
| `adaptive_thresh_wide_block_size` | `61` | **Wide scale background window.**<br>• *Rule of thumb*: **$2.5\times$ to $3.5\times$ of `cell_size`** (e.g., $51\text{--}65\text{ px}$ for a $21\text{ px}$ cell).<br>• Measures baseline illumination over the whole marker, preventing the bright phase-contrast halos around isolated $1\times 1$ cells from producing false dark spots at cell centers. |
| `adaptive_thresh_wide_C` | `14.0` | **Contrast threshold for broad background rejection.** |
| `grid_center_ratio` | `0.50` | **Fraction of grid cell width/height checked at the center.**<br>• `0.50` specifies the middle $50\%$ ($[0.25, 0.75] \times [0.25, 0.75]$) of each cell. |
| `max_center_edge_density` | `0.16` | **Max allowable edge pixel percentage in the 50% center box.**<br>• *Rule of thumb*: **$0.10$ to $0.18$**.<br>• True marker cell centers have $0.00\text{--}0.05$ density. Outer decorative frames or misaligned squares cut directly through the center boxes ($0.30\text{--}0.65$ density) and are immediately rejected. |
| `neighbor_edge_thresh` | `35.0` | **Threshold to detect a color change between neighboring cells.**<br>• *Rule of thumb*: $25\text{--}45$ for $0\text{--}255$ binary edge response along shared cell boundaries. |
| `marker_border_scale` | `1.0` | **Scale factor applied to candidate quad corners relative to center.**<br>• *Rule of thumb*: Keep `1.0` for thin/standard border strokes; set to `(inner_grid_pitch * total_cells) / snapped_outer_width` (typically $0.95\text{--}0.97$) for thick etched borders / lithography etch bias. |
| `enable_subpixel_refinement` | `True` | **Enables 8-DOF Gauss-Newton subpixel homography optimization.**<br>• Samples perpendicular 1D bicubic grayscale profiles along both outer square sides and internal tag edges, fitting parabolic valleys and updating the homography via Huber IRLS ($< 0.01\text{ px}$ convergence, $\sim 0.25\text{ px}$ edge residual). |


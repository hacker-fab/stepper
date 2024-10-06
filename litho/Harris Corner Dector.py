import cv2
import numpy as np

def detect_corners_and_orientation(image_path, output_image_path, threshold=0.01):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image was loaded
    if image is None:
        print("Error: Unable to load image")
        return None
    
    # Detect corners using the Harris Corner Detector
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Normalize result to identify strong corners
    dst = cv2.dilate(dst, None)  # Dilates corner points to make them more visible
    corners = np.where(dst > threshold * dst.max())
    
    # Get image center
    image_height, image_width = image.shape
    center_x, center_y = image_width // 2, image_height // 2
    
    orientations = []
    
    # Convert grayscale image to BGR to plot colored corners
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(corners[0])):
        corner_x = corners[1][i]
        corner_y = corners[0][i]
        
        # Determine corner orientation
        if corner_x > center_x and corner_y < center_y:
            orientation = "Upper Right"
        elif corner_x < center_x and corner_y < center_y:
            orientation = "Upper Left"
        elif corner_x > center_x and corner_y > center_y:
            orientation = "Lower Right"
        elif corner_x < center_x and corner_y > center_y:
            orientation = "Lower Left"
        else:
            orientation = "Center"
        
        orientations.append((corner_x, corner_y, orientation))
        
        # Draw circle on detected corners (in red)
        cv2.circle(image_with_corners, (corner_x, corner_y), 5, (0, 0, 255), 1)
    
    # Save the grayscale image with the detected corners highlighted
    cv2.imwrite(output_image_path, image_with_corners)
    
    print(f"Corners with orientations saved to {output_image_path}")
    
    # Return detected corners and their orientation
    return orientations

# Example usage
image_path = 'your_image_path_here.jpg'
output_image_path = 'output_with_corners.jpg'
corners_with_orientation = detect_corners_and_orientation(image_path, output_image_path)

# Print detected corners and their orientations
if corners_with_orientation:
    for corner in corners_with_orientation:
        print(f"Corner at (x={corner[0]}, y={corner[1]}) with orientation: {corner[2]}")

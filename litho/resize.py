import cv2
import json
from pathlib import Path
import numpy as np

def resize_with_metadata(input_dir: str, output_dir: str, target_size: tuple[int,int] = (640,640)):
    """
    Resize images and save metadata about original dimensions.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_files = list(Path(input_dir).glob('*.png'))
    
    # Dictionary to store original dimensions
    dimensions_map = {}
    
    for i, input_path in enumerate(input_files, 1):
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Failed to read {input_path}")
            continue
        
        # Store original dimensions
        orig_h, orig_w = img.shape[:2]
        dimensions_map[input_path.name] = {
            "original_height": orig_h,
            "original_width": orig_w
        }
        
        # Resize image
        resized = cv2.resize(img, target_size)
        
        # Save resized image
        output_path = Path(output_dir) / input_path.name
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        cv2.imwrite(str(output_path), resized, encode_params)
        
        print(f"Processed {i}/{len(input_files)}: {input_path.name}")
    
    # Save dimensions metadata
    metadata_path = Path(output_dir) / "image_dimensions.json"
    with open(metadata_path, 'w') as f:
        json.dump(dimensions_map, f, indent=2)

def scale_detection_to_original(detection, original_width, original_height, model_width=640, model_height=640):
    """
    Scale YOLO detection coordinates from model size back to original image size.
    
    Args:
        detection: YOLO detection (x, y, w, h) in normalized coordinates (0-1)
        original_width: Width of original camera image
        original_height: Height of original camera image
        model_width: Width used for model input (default 640)
        model_height: Height used for model input (default 640)
    """
    x, y, w, h = detection  # normalized coordinates
    
    # Scale back to original dimensions
    orig_x = x * original_width
    orig_y = y * original_height
    orig_w = w * original_width
    orig_h = h * original_height
    
    return orig_x, orig_y, orig_w, orig_h

# Example usage during inference:
def process_camera_frame(frame):
    # Get original dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Resize for model
    model_input = cv2.resize(frame, (640, 640))
    
    # Get predictions from model (this is pseudocode - use actual YOLO inference)
    predictions = model.predict(model_input)
    
    # Scale each detection back to original size
    original_detections = []
    for det in predictions:
        orig_x, orig_y, orig_w, orig_h = scale_detection_to_original(
            det, 
            original_width=orig_width, 
            original_height=orig_height
        )
        original_detections.append((orig_x, orig_y, orig_w, orig_h))
    
    return original_detections

if __name__ == "__main__":
    INPUT_DIR = "original_images"
    OUTPUT_DIR = "resized_images"
    
    resize_with_metadata(INPUT_DIR, OUTPUT_DIR)
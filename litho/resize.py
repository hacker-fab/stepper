import cv2
import numpy as np
from pathlib import Path

def resize_with_padding(input_dir: str, output_dir: str, target_size: tuple[int,int] = (640,640)):
    """
    Resize images maintaining aspect ratio and pad to square.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_files = list(Path(input_dir).glob('*.png'))
    
    print(f"Found {len(input_files)} PNG files to process")
    
    for i, input_path in enumerate(input_files, 1):
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Failed to read {input_path}")
            continue
        
        # Calculate scaling factor to fit within target size
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        
        # Resize maintaining aspect ratio
        new_size = (int(w*scale), int(h*scale))
        resized = cv2.resize(img, new_size)
        
        # Create black square canvas
        square = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (target_size[1] - new_size[1]) // 2
        x_offset = (target_size[0] - new_size[0]) // 2
        
        # Place resized image in center
        square[
            y_offset:y_offset+new_size[1],
            x_offset:x_offset+new_size[0]
        ] = resized
        
        # Save
        output_path = Path(output_dir) / input_path.name
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        cv2.imwrite(str(output_path), square, encode_params)
        
        print(f"Processed {i}/{len(input_files)}: {input_path.name}")

if __name__ == "__main__":
    INPUT_DIR = "original_images"
    OUTPUT_DIR = "resized_images"
    
    resize_with_padding(INPUT_DIR, OUTPUT_DIR)
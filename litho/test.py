import numpy as np
import time
import cv2

def find_edges_2(img):
  threshold = 50
  ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (250, 200))

  (height, width) = img.shape

  # Crop to the area we have lighting at
  img = img[50:height-50, 50:width-50]
  
  cv2.imshow("corped image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  (height, width) = img.shape
  top_half = img[:height // 2, :]
  bottom_half = img[height // 2:, :]

  # Vertical half (left or right)
  left_half = img[:, :width // 2]
  right_half = img[:, width // 2:]

  brightness_top = np.mean(top_half)
  brightness_bottom = np.mean(bottom_half)
  brightness_left = np.mean(left_half)
  brightness_right = np.mean(right_half)
  h_edge = None
  v_edge = None
  print(f'Vertical Difference {abs(brightness_top-brightness_bottom)}')
  print(f'Horizontal Difference {abs(brightness_left-brightness_right)}')
  if abs(brightness_top-brightness_bottom) > threshold:
    h_edge = True
  
  if abs(brightness_left - brightness_right) > threshold:
    v_edge = True
  return (h_edge, v_edge)

if __name__ =='__main__':
  image_path = "output_2024-11-26_16-22-57.png"
  image = cv2.imread(image_path)
  output = find_edges_2(image)
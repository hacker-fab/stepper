import numpy as np
import cv2

def find_edges(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img = cv2.resize(img, (250, 200))

    (height, width) = img.shape

    # Crop to the area we have lighting at
    img = img[50:height-50, 50:width-50]
    cv2.imshow("corpped image", img)
    cv2.waitKey(0)

    (height, width) = img.shape

    # obtain halves of image
    mid_height = height//2
    mid_width = width//2
    top_half = img[:mid_height, :]
    bot_half = img[mid_height:, :]
    left_half = img[:, :mid_width]
    right_half = img[:, mid_width:]
    
    top_b = np.mean(top_half)
    bot_b = np.mean(bot_half)
    left_b = np.mean(left_half)
    right_b = np.mean(right_half)
    
    h_diff = abs(top_b - bot_b)
    v_diff = abs(left_b - right_b)

    print(f'h_diff:{h_diff}, v_diff:{v_diff}')
    return
if __name__ == '__main__':
    img_path = "/home/louis/project/stepper/litho/output_2024-11-26_16-39-51.png"
    img = cv2.imread(img_path)
    find_edges(img=img)
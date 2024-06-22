# sudo modprobe v4l2loopback
# scrcpy --v4l2-sink=/dev/video4 --no-video-playback --video-source=camera --camera-size=2560x1920  --camera-facing=front
# scrcpy --v4l2-sink=/dev/video4 --no-video-playback --video-source=camera --camera-id=3 --camera-size=640x480
import numpy as np
import sys
import cv2

# Get Image From Camera
camera_port=4
camera=cv2.VideoCapture(camera_port) #this makes a web cam object

def get_img(buffer):
	global camera
	_, im = camera.read()
	img = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
	buffer[:, :] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return 1


_, im = camera.read()
print(np.repeat(np.repeat(im, 1), 2).shape)
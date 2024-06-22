# %%
import sys
import zmq
import numpy as np
import cv2
import time
import msgpack
port = "5556"

# %% ZMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
# socket.connect ("tcp://10.193.10.141:%s" % port)
socket.connect ("tcp://127.0.0.1:%s" % port)
socket.subscribe("")
socket.setsockopt(zmq.CONFLATE, 1)

# %% OpenCV
try:
	# Returns True if OpenCL is present
	ocl = cv2.ocl.haveOpenCL()
	# Prints whether OpenCL is present
	print("OpenCL Supported?: ", end='')
	print(ocl)
	print()
	# Enables use of OpenCL by OpenCV if present
	if ocl == True:
		print('Now enabling OpenCL support')
		cv2.ocl.setUseOpenCL(False)
		print("Has OpenCL been Enabled?: ", end='')
		print(cv2.ocl.useOpenCL())

except cv2.error as e:
	print('Error using OpenCL')

# %%

while True:
	data = socket.recv()
	t, x, y, theta, img1, img2 = msgpack.unpackb(data, raw=False)
	img1 = np.array(img1).astype(np.uint8)
	img2 = np.array(img2).astype(np.uint8)
	alpha = 0.3
	# blended_image = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
	blended_image = img2
	# 3x zoom
	blended_image = cv2.resize(blended_image, (0,0), fx=3, fy=3)
	cv2.namedWindow('image')
	cv2.imshow('image',blended_image)
	cv2.waitKey(1)

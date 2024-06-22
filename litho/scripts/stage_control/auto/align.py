import cv2
import numpy as np
import zmq
import msgpack
import struct

SEND_CAMERA_IMAGE_TO_GUI = True

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
		cv2.ocl.setUseOpenCL(True)
		print("Has OpenCL been Enabled?: ", end='')
		print(cv2.ocl.useOpenCL())
except cv2.error as e:
	print('Error using OpenCL')

# Perform Alignment
def align(livimg, refimg, annoimg):
	liveimg_U = cv2.UMat(livimg)

	if SEND_CAMERA_IMAGE_TO_GUI:
		# give live image to GUI and get displaced image for alignment
		refimg = exchange_images(livimg)

	h, w = refimg.shape
	res = cv2.matchTemplate(refimg, liveimg_U, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	dx, dy = (top_left[0], top_left[1])

	annoimg[:, :] = livimg
	cv2.rectangle(annoimg[:, :],top_left, bottom_right, 255, 5)
	return [dy, dx]

# if communicating with GUI, initialize ZMQ communication
# See https://zguide.zeromq.org/docs and https://pyzmq.readthedocs.io/en/latest/api/zmq.html
if SEND_CAMERA_IMAGE_TO_GUI:
	JULIA_PORT = "5555"
	zmq_context = zmq.Context()

	py_socket = zmq_context.socket(zmq.PUB)
	py_socket.bind("tcp://*:%s" % JULIA_PORT)
	print("Julia-side publisher initialized")

# cleanup function for graceful program exit
def cleanup():
	py_socket.close()
	zmq_context.destroy()

def exchange_images(live_image):
	py_socket.send_pyobj(np.transpose(live_image))

	# TODO: get displaced image from GUI/alignment/pattern logic
	displaced_image = live_image
	return displaced_image

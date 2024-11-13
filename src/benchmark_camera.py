# J. Kent Wirant
# Hacker Fab
# Camera FPS benchmark script

# This is a standalone script and is not designed for use within the larger Stepper software framework.
# For performance evaluation, this script reports the framerate of a particular camera. For now,
# manual modification of settings is required for profiling different hardware or hardware settings.

from camera.flir.flir_camera import FlirCamera
import time

last_time = 0
this_time = 0

# The assumption here is that the inter-callback interval is much larger than the execution time
# of the callback itself. If that is the case, the frame acquisition measurement speed should not
# be affected, just the latency by which it is measured and reported is affected (which is likely
# negligible anyway).
def cameraCallback(image, dimensions, format):
    global last_time
    global this_time

    if last_time == 0: # first iteration
        print(f"Image dimensions: {dimensions[0]}, {dimensions[1]}. Format: '{format}'") 
        print("Period | Speed:")
    else:
        latency = (this_time - last_time) * 1000
        if latency != 0:
            print(f"{latency: .2f} ms | {1000/latency: .2f} FPS")

    last_time = this_time
    this_time = time.time()


def resetTime():
    global last_time
    global this_time
    last_time = 0
    this_time = 0


print("This program runs indefinitely. Use Ctrl-C or another method to exit.")
camera = FlirCamera()

if not camera.open():
    print("Camera failed to start.")
else:
    print(f"Format set? {camera.setSetting('image_format', 'rgb888')}")
    camera.setStreamCaptureCallback(cameraCallback)

    if not camera.startStreamCapture():
        print('Failed to start stream capture for camera')

while(1):
    pass

import amcam
import zmq
import random
import sys
import time
import cv2
import numpy as np
import json
import time
import msgpack

MIN_MATCH_COUNT = 10

# %% ZMQ
port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)

socketimg = context.socket(zmq.PUB)
socketimg.bind("tcp://*:%s" % "5557")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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



class App:
    def __init__(self):
        self.hcam = None
        self.buf = None
        self.total = 0

        MIN_MATCH_COUNT = 10
        self.img1 = None
        self.img1_kp = None
        self.img1_desc = None
        self.sift=cv2.SIFT_create()

# the vast majority of callbacks come from amcam.dll/so/dylib internal threads
    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.CameraCallback(nEvent)

    def CameraCallback(self, nEvent):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            try:
                self.hcam.PullImageV2(self.buf, 24, None)
                socketimg.send(self.buf)
                self.total += 1
                # print('pull image ok, total = {}'.format(self.total))


                # tracking
                if self.img1 is None:
                    # img1 = cv2.imread('img1.png',0) #0 theta
                    # img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)

                    self.img1 = np.frombuffer(self.buf, dtype=np.uint8).reshape(1216, 1824, 3)
                    # crop the middle reigion ~1/4 to 3/4 both in x and y
                    # self.img1 = self.img1[1216//4:1216*3//4, 1824//4:1824*3//4]
                    self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
                    self.img1 = cv2.normalize(self.img1, None, 0, 255, cv2.NORM_MINMAX)
                    # self.img1 = cv2.resize(self.img1, None, fx=0.5, fy=0.5)
                    self.img1 = cv2.resize(self.img1, None, fx=0.25, fy=0.25)
                    self.img1_kp, self.img1_desc = self.sift.detectAndCompute(self.img1,None)

                    # rescale the brightness of img1 by min max
                    # cv2.imwrite('img1.png', img1)

                img1 = self.img1
                img1_desc = self.img1_desc
                img1_kp = self.img1_kp
                img2 = np.frombuffer(self.buf, dtype=np.uint8).reshape(1216, 1824, 3)
                # crop the middle reigion ~1/4 to 3/4 both in x and y
                # img2 = img2[1216//4:1216*3//4, 1824//4:1824*3//4]
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
                # img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)
                img2 = cv2.resize(img2, None, fx=0.25, fy=0.25)
                img2_kp, img2_desc = self.sift.detectAndCompute(img2,None)
                
                #find the keypoints and descriptors with SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks = 50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(img1_desc,img2_desc,k=2)


                good = []
                for m,n in matches:
                    good.append(m)
                if len(good)>MIN_MATCH_COUNT:
                    src_pts = np.float32([ img1_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ img2_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()
                    h,w = img1.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) #top-left, bottom-left, bottom-right, top-right; 4 corner points at img1
                    dst = cv2.perspectiveTransform(pts,M)  
                    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)      #Draw White rectangle dst on img2
                    # print(dst)                                #Transform to img2 use M
                    # dx,dy is x,y offset between center of rectangle dst and center of img2
                    rect_dst=np.int32(dst)
                    h2,w2=img2.shape
                    dx = w2//2 - (rect_dst[0][0][0]+rect_dst[2][0][0])//2
                    dy = h2//2 - (rect_dst[0][0][1]+rect_dst[2][0][1])//2

                    # Calculate theta
                    theta = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

                    # Display the displacement and rotation angle
                    # print(f"Displacement (dx, dy): ({dx}, {dy})")
                    # print(f"Rotation angle (theta): {theta} degrees")

                    socket.send(msgpack.packb([
                         time.time_ns(),
                            dx.tolist(),
                            dy.tolist(),
                            theta.tolist(),
                            img1.tolist(),
                            img2.tolist()
                        ]))
                

        
            except amcam.HRESULTException as ex:
                print('pull image failed, hr=0x{:x}'.format(ex.hr))
        else:
            print('event callback: {}'.format(nEvent))

    def run(self):
        a = amcam.Amcam.EnumV2()
        if len(a) > 0:
            print('{}: flag = {:#x}, preview = {}, still = {}'.format(a[0].displayname, a[0].model.flag, a[0].model.preview, a[0].model.still))
            for r in a[0].model.res:
                print('\t = [{} x {}]'.format(r.width, r.height))
            self.hcam = amcam.Amcam.Open(a[0].id)
            print(self.hcam.MaxSpeed())
            if self.hcam:
                try:
                    self.hcam.put_Size(1824, 1216)
                    self.hcam.put_AutoExpoEnable(True)
                    self.hcam.put_HZ(0)
                    width, height = self.hcam.get_Size()
                    bufsize = ((width * 24 + 31) // 32 * 4) * height
                    print('image size: {} x {}, bufsize = {}'.format(width, height, bufsize))
                    self.buf = bytes(bufsize)
                    if self.buf:
                        try:
                            self.hcam.StartPullModeWithCallback(self.cameraCallback, self)
                        except amcam.HRESULTException as ex:
                            print('failed to start camera, hr=0x{:x}'.format(ex.hr))
                    input('press ENTER to exit')
                finally:
                    self.hcam.Close()
                    self.hcam = None
                    self.buf = None
            else:
                print('failed to open camera')
        else:
            print('no camera found')

if __name__ == '__main__':
    app = App()
    app.run()
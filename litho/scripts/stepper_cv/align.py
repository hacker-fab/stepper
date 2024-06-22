#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:26:39 2023

@author: frankzhao
"""
import time
import numpy as np
import cv2

#Threshold for matching
MIN_MATCH_COUNT = 10

sift=cv2.SIFT_create()

def find_displacement(ref,input_frame,scale_factor=0.25,MIN_MATCH_COUNT=10):
    imgsize = (640, 480)
    #start_time=time.time()
    img2 = input_frame
    img2 = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor)
    #Resize image2 to match the size of image1, for test only, no need to have this in real application
    img1 = ref
    img1 = cv2.resize(img1, imgsize)

    #find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #Create FLANN Match
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    #Store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
       if m.distance < 0.7*n.distance:
           good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = imgsize
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) #top-left, bottom-left, bottom-right, top-right; 4 corner points at img1
        dst = cv2.perspectiveTransform(pts,M)                                  #Transform to img2 use M
        img2 = cv2.UMat.get(img2)
        imganno = cv2.polylines(np.copy(img2),[np.int32(dst)],True,255,3, cv2.LINE_AA)      #Draw White rectangle dst on img2

        # dx,dy is x,y offset between center of rectangle dst and center of img2
        rect_dst=np.int32(dst)
        h2,w2=imgsize
        dx = w2//2 - (rect_dst[0][0][0]+rect_dst[2][0][0])//2
        dy = h2//2 - (rect_dst[0][0][1]+rect_dst[2][0][1])//2
        theta = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        return (dx/scale_factor,dy/scale_factor,theta), (imganno, img2)
    else:
        img2 = cv2.UMat.get(img2)
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        return np.zeros(3), (img2, img2)
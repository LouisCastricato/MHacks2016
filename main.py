# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:37:33 2016

@author: louis
"""
#Basic includes
import cPickle
import gzip
import os
import sys
import timeit
import math

#Computer Vision Includes
import cv2
import argparse

import MLHelper

def rots(e1, e2, ordis=1):
	ydif = e2[1]-e1[1]
	xdif = e2[0]-e1[0]
	yrot = math.atan(ydif/xdif)*(180/3.14159)
	dis = (xdif**2 + ydif**2)**.5
        print dis
        print ordis
	if dis < 0.8*ordis:
		return[yrot,1]

	return [yrot, 0]

class cvHelper:
    def __init__(self):
        self.haarFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        self.haarEyes = cv2.CascadeClassifier("haarcascade_eye.xml");
    def detectFace(self,image):
        detectedFaces = self.haarFace.detectMultiScale(image,1.3,5)
        return detectedFaces
    def detectEye(self,image):
        detectedEyes = self.haarEyes.detectMultiScale(image,1.3,5)
        return detectedEyes


if __name__ == '__main__':    
    """
    #Training Code
    """            
    #Runtime Code
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())
    
    
    # if the video path was not supplied, grab the reference to the
    # camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
    img = cv2.imread("dickbutt.png")
    x_offset = 0
    y_offset = 0
    helper = cvHelper()
    print 'yo'
    a = True;
    counter = 0; 
    rcounter = 0;
    yrot = 0;
    zrot = 0;
    ordis = 1;
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # check to see if we have reached the end of the
        # video
        if not grabbed:
		break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFaces = helper.detectFace(gray)
        detectedEyes = helper.detectEye(gray)
        if detectedFaces is not None:
            for face in detectedFaces:
                   cv2.rectangle(frame,(face[0],face[1]),
                                          (face[0]+face[2],face[1]+face[3]),
                                                         (155, 255, 25),2)
        if detectedEyes is not None:
            for eye in detectedEyes:
                    cv2.rectangle(frame, (eye[0],eye[1]),
                            (eye[0]+eye[2],eye[1]+eye[3]),
                            (155,55,200),2)
        if len(detectedEyes) == 0:
            print 'blink!'
    	if len(detectedEyes) >= 2:
    	    vec1 = (2* detectedEyes[0][0] + 0.5 * detectedEyes[0][2],2* detectedEyes[0][1] + 0.5 * detectedEyes[0][3])
    	    vec2 = (2* detectedEyes[1][0] + 0.5 * detectedEyes[1][2],2* detectedEyes[1][1] + 0.5 * detectedEyes[1][3])
            if a and counter > 10:
                 print "to"
                 ordis = ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**0.5
                 a = False;
            elif a:
                counter += 1;
            rotations = rots(vec1,vec2, ordis)
            yrot += rotations[0];
            zrot += rotations[1];
            print zrot;
            rcounter += 1;
            if rcounter > 2:
                 yrot = yrot/3;
                 print yrot;
                 if zrot > 0:
                    print '3/4 view';
                 else:
                    print 'front view';
                 yrot = 0;
                 zrot = 0;
                 rcounter = 0
        # show the frame and record if the user presses a key
            #frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
		break

        # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

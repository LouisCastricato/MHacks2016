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

#Machine learning includes
import numpy as np
import theano
import theano.tensor as T

#Computer Vision Includes
import cv2
import argparse


class HiddenLayer:
    def __init__(self,input, W_in, b_in, activator = T.tanh):
        
        if W_in is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W_in = theano.shared(value=W_values, name='W', borrow=True)
        if b_in is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_in = theano.shared(value=b_values, name='b', borrow=True)
            
            
        self.W = W_in
        self.b = b_in
        
        lin_func = T.dot(self.W,input) + self.b
        self.y_pred_x = activator(lin_func)
        
        
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
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
 
        # check to see if we have reached the end of the
        # video
        if not grabbed:
		break
        # show the frame and record if the user presses a key
        frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
		break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
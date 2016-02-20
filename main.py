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
    
            
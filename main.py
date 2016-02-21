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

#Machine learning includes
import numpy as np
import theano
import theano.tensor as T
import csv
import numpy
import six.moves.cPickle as pickle
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM

def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """
	
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
	
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
    def depickle(self,n_in = 48* 48,hidden_layers_sizes=[1000,1000,1000]):
        numpy_rng=numpy.random.RandomState(123)
        self.sigmoid_layers = []
        self.rbm_layers = []
        last_later = self.x
        cur_index= 0
        theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))
        for param in self.params:
            print param.get_value().shape
        for index in range(self.n_layers):
            if index == 0:
                input_size= n_in
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[index- 1]
                layer_input =self.sigmoid_layers[-1].output

            simple_sigmoid=HiddenLayer(1, input = layer_input,n_in = input_size,n_out=1000,W = self.params[cur_index], b = self.params[cur_index+1],activation=T.nnet.sigmoid)
            print self.params[cur_index].get_value().shape
            print self.params[cur_index+1].get_value().shape
            cur_index+= 2
            self.sigmoid_layers.append(simple_sigmoid)

            rbm_layer = RBM(numpy_rng=numpy_rng,input=layer_input,
                    W=simple_sigmoid.W,
                    n_visible=input_size,
                    n_hidden=hidden_layers_sizes[index],
                    hbias=simple_sigmoid.b)
            self.rbm_layers.append(rbm_layer)
            
        self.logLayer = LogisticRegression(input = self.sigmoid_layers[-1].output,n_in=hidden_layers_sizes[-1], n_out=7)
        self.logLayer.W = self.params[cur_index]
        self.logLayer.b = self.params[cur_index+1]

        self.y_pred = self.logLayer.p_y_given_x
    def predict(self,inputval):
        return self.y_predict(inputval)
    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	print n_batches
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, learning_rate],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score




def loadmodel():
    f = open("file.save","rb")
    model = cPickle.load(f)
    model.depickle()
    f.close()
    return theano.function(inputs=[model.x],outputs=model.y_pred)
def predict(image,model):
    return model(image)

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
        self.haarFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.haarEyes = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.haarMouth = cv2.CascadeClassifier("Mouth.xml")
    def detectFace(self,image):
        detectedFaces = self.haarFace.detectMultiScale(image,1.3,5)
        return detectedFaces
    def detectEye(self,image):
        detectedEyes = self.haarEyes.detectMultiScale(image,1.3,5)
        return detectedEyes
    def detectMouth(self,image):
        detectedMouth = self.haarMouth.detectMultiScale(image,1.3,5)
        return detectedMouth


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
    
    emotionalModel =loadmodel()
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # check to see if we have reached the end of the
        # video
        if not grabbed:
		break
        gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFaces = helper.detectFace(gray)
        if len(detectedFaces) == 1:
            gray = gray[detectedFaces[0][1]:
                    detectedFaces[0][1] + detectedFaces[0][3]+50,
                    detectedFaces[0][0]: 
                    detectedFaces[0][0] + detectedFaces[0][2]+50]
            facePic = cv2.resize(gray, (48, 48))
            facePic = cv2.equalizeHist(facePic)
            arr = facePic.flatten().reshape(1,48*48)
                     
            print(predict(arr,emotionalModel))
        detectedEyes = helper.detectEye(gray)
        detectedMouths = helper.detectMouth(gray)
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

        if detectedMouths is not None:
            if not len(detectedMouths) == 0:
                lowestMouth = detectedMouths[0]
                for mouth in detectedMouths:
                    if lowestMouth[1]+lowestMouth[3] <  mouth[1]+mouth[3]:
                        lowestMouth = mouth
                cv2.rectangle(frame,(lowestMouth[0],lowestMouth[1]), (lowestMouth[0]+lowestMouth[2],lowestMouth[1]+lowestMouth[3]),(255,55,155),2)
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

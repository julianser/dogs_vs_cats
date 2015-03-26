"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time, datetime
import copy

import numpy

import gc
import random

import theano
import theano.tensor as T

from theano import shared

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

from logreg import LogisticRegression

# Load in commandline arguments:
# - Number of nodes in the hidden layer

if sys.argv[0] == 'mlp.py': # Makse sure module is not used as a library (for example in conv_net.py)
	if len(sys.argv) < 2:
		print 'ERROR! YOU MUST PROVIDE NUMBER OF HIDDEN NODES AS INPUT ARGUMENT'

	set_nodes_in_hidden_layer = int(sys.argv[1])

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))

        # parameters of the model
        self.params = [self.W, self.b]

        # store moving average gradient
	self.avg_gparams = [theano.shared(value=numpy.asarray(numpy.zeros((n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True), theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)]       

	# store the accumulated squared gradients used for AdaGrad
	self.accumulated_squared_gparams = [theano.shared(value=numpy.asarray(numpy.zeros((n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True), theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)]       

	# store the true parameters used for Nesterov's Accelerated Gradient        
	self.nag_grad = [theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True), theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out


        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors and confusion matrix
        self.errors = self.logRegressionLayer.errors
        self.y_predicted_incorrect = self.logRegressionLayer.y_predicted_incorrect
        self.confusion_matrix = self.logRegressionLayer.confusion_matrix

        # the parameters of the model are the parameters of the two layer it is
        #self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # store moving average gradient
        self.avg_gparams = self.hiddenLayer.avg_gparams + self.logRegressionLayer.avg_gparams

	# store the accumulated squared gradients used for AdaGrad
	self.accumulated_squared_gparams = self.hiddenLayer.accumulated_squared_gparams + self.logRegressionLayer.accumulated_squared_gparams

	# store the true parameters used for Nesterov's Accelerated Gradient        
	self.nag_grad = self.hiddenLayer.nag_grad + self.logRegressionLayer.nag_grad


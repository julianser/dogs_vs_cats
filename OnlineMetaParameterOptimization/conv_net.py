"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
__docformat__ = 'restructedtext en'


import cPickle
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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0,parentdir + '/Config') 
import config

sys.path.insert(0,parentdir + '/LoadData') 
from load_data import load_data

sys.path.insert(0,parentdir + '/LogReg') 
from logreg import LogisticRegression

sys.path.insert(0,parentdir + '/MLP') 
from mlp import HiddenLayer

# Load in commandline arguments:
# - Pooling size across ranks at level one of network; for example: 3
# - Pooling size across files at level one of network; for example: 3
# - Pooling size across ranks at level two of network; for example: 3
# - Pooling size across files at level two of network; for example: 3
# - Number of feature maps at level one; for example: 20
# - Number of feature maps at level two; for example: 50
# - Number of hidden units at the third (top) level of network; for example: 500

if sys.argv[0] == 'conv_net.py': # Makse sure module is not used as a library (for example in conv_net.py)
	if len(sys.argv) < 8:
		print 'ERROR! YOU MUST PROVIDE FOLLOWING ARGUMENTS: '
		print ' - Pooling size across ranks at level one of network; for example: 3'
		print ' - Pooling size across files at level one of network; for example: 3'
		print ' - Pooling size across ranks at level two of network; for example: 3'
		print ' - Pooling size across files at level two of network; for example: 3'
		print ' - Number of feature maps at level one; for example: 20'
		print ' - Number of feature maps at level two; for example: 50'
		print ' - Number of hidden units at the third (top) level of network; for example: 500'
	else:
	    pooling_rank_level_one = int(sys.argv[1])
	    pooling_file_level_one = int(sys.argv[2])
	    pooling_rank_level_two = int(sys.argv[3])
	    pooling_file_level_two = int(sys.argv[4])
	    model_kernel = [int(sys.argv[5]), int(sys.argv[6])]
	    model_third_layer_hidden_units = int(sys.argv[7])



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # store moving average gradient
	self.avg_gparams = [theano.shared(numpy.asarray(numpy.zeros((filter_shape)), dtype=theano.config.floatX), borrow=True), theano.shared(value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True)]

            
def split_data_into_partitions(leftover_set_x, leftover_set_y, new_set_x, new_set_y, divisible_number):
    """ Function takes two data sets (for respectively x and y pair values), 
    and returns two new data sets (for respectively x and y pair values) such 
    that the latter set of x and y pairs have a number of entries divisible
    by divisible_number
    
    :type leftover_set_x: matrix
    :param leftover_set_x: first data set corresponding to x values

    :type leftover_set_y: vector
    :param leftover_set_y: first data set corresponding to y values

    :type new_set_x: matrix
    :param new_set_x: second data set corresponding to x values

    :type new_set_y: vector
    :param new_set_y: second data set corresponding to y values
    
    """

    complete_set_x = numpy.concatenate((leftover_set_x, new_set_x))
    complete_set_y = numpy.concatenate((leftover_set_y, new_set_y))

    instances = complete_set_x.shape[0]
    leftover_instances = instances  % divisible_number
    current_set_x = complete_set_x[0:instances-leftover_instances,:]
    current_set_y = complete_set_y[0:instances-leftover_instances]
    leftover_set_x = complete_set_x[instances-leftover_instances:instances,:]
    leftover_set_y = complete_set_y[instances-leftover_instances:instances]
  
    return leftover_set_x, leftover_set_y, current_set_x, current_set_y
  
  
def optimization(learning_rate=1, lambda_decay=0.5, beta_constant=0.5, nkerns=[2, 3], n_hidden=50, n_updates_per_batch=1, n_epochs=10, file_set = range(1, 2)):
    """ Function performs a stochastic gradient descent optimization for a multilayer perceptron
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_hidden: int
    :param n_hidden: number of units in hidden layer
    
    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)
    
    :type n_updates_per_batch: int
    :param n_updates_per_batch: number of gradient updates per batch (file)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type file_set: list[ints]
    :param file_set: a list of file indices to use for training and early-validation (filename loaded will be chess_x.txt)
    """
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    
    conv_net_batch_size = 5
    
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # Define training and validation sets as shared variables
    train_set_x = theano.shared(numpy.zeros((0, 0), dtype=theano.config.floatX))
    train_set_y = theano.shared(numpy.zeros((0), dtype=numpy.int32))
    valid_set_x = theano.shared(numpy.zeros((0, 0), dtype=theano.config.floatX))
    valid_set_y = theano.shared(numpy.zeros((0), dtype=numpy.int32))
    
    # Reshape matrix of rasterized images of shape (conv_net_batch_size,8*8)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((conv_net_batch_size, 13, 8, 8))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (conv_net_batch_size,nkerns[0],12,12)
    rng = numpy.random.RandomState(23455)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(conv_net_batch_size, 13, 8, 8),
            filter_shape=(nkerns[0], 13, pooling_rank_level_one, pooling_file_level_one), poolsize=(1, 1))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(conv_net_batch_size, nkerns[0], 8-pooling_rank_level_one+1, 8-pooling_file_level_one+1),
            filter_shape=(nkerns[1], nkerns[0], pooling_rank_level_two, pooling_file_level_two), poolsize=(1, 1))


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (conv_net_batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * (8-pooling_rank_level_one-pooling_rank_level_two+2) * (8-pooling_file_level_one-pooling_file_level_two+2),
                         n_out=n_hidden, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=3)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    validate_model = theano.function(inputs=[], outputs=layer3.errors(y),
            givens={
                x: valid_set_x,
                y: valid_set_y})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    current_learning_rate = theano.shared(value=numpy.asarray((1), dtype=theano.config.floatX), name='current_learning_rate', borrow=True)                   
        
    grad_params = layer3.avg_gparams + layer2.avg_gparams + layer1.avg_gparams + layer0.avg_gparams
    updates = []
    #for param, avg_gparam, gparam in zip(params, grad_params, grads):
    #    updates.append((param, param - learning_rate * gparam)) 
    for param, gparam, avg_gparam in zip(params, grads, grad_params):
        new_avg_gparam = (1-beta_constant)*avg_gparam + beta_constant * gparam
        updates.append((param, param - current_learning_rate * new_avg_gparam)) 
        updates.append((avg_gparam, new_avg_gparam)) 




    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    index = T.lscalar()  # index to a [mini]batch
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * conv_net_batch_size:(index + 1) * conv_net_batch_size],
                y: train_set_y[index * conv_net_batch_size:(index + 1) * conv_net_batch_size]})

    #train_model = theano.function(inputs=[], outputs=cost, updates=updates,
    #      givens={
    #        x: train_set_x,
    #        y: train_set_y})
                        
    ###############
    # TRAIN MODEL #
    ###############                         
    best_params = None
    best_validation_loss = numpy.inf    

    # These variables keep track of 'left-over' data, i.e. instances which 
    # could not be partitioned into batches of size conv_net_batch_size
    # At the end of the execution up to conv_net_batch_size - 1 data instances 
    # may be lost, but its worth it  as removing this procedure will be a magnitude slower (10-20x slower!)
    leftover_train_set_x = numpy.empty((0,832), dtype=theano.config.floatX)
    leftover_train_set_y = numpy.empty((0), dtype=numpy.int32)
    leftover_valid_set_x = numpy.empty((0,832), dtype=theano.config.floatX)
    leftover_valid_set_y = numpy.empty((0), dtype=numpy.int32)

    # We keep track of the total number of training points used 
    # (as inevitably, due to training in fixed mini-batch sizes, we will loose some points):
    totalTrainingInstancesUsed = 0



    ###############
    # LOAD PREVIOUSLY TRAINED MODEL (IF ANY) #
    ###############       

    epoch = 0
    best_params = None
    best_validation_loss = numpy.inf 

    # Check if there is a previously saved version of this model (trained in any epoch). 
    # If there is, then load it in and resume training from there
    model_identifier = 'CONVNET_Model_' + str(config.unique_save_load_id) + "_" + str(numpy.sum(file_set)) + str(pooling_rank_level_one) + "_" + str(pooling_file_level_one) + "_" + str(pooling_rank_level_two) + "_" + str(pooling_file_level_two) + "_" + str(nkerns[0]) + "_" + str(nkerns[1]) + "_" + str(n_hidden) + "_" + str(learning_rate) + "_" + str(beta_constant) + "_" + str(lambda_decay) + "_" + str(n_updates_per_batch) + "_"
    if config.do_save_load_temporary_models==True:
	found_previous_model = False
	previous_model_filename = ''
	previous_model_epoch = 0
        for epoch in range(1, n_epochs+1):
	    model_filename = config.save_load_temporary_models_directory + model_identifier + str(epoch)
	    if os.path.isfile(model_filename):
		found_previous_model = True
		previous_model_filename = model_filename
		previous_model_epoch = epoch

	# Set parameters to previous trained model
	if found_previous_model==True:
	    print '  Found model already trained as ' + previous_model_filename + ' in epoch ' + str(previous_model_epoch) + ' ...'
	    load_file = open(previous_model_filename, 'rb')
	    modparam = cPickle.load(load_file)  # the -1 is for HIGHEST_PROTOCOL
	    load_file.close()
		
	    model_parameters = modparam[0]
	    best_params = modparam[1]
	    best_validation_loss = modparam[2]

	    # Set model parameters 
	    updates = []
	    updates.append((layer0.W, model_parameters[0]))
	    updates.append((layer0.b, model_parameters[1]))
	    updates.append((layer1.W, model_parameters[2]))
	    updates.append((layer1.b, model_parameters[3]))
	    updates.append((layer2.W, model_parameters[4]))
	    updates.append((layer2.b, model_parameters[5]))
	    updates.append((layer3.W, model_parameters[6]))
	    updates.append((layer3.b, model_parameters[7]))
	    
	    set_model_parameters = theano.function(inputs=[], outputs=[], updates=updates)
	    set_model_parameters()

	    # Set epoch to begin training from previous training
	    epoch = previous_model_epoch
	else:
	    epoch = 0

    # Loop over each epoch
    while (epoch < n_epochs):        
        print '  epoch: ' + str(epoch)
        epoch = epoch + 1
	
	# Set the current learning rate 
	current_learning_rate.set_value(learning_rate / (1 + learning_rate * lambda_decay * epoch))
        
        # Keep track of current model loss (its a running mean over one minus the accuracy)
	loss_values = []
	loss_weights = []

        # Shuffle the files
        numpy.random.shuffle(file_set)

	# Take each file as a batch of points and train on mini-batches of these        
        for fileIndex in file_set:
            print '    file: ' + str(fileIndex)

            datasets = load_data(datasetIndex=fileIndex, validationSetPercentage=config.validationSetPercentage, useMeepFeatures=False, useBitboardFeatures=True, useStockfishQuiescenceFeatures=False, useStockfishStaticFeatures=False, appendSideToMoveAsBooleanFeature=False, appendSideToMoveAsBitboardFeatures=True, appendFilledSquaresBitBoardFeatures=False, appendPieceCountFeatures=False)


            # Now we split the data such that the training and validation sets are divisible by conv_net_batch_size
            leftover_train_set_x, leftover_train_set_y, current_train_set_x, current_train_set_y = split_data_into_partitions(leftover_train_set_x, leftover_train_set_y, datasets[0][0].get_value(), datasets[0][3].eval(), conv_net_batch_size)
            leftover_valid_set_x, leftover_valid_set_y, current_valid_set_x, current_valid_set_y = split_data_into_partitions(leftover_valid_set_x, leftover_valid_set_y, datasets[1][0].get_value(), datasets[1][3].eval(), conv_net_batch_size)

            train_set_x.set_value(current_train_set_x)
            train_set_y.set_value(current_train_set_y)

            
            # Train on this batch 'n_updates_per_batch' times
            for batch_index in range(0, n_updates_per_batch):
                # Train this on sub-batches of size conv_net_batch_size
                sub_batches_count = current_train_set_x.shape[0] / conv_net_batch_size
                for sub_batch_index in range(0, sub_batches_count):
                    # Set training data and train
                    #train_set_x.set_value(current_train_set_x[sub_batch_index*conv_net_batch_size:(sub_batch_index+1)*conv_net_batch_size, :])
                    #train_set_y.set_value(current_train_set_y[sub_batch_index*conv_net_batch_size:(sub_batch_index+1)*conv_net_batch_size])
                    filebatch_avg_cost = train_model(sub_batch_index)
                    #print 'filebatch_avg_cost: ' + str(filebatch_avg_cost)

            sub_batches_count = current_valid_set_x.shape[0] / conv_net_batch_size
                        
	    # Calculate loss on validation set
            for sub_batch_index in range(0, sub_batches_count):            
                # Set validation data and validate
                valid_set_x.set_value(current_valid_set_x[sub_batch_index*conv_net_batch_size:(sub_batch_index+1)*conv_net_batch_size, :])
                valid_set_y.set_value(current_valid_set_y[sub_batch_index*conv_net_batch_size:(sub_batch_index+1)*conv_net_batch_size])
            	loss_values.append(numpy.squeeze(validate_model()))
            	loss_weights.append(valid_set_x.get_value().shape[0])
                #sub_validation_loss = sub_validation_loss + validate_model()
            

	    totalTrainingInstancesUsed = totalTrainingInstancesUsed + current_train_set_x.shape[0] + current_valid_set_x.shape[0]
	    
            # Clean up memory, as each single file takes up +600 MB
            gc.collect()

	if epoch==1:
		print '    (total training data instances used was ' + str(totalTrainingInstancesUsed) + ')'
	
	this_validation_loss = numpy.average(loss_values, weights=loss_weights, axis=0)
        print '    current_validation_loss: ' + str(this_validation_loss)      
        
        # if we got the best validation score until now, save the score and parameters
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_params = copy.deepcopy([layer0.W.eval(), layer0.b.eval(), layer1.W.eval(), layer1.b.eval(), layer2.W.eval(), layer2.b.eval(), layer3.W.eval(), layer3.b.eval()])

	# If enabled save current model parameters at the end of epoch
	if config.do_save_load_temporary_models==True:
		model_filename = config.save_load_temporary_models_directory + model_identifier + str(epoch)
		print '    Saving model as ' + model_filename + '...'
		save_file = open(model_filename, 'wb')
		current_model_paramters = copy.deepcopy([layer0.W.eval(), layer0.b.eval(), layer1.W.eval(), layer1.b.eval(), layer2.W.eval(), layer2.b.eval(), layer3.W.eval(), layer3.b.eval()])
		cPickle.dump([current_model_paramters, best_params, best_validation_loss], save_file, -1)
		save_file.close()

    return best_validation_loss, best_params

def get_errors_at_movenumber(prediction_incorrect_vector, movenumber_vector, movenumber):
    """ Function returns the number of errors at a certain movenumber (move) and 
        the total number of positions at that movenumber (move).

    :type prediction_incorrect_vector: int vector
    :param prediction_incorrect_vector: vector containing a one for all instances predicted incorrectly and zero otherwise

    :type movenumber_vector: int vector
    :param movenumber_vector: vector containing the move number of the corresponding position

    :type movenumber: int
    :param movenumber: move number to calculate errors for
    """

    return numpy.dot(numpy.squeeze(prediction_incorrect_vector), numpy.equal(movenumber_vector, movenumber).astype('int32')), numpy.sum(numpy.equal(movenumber_vector, movenumber).astype('int32'))
    
def test_model(model_parameters, nkerns, n_hidden, file_set):
    """ Function tests a convolutional neural network on a data set.
        Returns the validation error (in percentage) and the confusion matrix.
        This is much slower compared to other methods, as we do not want to 
        train only multiplies of a 1000, but want to train on every single 
        test instance.

    :type model_parameters: list
    :param model_parameters: Parameters of model to test

    :type file_set: list[ints]
    :param file_set: a list of file indices to use for testing (filename loaded will be chess_x.txt)
    """
    
    conv_net_batch_size = 100
    
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # Declare movenumber related Theano variables
    movenumber_to_calculate_error_at = T.scalar('movenumber_to_calculate_error_at')
    movenumbers = T.ivector('movenumbers')
                           
    # Define training and validation sets as shared variables
    test_set_x = theano.shared(numpy.zeros((0, 0), dtype=theano.config.floatX))
    test_set_y = theano.shared(numpy.zeros((0), dtype=numpy.int32))
    
    # Reshape matrix of rasterized images of shape (conv_net_batch_size,8*8)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((conv_net_batch_size, 13, 8, 8))


    
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (conv_net_batch_size,nkerns[0],12,12)
    rng = numpy.random.RandomState(23455)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(conv_net_batch_size, 13, 8, 8),
            filter_shape=(nkerns[0], 13, pooling_rank_level_one, pooling_file_level_one), poolsize=(1, 1))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(conv_net_batch_size, nkerns[0], 8-pooling_rank_level_one+1, 8-pooling_file_level_one+1),
            filter_shape=(nkerns[1], nkerns[0], pooling_rank_level_two, pooling_file_level_two), poolsize=(1, 1))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (conv_net_batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=n_hidden, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=3)

    # Set model parameters 
    updates = []
    updates.append((layer0.W, model_parameters[0]))
    updates.append((layer0.b, model_parameters[1]))
    updates.append((layer1.W, model_parameters[2]))
    updates.append((layer1.b, model_parameters[3]))
    updates.append((layer2.W, model_parameters[4]))
    updates.append((layer2.b, model_parameters[5]))
    updates.append((layer3.W, model_parameters[6]))
    updates.append((layer3.b, model_parameters[7]))
    
    set_model_parameters = theano.function(inputs=[], outputs=[], updates=updates)
    set_model_parameters()


    # Compile theano functions to return loss, confusion matrix and loss w.r.t. time steps
    test_set_x = theano.shared(numpy.zeros((0, 0), dtype=theano.config.floatX))
    test_set_y = theano.shared(numpy.zeros((0), dtype=numpy.int32))
    test_set_movenumbers = theano.shared(numpy.zeros((0), dtype=numpy.int32))

    test_model_loss = theano.function(inputs=[],
         outputs=[layer3.errors(y)],
         givens={
            x: test_set_x,
            y: test_set_y}) 
        
    test_model_confusion_matrix = theano.function(inputs=[],
         outputs=[layer3.confusion_matrix(y, 0, 0), layer3.confusion_matrix(y, 0, 1), layer3.confusion_matrix(y, 0, 2), layer3.confusion_matrix(y, 1, 0), layer3.confusion_matrix(y, 1, 1), layer3.confusion_matrix(y, 1, 2), layer3.confusion_matrix(y, 2, 0), layer3.confusion_matrix(y, 2, 1), layer3.confusion_matrix(y, 2, 2)],
         givens={
            x: test_set_x,
            y: test_set_y})
        
    # Helper function for get_errors_at_movenumber
    get_y_predicted_incorrect = theano.function(inputs=[],
	     outputs=[layer3.y_predicted_incorrect(y)],
	     givens={
		x: test_set_x,
		y: test_set_y})

    # Loop over each file and calculate the loss, loss at every movenumber and confusion matrix
    loss_values = []
    loss_weights = []

    loss_movenumber_values = numpy.zeros((0, config.calculate_errors_at_movenumbers.shape[0]))
    loss_movenumber_weights = numpy.zeros((0, config.calculate_errors_at_movenumbers.shape[0]))

    confusion_matrix = [[0 for i in xrange(3)] for i in xrange(3)]
    
    totalTestInstancesUsed = 0;
    for fileIndex in file_set:
	# Load in data
        datasets = load_data(datasetIndex=fileIndex, validationSetPercentage=0, useMeepFeatures=False, useBitboardFeatures=True, useStockfishQuiescenceFeatures=False, useStockfishStaticFeatures=False, appendSideToMoveAsBooleanFeature=False, appendSideToMoveAsBitboardFeatures=True, appendFilledSquaresBitBoardFeatures=False, appendPieceCountFeatures=False)
	
	complete_set_x = datasets[0][0].get_value()
	complete_set_movenumbers = datasets[0][2].eval()
        complete_set_y = datasets[0][3].eval()

	batch_count = numpy.floor(complete_set_x.shape[0] / conv_net_batch_size).astype(int)
	totalTestInstancesUsed = totalTestInstancesUsed + batch_count*conv_net_batch_size
	        
	for batch_index in range(0, batch_count):    
		#print 'batch_index: ' + str(batch_index) + ' / ' + str(batch_count)
		test_set_x.set_value(complete_set_x[(batch_index*conv_net_batch_size):((batch_index+1)*conv_net_batch_size), :])
		test_set_movenumbers.set_value(complete_set_movenumbers[(batch_index*conv_net_batch_size):((batch_index+1)*conv_net_batch_size)])
		test_set_y.set_value(complete_set_y[(batch_index*conv_net_batch_size):((batch_index+1)*conv_net_batch_size)])

		loss_values.append(numpy.squeeze(test_model_loss()))
		loss_weights.append(test_set_x.get_value().shape[0])
		
		confusion_matrix = confusion_matrix + numpy.reshape(numpy.squeeze(test_model_confusion_matrix()), [3, 3])

		# Calculate model loss over each time step
		current_model_loss_movenumbers_values = numpy.zeros((1, config.calculate_errors_at_movenumbers.shape[0]))
		current_model_loss_movenumbers_weights = numpy.zeros((1, config.calculate_errors_at_movenumbers.shape[0]))
		for i in range(0, config.calculate_errors_at_movenumbers.shape[0]):
			#current_model_loss_movenumbers_values[0, i], current_model_loss_movenumbers_weights[0, i] = test_model_loss_at_movenumber(config.calculate_errors_at_movenumbers[i])
			current_model_loss_movenumbers_values[0, i], current_model_loss_movenumbers_weights[0, i] = get_errors_at_movenumber(get_y_predicted_incorrect(), test_set_movenumbers.get_value(), config.calculate_errors_at_movenumbers[i])

			if (numpy.isfinite(current_model_loss_movenumbers_values[0, i]) == True) and (current_model_loss_movenumbers_weights[0, i] > 0):
				current_model_loss_movenumbers_values[0, i] = current_model_loss_movenumbers_values[0, i] / current_model_loss_movenumbers_weights[0, i]
			else:
				current_model_loss_movenumbers_values[0, i]  = 0
				current_model_loss_movenumbers_weights[0, i] = 0


		loss_movenumber_values = numpy.concatenate((loss_movenumber_values, numpy.vstack(current_model_loss_movenumbers_values)))
		loss_movenumber_weights = numpy.concatenate((loss_movenumber_weights, numpy.vstack(current_model_loss_movenumbers_weights)))

    # Average model loss over each move step (we do it in this order to prevent loss of precision)
    average_loss_movenumbers = numpy.zeros(config.calculate_errors_at_movenumbers.shape[0])
    for i in range(0, config.calculate_errors_at_movenumbers.shape[0]):
	average_loss_movenumbers[i] = numpy.dot(loss_movenumber_values[:, i], loss_movenumber_weights[:, i])
	average_loss_movenumbers[i] = average_loss_movenumbers[i] / numpy.sum(loss_movenumber_weights[:, i])

    print '    (total test data instances used was ' + str(totalTestInstancesUsed) + ')'

    return numpy.average(loss_values, weights=loss_weights, axis=0), confusion_matrix, average_loss_movenumbers
    
def hyper_optimization():   
    """ Performs a hyper-parameter optimization for multi-layered neural network based on the entire data set, 
       and prints the model accuracy and confusion matrix on the test set based on the model performing best on the validation set.
       It also saves the parameters of the optimal model to disc.
    """

    # If kernels and hidden units in top layer are given use these
    #learning_rate_list = [0.1, 0.01]   
    learning_rate_list = [10]   
    kernel_list = [model_kernel] 
    n_hidden_list = [model_third_layer_hidden_units]
    lambda_decay_list = [0]
    beta_constant_list = [1]

    n_updates_per_batch = 1
    n_epochs = 5
    
    best_validation_loss = numpy.inf 
    
    optimal_learning_rate = 0.01
    optimal_lambda_decay = 0
    optimal_beta_constant = 0.5
    optimal_kernel = [10, 20]
    optimal_n_hidden = 10012
    
    print 'Running hyper-parameter optimization...'
    for learning_rate in learning_rate_list:
       for lambda_decay in lambda_decay_list:
           for beta_constant in beta_constant_list:

	      for nkerns in kernel_list:
	          for n_hidden in n_hidden_list:
	              print 'Running model ...'
	              print datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
	              print ' learning_rate: ' + str(learning_rate)
                      print ' beta_constant: ' + str(beta_constant)                
                      print ' lambda_decay: ' + str(lambda_decay)      
	              print ' nkerns: ' + str(nkerns)
	              print ' n_hidden: ' + str(n_hidden)
	        
	              early_stopping_validation_loss, model_parameters = optimization(learning_rate, lambda_decay, beta_constant, nkerns, n_hidden, n_updates_per_batch, n_epochs, config.trainingfile_set)
	              validation_loss, validation_confusion_matrix, validation_loss_on_movenumbers = test_model(model_parameters, nkerns, n_hidden, config.validationfile_set)
	        
	              print ' model validation accuracy: ' + str(1-validation_loss)
	              print ' model validation confusion matrix: '
 		      print validation_confusion_matrix
	              print ' model validation accuracy on time-steps: '
 		      print 1-validation_loss_on_movenumbers
	        
	              if validation_loss < best_validation_loss:
	                  best_validation_loss = validation_loss
	                  optimal_learning_rate = learning_rate
                          optimal_lambda_decay = lambda_decay
                          optimal_beta_constant = beta_constant
	                  optimal_kernel = nkerns
	                  optimal_n_hidden = n_hidden
	    
	    
    print 'Found optimal hyper parameters:'
    print ' learning_rate: ' + str(optimal_learning_rate)
    print ' lambda_decay: ' + str(optimal_lambda_decay)                
    print ' beta_constant: ' + str(optimal_beta_constant)      
    print ' optimal_kernel: '
    print optimal_kernel
    print ' optimal_n_hidden: ' + str(optimal_n_hidden)

    
    print 'Training optimal model on entire training+validation data set...'
    early_stopping_validation_loss, model_parameters = optimization(optimal_learning_rate, optimal_lambda_decay, optimal_beta_constant, optimal_kernel, optimal_n_hidden, n_updates_per_batch, n_epochs, config.trainingValidationfile_set)
    print ''
    
    print 'Testing model...'
    test_loss, test_confusion_matrix, test_loss_on_movenumbers = test_model(model_parameters, optimal_kernel, optimal_n_hidden, config.testfile_set)
    print 'Test Statistics: '
    print ' Mean Accuracy: ' + str(1-test_loss)
    print ''
    print ' Confusion Matrix (Rows: True Class; Columns: Predicted Class): '
    print ' Black Wins  Draw    White Wins'
    print test_confusion_matrix
    print ''
    print ' Accuracy w.r.t. time-steps: '
    print 1-test_loss_on_movenumbers

    print ''
    print ''

    print 'Saving Optimal Model Parameters'
    save_file = open('Optimal_Model', 'wb')  # this will overwrite current contents
    cPickle.dump(model_parameters, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()
    
    print datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print 'Finished'
    
if __name__ == '__main__':
    print '### Running ConvNet: Bitboard Representation Only ###'
    print '### Convolutional Layer => Pooling Layer => Convolutional Layer => Pooling Layer => MLP Player => Output ###'

    print 'pooling_rank_level_one: ' + str(pooling_rank_level_one)
    print 'pooling_file_level_one: ' + str(pooling_file_level_one)
    print 'pooling_rank_level_two: ' + str(pooling_rank_level_two)
    print 'pooling_file_level_two: ' + str(pooling_file_level_two)


    print 'model_kernel'
    print model_kernel
    print 'model_third_layer_hidden_units: ' + str(model_third_layer_hidden_units)

    hyper_optimization()

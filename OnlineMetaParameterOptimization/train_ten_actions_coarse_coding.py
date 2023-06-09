from dataset import *
from pylearn2.utils.rng import make_np_rng
import numpy as np

import os
import sys
import time

import theano
import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logreg import LogisticRegression
from mlp import HiddenLayer
from theano import shared

import argparse

def relu(x):
    return theano.tensor.switch(x<0, 0, x)
    
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
        

class RandomCrop():
    """
    Crops a square at random on a rescaled version of the image

    Parameters
    ----------
    scaled_size : int
        Size of the smallest side of the image after rescaling
    crop_size : int
        Size of the square crop. Must be bigger than scaled_size.
    rng : int or rng, optional
        RNG or seed for an RNG
    """
    _default_seed = 2015 + 1 + 18

    def __init__(self, scaled_size, crop_size, rng=_default_seed):
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        assert self.scaled_size > self.crop_size
        self.rng = make_np_rng(rng, which_method="random_integers")

    def get_shape(self):
        return (self.crop_size, self.crop_size)

    def preprocess(self, image):
        small_axis = numpy.argmin(image.shape[:-1])
        ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
        resized_image = misc.imresize(image, ratio)

        max_i = resized_image.shape[0] - self.crop_size
        max_j = resized_image.shape[1] - self.crop_size
        i = self.rng.randint(low=0, high=max_i)
        j = self.rng.randint(low=0, high=max_j)
        return resized_image[i : i + self.crop_size, j : j + self.crop_size, :]

# Least Squares Q-Learning Agent
class RLAgent():

  def __init__(self):
		self.n_actions = 10
		self.n_features = 5

		self.theta = np.zeros((self.n_features*self.n_actions))
		self.exploration_prob = 0.2
		self.gamma = 0.9
                self.features_history = np.zeros((1, self.n_features))
                self.action_history = np.zeros((1, 1))
                self.reward_history = np.zeros((1, 1))

                #self.matrix_to_invert_history = np.zeros((self.n_features*self.n_actions, self.n_features*self.n_actions))
                # Simulate prior on lowest learning rate
                features = np.zeros((self.n_features,1))
                features[0] = 1

                action = np.ones((1,1))*9
                reward = np.ones((1,1))*0.1
                for i in range(0, 100):
                   self.make_action(features, action, reward, features)



  def make_action(self, features_p, action_p, reward, features):
        self.features_history = np.append(self.features_history, features_p.T, axis=0)
        self.action_history = np.append(self.action_history, action_p, axis=0)
        self.reward_history = np.append(self.reward_history, reward, axis=0)
        
        matrix_to_invert = np.zeros((self.n_features*self.n_actions, self.n_features*self.n_actions))
        for t in range(0, self.features_history.shape[0]-1):
           a_t = int(self.action_history[t])
           s_t = np.zeros((self.n_features*self.n_actions))
           s_t[self.n_features*a_t:self.n_features*(a_t+1)] = self.features_history[t, :] # Coarse grid features
           
           # Coarse grid features
           if a_t > 0:
              s_t[self.n_features*(a_t-1):self.n_features*(a_t)] = self.features_history[t, :]
           if a_t < self.n_actions - 1:
              s_t[self.n_features*(a_t+1):self.n_features*(a_t+2)] = self.features_history[t, :]

           best_v = - np.inf
           best_a = 0
           for a in range(self.n_actions):
              v = np.dot(self.features_history[t+1], self.theta[a*self.n_features:(a+1)*self.n_features])
              if v > best_v:
                 best_v = v
                 best_a = a


           s_t_plus_one = np.zeros((self.n_features*self.n_actions))
           s_t_plus_one[self.n_features*best_a:self.n_features*(best_a+1)] = self.features_history[t+1]

           matrix_to_invert = matrix_to_invert + s_t*np.transpose(s_t - self.gamma * s_t_plus_one)

        vector_to_multiply = np.zeros((self.n_features*self.n_actions))
        for t in range(0, self.features_history.shape[0]-1):
           a_t = int(self.action_history[t])
           s_t = np.zeros((self.n_features*self.n_actions))
           s_t[self.n_features*a_t:self.n_features*(a_t+1)] = self.features_history[t, :]

           # Coarse grid features
           if a_t > 0:
              s_t[self.n_features*(a_t-1):self.n_features*(a_t)] = self.features_history[t, :]
           if a_t < self.n_actions - 1:
              s_t[self.n_features*(a_t+1):self.n_features*(a_t+2)] = self.features_history[t, :]

           vector_to_multiply = vector_to_multiply + s_t * self.reward_history[t]

        vector_to_multiply = np.reshape(vector_to_multiply, ((self.n_features*self.n_actions), 1))

        self.theta = np.linalg.pinv(np.asmatrix(matrix_to_invert))*np.asmatrix(vector_to_multiply)
        print 'Agent: theta', self.theta        

        previous_action = action_p[0]

        best_v = - np.inf
        best_a = previous_action
        for a in range(self.n_actions):
           if a >= previous_action - 1 and a <= previous_action + 1:
              v = np.dot(features.T, self.theta[a*self.n_features:(a+1)*self.n_features])

              # Coarse grid features
              if a > 0:
                 v = v + np.dot(features.T, self.theta[(a-1)*self.n_features:(a)*self.n_features])
              if a < self.n_actions - 1:
                 v = v + np.dot(features.T, self.theta[(a+1)*self.n_features:(a+2)*self.n_features])

              if v > best_v:
                 best_v = v
                 best_a = a


        #best_v = - np.inf
        #best_a = 0
        #for a in range(self.n_actions):
        #   v = np.dot(features.T, self.theta[a*self.n_features:(a+1)*self.n_features])
        #   if v > best_v:
        #      best_v = v
        #      best_a = a

        ran = np.random.uniform(0,1)
        if ran < self.exploration_prob: # Choose random action
            best_a = previous_action
	    ran_three = np.random.uniform(0,1)
	    if ran_three > 0.5:
	       best_a = best_a + 1
	    else:
	       best_a = best_a - 1
	      
	    best_a = min(max(best_a, 0), self.n_actions - 2)          

        return best_a



def train(useRLAgent, initialLearningRate, dataset_train, dataset_valid, dataset_test, batch_size, minibatches_per_action, training_batches, validation_batches, test_batches):
    # General variables
    #image_width = 256
    #image_height = 221
    #image_scaled_size = 32
    #image_width = 28
    #image_height = 28

    #image_scaled_size = 256
    #image_width = 221
    #image_height = 221

    #image_scaled_size = 128
    #image_width = 96
    #image_height = 96
    image_scaled_size = 96
    image_width = 60
    image_height = 60
    
    momentum = 0.5

    agent = RLAgent()   

    minibatches_between_RL_actions = 100
    
    image_channels = 3
    transformer=RandomCrop(image_scaled_size, image_height)
    nkerns=[32, 32]
    
    max_epochs = 500
    rng = numpy.random.RandomState(23455)
    
    train_set_x = theano.shared(np.zeros((0,0), dtype=theano.config.floatX))
    train_set_y = theano.shared(np.zeros((0), dtype=np.int32))
    valid_set_x = theano.shared(np.zeros((0,0), dtype=theano.config.floatX))
    valid_set_y = theano.shared(np.zeros((0), dtype=np.int32))
    test_set_x = theano.shared(np.zeros((0,0), dtype=theano.config.floatX))
    test_set_y = theano.shared(np.zeros((0), dtype=np.int32))

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
                        
    learning_rate = shared(float(initialLearningRate))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, image_channels, image_width, image_height))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, image_channels, image_width, image_height),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        #image_shape=(batch_size, nkerns[0], 12, 12),
        image_shape=(batch_size, nkerns[0], 28, 28),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    #layer2 = HiddenLayer(
    #    rng,
    #    input=layer2_input,
    #    #n_in=nkerns[1] * 4 * 4,
    #    n_in=nkerns[1] * 21 * 21,
    #    #n_out=500,
    #    n_out=50,
    #    #activation=T.tanh
    #    activation=relu
    #)

    layer2 = LogisticRegression(input=layer2_input, n_in=4608, n_out=2)
    cost = layer2.negative_log_likelihood(y)
    
    layer3 = LogisticRegression(input=layer2_input, n_in=4608, n_out=2) # DUMMY LAYER!

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=50, n_out=2)

    # the cost we minimize during training is the NLL of the model
    #cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        #layer3.errors(y),
        layer2.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        #[index],
        [],
        #layer3.errors(y),
        layer2.errors(y),
        givens={
            #x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            #y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            x: valid_set_x,
            y: valid_set_y
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    #params = layer3.params + layer2.params + layer1.params + layer0.params
    params = layer2.params + layer1.params + layer0.params
    avg_gparams = layer2.avg_gparams + layer1.avg_gparams + layer0.avg_gparams

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]
    
    #  Implementing momentum
    updates = []
    for param, avg_gparam, gparam in zip(params, avg_gparams, grads):
        new_avg_param = (1 - momentum) * avg_gparam + momentum * gparam
        updates.append((param, param - learning_rate * new_avg_param))
        updates.append((avg_gparam, new_avg_param))
    

    train_model = theano.function(
        [], #[index],
        cost,
        updates=updates,
        givens={
            #x: train_set_x[index * batch_size: (index + 1) * batch_size],
            #y: train_set_y[index * batch_size: (index + 1) * batch_size]
            x: train_set_x,
            y: train_set_y
        }
    )
    # end-snippet-1
    
    validation_minibatch_index = 0
    test_minibatch_index = 0
    
    previous_validation_error = -1
    
    previous_features = np.zeros((agent.n_features,1))
    features = np.zeros((1,1))
    previous_action = np.ones((1,1)) * agent.n_actions - 1 # Always start off agent from the lowest learning rate (i.e. last action)
    action = 0
    previous_reward = np.zeros((1,1))
    reward = 0

    current_training_cost = 0
    previous_training_cost = -1
    
    previous_model = [layer0.W.get_value(), layer0.b.get_value(), layer1.W.get_value(), layer1.b.get_value(), layer2.W.get_value(), layer2.b.get_value(), layer3.W.get_value(), layer3.b.get_value()]
    
    # TRAIN MODEL
    lowest_valid_error = np.inf
    for epoch in range(max_epochs):
        #print 'Running epoch ', epoch
        start = time.time()
        train_iterator = dataset_train.get_iterator(batch_size=batch_size, n_batches=training_batches)
        valid_iterator = dataset_valid.get_iterator(batch_size=batch_size, n_batches=validation_batches)
        test_iterator = dataset_test.get_iterator(batch_size=batch_size, n_batches=test_batches)
        
        current_train_set_x = np.zeros((batch_size,image_channels,image_width,image_height), dtype=theano.config.floatX)
        current_train_set_y = np.zeros((batch_size), dtype=np.int32)
        
        for minibatch_index in range(training_batches):
	  #print 'Running batch ', minibatch_index
	  
	  b = train_iterator.get_batch(minibatch_index)
	  #print 'b', b.shape
	  for image_index in range(batch_size):
	      # Retrieve the current image
	      
	      image = b[0][image_index][0]
	      image_size = b[0][image_index][1]
	      label = b[0][image_index][2]
	      
	      # Crop and resize image to have size 256x221:
	      #image_cropped = transformer.preprocess(np.reshape(image, image_size))
	      current_train_set_x[image_index,:,:,:] = np.reshape(transformer.preprocess(np.reshape(image, image_size)), (image_channels, image_width, image_height))
	      current_train_set_y[image_index] = np.squeeze(label)
	  
	  train_set_x.set_value(np.squeeze(np.reshape(current_train_set_x, (batch_size, image_channels*image_width*image_height))))
	  train_set_y.set_value(current_train_set_y)
	  
	  #print 'train_set_x', train_set_x.get_value().shapebatch_size
	  cost = train_model()
	  
          #print 'cost', cost
	  current_training_cost = current_training_cost + cost


	  # Time for RL agent to make an action.
	  if minibatch_index % minibatches_between_RL_actions == 0:

            current_validation_error = 0
            for z in range(minibatches_per_action):
		    current_valid_set_x = np.zeros((batch_size,image_channels,image_width,image_height), dtype=theano.config.floatX)
		    current_valid_set_y = np.zeros((batch_size),dtype=np.int32)
		    b = valid_iterator.get_batch(validation_minibatch_index)
		    validation_minibatch_index = (validation_minibatch_index + 1) % validation_batches
		    
		    for image_index in range(batch_size):
			# Retrieve the current image
			
			image = b[0][image_index][0]
			image_size = b[0][image_index][1]
			label = b[0][image_index][2]
			
			# Crop and resize image to have size 256x221:
			#image_cropped = transformer.preprocess(np.reshape(image, image_size))
			current_valid_set_x[image_index,:,:,:] = np.reshape(transformer.preprocess(np.reshape(image, image_size)), (image_channels, image_width, image_height))
			current_valid_set_y[image_index] = np.squeeze(label)
		    
		    valid_set_x.set_value(np.squeeze(np.reshape(current_valid_set_x, (batch_size, image_channels*image_width*image_height))))
		    valid_set_y.set_value(current_valid_set_y)
	    
		    current_validation_error = current_validation_error + validate_model()
	    
            current_validation_error = current_validation_error / minibatches_per_action
            print 'current_validation_error', current_validation_error
            
            current_training_cost = current_training_cost / minibatches_between_RL_actions
            print 'current_training_cost', current_training_cost

            if useRLAgent==True:
	      reward = np.zeros((1,1))
	      if previous_validation_error > 0:
		reward = reward + current_validation_error - previous_validation_error
	      #print 'reward ', reward
	      
	      features = np.zeros((agent.n_features,1))
	      features[0] = 1 # On if agent is able to revert to previous state, and otherwise off... If off, then reverting to previous state is equal to doing nothing...
	      features[1] = reward
	      features[2] = previous_reward
	      
	      # Change in parameters norm divided by learning rate (this, of course, assumes we keep number of mini-batches constant)
	      parameter_norm_diff = np.linalg.norm(layer0.W.get_value() - previous_model[0])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer0.b.get_value() - previous_model[1])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer1.W.get_value() - previous_model[2])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer1.b.get_value() - previous_model[3])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer2.W.get_value() - previous_model[4])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer2.b.get_value() - previous_model[5])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer3.W.get_value() - previous_model[6])
	      parameter_norm_diff = parameter_norm_diff + np.linalg.norm(layer3.b.get_value() - previous_model[7])
	      parameter_norm_diff = parameter_norm_diff / learning_rate.get_value()
	      features[3] = parameter_norm_diff
	      # relative improvement in training cost
	      if previous_training_cost > 0:
		features[4] = current_training_cost / previous_training_cost 
	      else:
		features[4] = 1

	      # Should also include norm of gradients on training examples, sparsity of feature activations etc...
	      
	      action = agent.make_action(previous_features, previous_action, reward, features)
	      
	      #if action == 5: # Revert parameters to previous state
	      #	  print 'Agent: Resetting parameters to previous state...'
	      #	  layer0.W.set_value(previous_model[0])
	      #	  layer0.b.set_value(previous_model[1])
	      #	  layer1.W.set_value(previous_model[2])
	      #	  layer1.b.set_value(previous_model[3])
	      #	  layer2.W.set_value(previous_model[4])
	      #	  layer2.b.set_value(previous_model[5])
	      #	  layer3.W.set_value(previous_model[6])
	      #	  layer3.b.set_value(previous_model[7])
              #
	      #	  previous_features = features
	      #	  features[0] = 0
	      #	  action = agent.make_action(features, action + np.zeros((1,1)), -reward, features)
	      #else:
              #     print 'Agent: No parameters reset.'


      
	      if action==0:
		learning_rate.set_value(0.03)
	      elif action==1:
		learning_rate.set_value(0.01)
	      elif action == 2:
		learning_rate.set_value(0.003)
	      elif action == 3:
		learning_rate.set_value(0.001)
	      elif action == 4:
		learning_rate.set_value(0.0003)
              if action==5:
                learning_rate.set_value(0.0001)
              elif action==6:
                learning_rate.set_value(0.00003)
              elif action == 7:
                learning_rate.set_value(0.00001)
              elif action == 8:
                learning_rate.set_value(0.000003)
              elif action == 9:
                learning_rate.set_value(0.000001)
	      

	      #if action == 1: # Increase learning rate
	      #    #print 'Agent: Increasing learning rate... ', learning_rate*1.05
	      #    learning_rate = learning_rate * 1.05
	      #elif action == 2: # Decrease learning rate
	      #    #print 'Agent: Decreasing learning rate... ', learning_rate*0.95
	      #    learning_rate = learning_rate * 0.95
	      #else:
		  #print 'Agent: Not doing anything. Just continue training...'
	     

	      previous_validation_error = current_validation_error
	      previous_training_cost = current_training_cost
	      current_training_cost = 0
	      previous_features = features
	      previous_action = np.zeros((1,1)) + action
	      previous_reward = reward
	      previous_model = [layer0.W.get_value(), layer0.b.get_value(), layer1.W.get_value(), layer1.b.get_value(), layer2.W.get_value(), layer2.b.get_value(), layer3.W.get_value(), layer3.b.get_value()]
 
 
            print 'learning_rate', learning_rate.get_value()
 
	    # Compute current test error
            current_test_error = 0
            for z in range(minibatches_per_action):
		    current_test_set_x = np.zeros((batch_size,image_channels,image_width,image_height), dtype=theano.config.floatX)
		    current_test_set_y = np.zeros((batch_size),dtype=np.int32)
		    b = test_iterator.get_batch(test_minibatch_index)
		    test_minibatch_index = (test_minibatch_index + 1) % test_batches
		    
		    for image_index in range(batch_size):
			# Retrieve the current image
			
			image = b[0][image_index][0]
			image_size = b[0][image_index][1]
			label = b[0][image_index][2]
			
			# Crop and resize image to have size 256x221:
			#image_cropped = transformer.preprocess(np.reshape(image, image_size))
			current_test_set_x[image_index,:,:,:] = np.reshape(transformer.preprocess(np.reshape(image, image_size)), (image_channels, image_width, image_height))
			current_test_set_y[image_index] = np.squeeze(label)
		    
		    valid_set_x.set_value(np.squeeze(np.reshape(current_test_set_x, (batch_size, image_channels*image_width*image_height))))
		    valid_set_y.set_value(current_test_set_y)
	    
		    current_test_error = current_test_error + validate_model()
	    
            current_test_error = current_test_error / minibatches_per_action
            print 'current_test_error', current_test_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_learning_rate", type=str, default="0.0001", help="The initial learning rate")
    parser.add_argument("--use_rl_agent", type=str, default='1', help="Whether to use LSPI-Q RL Agent to choose between learning rates")

    return parser.parse_args()
    
    
if __name__ == '__main__':
    args = parse_args()
    
    useRLAgent = False
    if int(args.use_rl_agent) == 1 or str(args.use_rl_agent) == 'True' or str(args.use_rl_agent) == 'true':
       useRLAgent = True
    
    initialLearningRate = float(args.initial_learning_rate)
        
    print 'useRLAgent', useRLAgent
    print 'initialLearningRate', initialLearningRate
    
    # Creation of the datasets
    #dataset_train = Dataset(start=0, stop=6000)
    #dataset_valid = Dataset(start=6000, stop=9000)
    
    dataset_train = Dataset(start=0, stop=15000)
    dataset_valid = Dataset(start=15000, stop=17500)
    dataset_test = Dataset(start=17500, stop=20000)

    
    train(useRLAgent, initialLearningRate, dataset_train, dataset_valid, dataset_test, 10, 50, 1500, 250, 250)

    #print("Starting training...")
    #for epoch in epochs:
    #    print("Epoch {} of {}".format(epoch['number'], num_epochs))
    #    print("  elapsed time:\t\t\t{}".format(epoch['elapsed_time']))
    #    print("  training loss:\t\t{}".format(epoch['train_loss']))
    #    print("  training error:\t\t{} %%".format(epoch['train_error'] * 100))
    #    print("  validation loss:\t\t{}".format(epoch['valid_loss']))
    #    print("  validation error:\t\t{} %%".format(epoch['valid_error'] * 100))

    #    if epoch['number'] >= num_epochs:
    #        break

__author__ = 'adeb'

import os
import numpy
import scipy
import matplotlib.pyplot as plt
import tables
import time
import shutil
import math
import multiprocessing
from sys import getsizeof

from scipy import misc
from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import cache
import theano


class Dataset():
    def __init__(self, start, stop, path=None):
        self.start = start
        self.stop = stop
        self.n_data = self.stop - self.start

        if path is None:
            #path='/data/lisa/exp/serbaniv/OnlineMetaParameterOptimization/data/train_mini.h5'
            #path='/data/lisa/exp/serbaniv/OnlineMetaParameterOptimization/data/train_mini.h5'
            path='/data/lisa/data/dogs_vs_cats/train.h5'
            
        # Locally cache the files before reading them
        path = preprocess(path)
        dataset_cache = cache.datasetCache
        self.path = dataset_cache.cache_file(path)
        self.h5file = tables.openFile(self.path, mode="r")
        node = self.h5file.getNode('/', 'Data')

        # You need at least 12 GB to load entire training set in like this...
        print "   load data on RAM"
        self.h5_x = getattr(node, 'X') #[self.start:self.stop] # To NOT load into ram, remove [self.start:self.stop]
        s = 0
        for img in self.h5_x:
            s += img.nbytes
        print "      size: " + str(s / 1024**2)
        self.h5_s = getattr(node, 's') #[self.start:self.stop]  # To NOT load into ram, remove [self.start:self.stop]
        self.h5_y = getattr(node, 'y') #[self.start:self.stop]  # To NOT load into ram, remove [self.start:self.stop]
        print "      end load"

    def get_iterator(self, batch_size, n_batches=None):
        return DataIterator(self, batch_size, n_batches=n_batches)


class DataIterator(object):
    """
    Manages the batch creation with a scheme
    """
    def __init__(self, dataset, batch_size, n_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        if n_batches is None:
            self.n_batches = dataset.n_data / batch_size
        else:
            self.n_batches = n_batches
        self.batch_id = 0

        self.permutation = numpy.random.permutation(dataset.n_data)

    def get_batch(self, idx_batch):
        """
        Returns a single batch corresponding to idx_batch
        """
        pos = idx_batch * self.batch_size  # To NOT load into ram, plus with self.dataset.start
        indices = self.permutation[pos: pos + self.batch_size]
        indices = self.dataset.start + indices
        return [(self.dataset.h5_x[i], self.dataset.h5_s[i], self.dataset.h5_y[i]) for i in indices], indices

    def get_batches(self, idx_batches):
        """
        Returns several batches correponding to idx_batches
        """
        batches = [0]*len(idx_batches)
        indices = [0]*len(idx_batches)
        for i, idx in enumerate(idx_batches):
            batches[i], indices[i] = self.get_batch(idx)
        return batches, indices
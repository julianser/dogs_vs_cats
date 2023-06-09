"""
A transformer able to rotate and crop images for Pylearn2.
"""
__authors__ = "Iulian Serban"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Iulian Serban"]
__license__ = "3-clause BSD"
__maintainer__ = "Iulian Serban"

import os
import numpy
import tables
from scipy import misc
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.iteration import SubsetIterator, resolve_iterator_class
from pylearn2.space import VectorSpace, IndexSpace, Conv2DSpace, CompositeSpace
from pylearn2.utils import safe_izip, wraps
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng
from pylearn2.datasets import cache
from pylearn2.datasets.dataset import Dataset

class BaseImageTransformer(object):
    """
    An object that preprocesses an image on-the-fly

    Notes
    -----
    Images are expected to be in ('b', 0, 1, 'c') format.
    """
    def get_shape(self):
        """
        Returns the shape of a preprocessed image
        """
        raise NotImplementedError()

    def preprocess(self, image):
        """
        Applies preprocessing on-the-fly

        Parameters
        ----------
        image : numpy.ndarray
            Image to preprocess
        """
        raise NotImplementedError()

    def __call__(self, image):
        return self.preprocess(image)

class RandomCropAndRotation(BaseImageTransformer):
    """
    Rotates and crops a square at random on a rescaled version of the image

    Parameters
    ----------
    scaled_size : int
        Size of the smallest side of the image after rescaling
    crop_size : int
        Size of the square crop. Must be bigger than scaled_size.
    max_rot_angle : int
        A positive rotation angle is drawn uniformly random from [-max_rot_angle, max_rot_angle], and the image is rotated with this before cropping.
    rng : int or rng, optional
        RNG or seed for an RNG
    """
    _default_seed = 2015 + 1 + 18

    def __init__(self, scaled_size, crop_size, max_rot_angle, rng=_default_seed):
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.max_rot_angle = int(max_rot_angle)
        assert self.scaled_size > self.crop_size
        self.rng = make_np_rng(rng, which_method="random_integers")

    @wraps(BaseImageTransformer.get_shape)
    def get_shape(self):
        return (self.crop_size, self.crop_size)

    @wraps(BaseImageTransformer.preprocess)
    def preprocess(self, image):
        small_axis = numpy.argmin(image.shape[:-1])
        ratio = (1.0 * self.scaled_size) / image.shape[small_axis]

        resized_image = misc.imresize(image, ratio)

        rotation_angle = self.rng.randint(low=-self.max_rot_angle, high=self.max_rot_angle)

        rotated_image = misc.imrotate(resized_image, angle=rotation_angle, interp='bilinear') #, interp='bilinear'

        max_i = rotated_image.shape[0] - self.crop_size
        max_j = rotated_image.shape[1] - self.crop_size
        i = self.rng.randint(low=0, high=max_i)
        j = self.rng.randint(low=0, high=max_j)
        return rotated_image[i : i + self.crop_size, j : j + self.crop_size, :]

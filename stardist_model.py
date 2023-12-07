from __future__ import print_function, unicode_literals, absolute_import, division


from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D
from tqdm.notebook import tqdm  
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt


from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D


class StarDist():
    def __init__(self, n_rays=32, use_gpu=False, grid=(2,2)):
        self.n_rays = n_rays
        self.use_gpu = use_gpu
        self.grid = grid
        self.score = list()
        self.taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.scores = list()

    def train(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, epochs, steps_per_epoch, cv=False):
        from tqdm.notebook import tqdm  # Use this if you're in a Jupyter notebook

        n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
        conf = Config2D(
            n_rays= self.n_rays,
            grid= self.grid,
            use_gpu= self.use_gpu,
            n_channel_in= n_channel,
            train_patch_size= (len(X_trn[0]), len(X_trn[0])),
            train_learning_rate= learning_rate
        )
        model = StarDist2D(conf, name='stardist', basedir='models')
        model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=epochs, steps_per_epoch=steps_per_epoch)

        model.optimize_thresholds(X_val, Y_val)
        Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(X_val)]

        stats = [matching_dataset(Y_val[i], Y_val_pred[i], thresh=t, show_progress=False) for i, t in enumerate(self.taus)]
        if cv:
            self.scores.append(
                {
                "learning_rate":learning_rate,
                "epochs":epochs,
                "steps_per_epoch":steps_per_epoch, 
                "stats":stats
                }
                )
        else:
            self.score = stats

    def CV(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epoch_settings, step_epochs, cv=True):
        for learning_rate in learning_rates:
            for epoch, steps_per_epoch in zip(epoch_settings, step_epochs):
                self.train(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, epoch, steps_per_epoch, cv)
        return self.scores
    


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1,)), np.reshape(y + dy, (-1,))

    return map_coordinates(image, indices, order=1).reshape(shape)



def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)

    x = elastic_transform(x, alpha=100, sigma=10)

    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)

    return x, y


def slice_numpy_array_list(array_list):
    sliced_arrays = []

    for image_array in array_list:
        height, width = image_array.shape

        crop_size = 144
        stride = crop_size

        for row in range(0, height, stride):
            for col in range(0, width, stride):
                upper = row
                left = col
                lower = min(row + crop_size, height)
                right = min(col + crop_size, width)

                subarray = image_array[upper:lower, left:right]
                subarray_height, subarray_width = subarray.shape

                if subarray_height == crop_size and subarray_width == crop_size:
                    # Only append if the subarray is exactly 144x144
                    sliced_arrays.append(subarray)

    return sliced_arrays

def load_splice(mask, input):
    X = list(glob(mask+'/*.tif'))
    Y = list(glob(input+'/*.tif'))

    X = list(map(imread,X))
    Y = list(map(imread,Y))

    X = slice_numpy_array_list(X)
    Y = slice_numpy_array_list(Y)
    return X, Y


def train_test_split(X, Y):
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]  #If no third dim. then number of channels = 1. Otherwise get the num channels from the last dim.
    #Normalize input images and fill holes in masks
    axis_norm = (0,1)   # normalize channels independently
    #axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))
    return X_trn, Y_trn, X_val, Y_val
import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import json
import glob
import shutil
import os
from glob import glob
from tqdm import tqdm
from tqdm.notebook import tqdm  
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from tqdm.notebook import tqdm  
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from keras.callbacks import TensorBoard
from scipy.ndimage import label
    
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json


class StarDist():
    def __init__(self, type_op="n", name="stardist", n_rays=32, use_gpu=True, grid=(2,2)):
        self.n_rays = n_rays
        self.use_gpu = use_gpu
        self.grid = grid
        self.score = list()
        self.taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.scores = list()
        self.type_op = type_op
        self.name = name

    def train(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, epochs, steps_per_epoch, cv=False):

        n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
        
        if self.type_op == "pretrained":
            StarDist2D.from_pretrained()
            model_loading = StarDist2D.from_pretrained('2D_versatile_fluo')
            source_dir = "/root/.keras/models/StarDist2D/2D_versatile_fluo"
            destination_dir = "models/finetuned"
            logs = "models/finetuned/logs"

            os.makedirs(destination_dir, exist_ok=True)
            os.makedirs(logs, exist_ok=True)

            file_paths = glob(os.path.join(source_dir, "*"))

            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                new_path = os.path.join(destination_dir, file_name)
                shutil.move(file_path, new_path)
                print(f"Moved: {file_path} to {new_path}")
                
            conf = Config2D(n_dim=2, axes='YXC', n_channel_in=1, n_channel_out=33,
                                         train_checkpoint='weights_best.h5', train_checkpoint_last='weights_last.h5',
                                         train_checkpoint_epoch='weights_now.h5', n_rays=32, grid=(2, 2), backbone='unet',
                                         n_classes=None, unet_n_depth=3, unet_kernel_size=[3, 3], unet_n_filter_base=32,
                                         unet_n_conv_per_depth=2, unet_pool=[2, 2], unet_activation='relu',
                                         unet_last_activation='relu', unet_batch_norm=False, unet_dropout=0.0, unet_prefix='',
                                         net_conv_after_unet=128, net_input_shape=[None, None, 1], net_mask_shape=[None, None, 1],
                                         train_shape_completion=False, train_completion_crop=32, train_patch_size=[256, 256],
                                         train_background_reg=0.0001, train_foreground_only=0.9, train_sample_cache=True,
                                         train_dist_loss='mae', train_loss_weights=[1, 0.2], train_class_weights=(1, 1),
                                         train_epochs=800, train_steps_per_epoch=400, train_learning_rate=0.0003,
                                         train_batch_size=8, train_n_val_patches=None, train_tensorboard=True,
                                         train_reduce_lr={'factor': 0.5, 'patience': 80, 'min_delta': 0, 'verbose': True},
                                         use_gpu=False)            
            self.model = StarDist2D(None, name=self.name, basedir='models')
            self.model.config.train_learning_rate = 0.00003
        else:
            conf = Config2D(
            n_rays= self.n_rays,
            grid= self.grid,
            use_gpu= self.use_gpu,
            n_channel_in= n_channel,
            train_patch_size= (len(X_trn[0]), len(X_trn[0])),
            train_learning_rate= learning_rate)
            self.model = StarDist2D(conf, name=self.name, basedir='models')


        history = self.model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=epochs, steps_per_epoch=steps_per_epoch)
        
        self.model.optimize_thresholds(X_val, Y_val)
        Y_val_pred = [self.model.predict_instances(normalize(x))[0] for x in X_val]
        stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for i, t in enumerate(self.taus)]
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
            


        return history

    def CV(self, X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epoch_settings, step_epochs, cv=True):
        self.histories = list()
        convergence_threshold = 1e-3  

        for learning_rate in learning_rates:
            for max_epochs, steps_per_epoch in zip(epoch_settings, step_epochs):
                history = self.train(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rate, max_epochs, steps_per_epoch, cv)
                loss_diff = abs(history.history['loss'][-1] - history.history['loss'][-2])
                if loss_diff < convergence_threshold:
                    print(f"Convergence reached at epoch {max_epochs}")
                    break

                self.histories.append(history)

        return self.histories

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

        crop_size = 256
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

def load_splice(inputs, masks):
    X = [inputs+"in1-sec.tif", inputs+"mip_2.tif"]
    Y = [masks+ "in1-sec_lab.tif", masks +"mip_2_lab.tif"]

    X = list(map(imread,X))
    Y = list(map(imread,Y))
    
    binary_array = (Y[0] > 0).astype(int)
    binary_array_2 = (Y[1] > 0).astype(int)

    labeled_array, num_features = label(binary_array)
    labeled_array_2, num_features_2 = label(binary_array_2)

    X = slice_numpy_array_list(X)
    Y = slice_numpy_array_list([labeled_array, labeled_array_2])
    
    return X, Y

def load_splice_mip(inputs, masks):
    X = [inputs+"mip_2.tif"]
    Y = [masks +"mip_2_lab.tif"]

    X = list(map(imread,X))
    Y = list(map(imread,Y))
    
    binary_array = (Y[0] > 0).astype(int)

    labeled_array, num_features = label(binary_array)

    X = slice_numpy_array_list(X)
    Y = slice_numpy_array_list([labeled_array])
    
    return X, Y

def train_test_split(X, Y):
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]  #If no third dim. then number of channels = 1. Otherwise get the num channels from the last dim.

    axis_norm = (0,1)  

    if (n_channel > 456):
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


"""def plot_result(result):
    fig_performance = go.Figure()
    fig_counts = go.Figure()

    performance_metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
    count_metrics = ('fp', 'tp', 'fn')

    for m in performance_metrics:
        metric_values = [r[m] for r in result]
        thresh_values = [r['thresh'] for r in result]
        fig_performance.add_trace(go.Scatter(x=thresh_values, y=metric_values, mode='lines+markers', name=m))

    # Update layout for performance metrics plot
    fig_performance.update_layout(title="Performance Metrics",
                                  xaxis_title="IoU threshold τ",
                                  yaxis_title="Metric value")

    # Plotting count metrics
    for m in count_metrics:
        metric_values = [r[m] for r in result]
        thresh_values = [r['thresh'] for r in result]
        fig_counts.add_trace(go.Scatter(x=thresh_values, y=metric_values, mode='lines+markers', name=m))

    # Update layout for count metrics plot
    fig_counts.update_layout(title="Count Metrics",
                             xaxis_title="IoU threshold τ",
                             yaxis_title="Number")

    return fig_performance, fig_counts"""


def plot_result(result):
    # Create a subplot with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Performance Metrics', 'TP/FP/FN'))

    performance_metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
    count_metrics = ('fp', 'tp', 'fn')

    # Plotting performance metrics in the first row
    for m in performance_metrics:
        metric_values = [r[m] for r in result]
        thresh_values = [r['thresh'] for r in result]
        fig.add_trace(go.Scatter(x=thresh_values, y=metric_values, mode='lines+markers', name=m), row=1, col=1)

    # Plotting count metrics in the second row
    for m in count_metrics:
        metric_values = [r[m] for r in result]
        thresh_values = [r['thresh'] for r in result]
        fig.add_trace(go.Scatter(x=thresh_values, y=metric_values, mode='lines+markers', name=m), row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text="TP/FP/FN")
    fig.update_xaxes(title_text="IoU threshold τ", row=1, col=1)
    fig.update_xaxes(title_text="IoU threshold τ", row=2, col=1)
    fig.update_yaxes(title_text="Metric value", row=1, col=1)
    fig.update_yaxes(title_text="Number", row=2, col=1)

    return fig


    
def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    
def plot_history(history):
    fig = go.Figure()

    global_min_loss = float('inf')
    global_min_epoch = 0

    val_loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = list(range(len(val_loss)))
    print(loss)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name=f'Loss Validation- '))
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', name=f'Loss- '))

    min_loss_epoch = val_loss.index(min(val_loss))
    min_loss = val_loss[min_loss_epoch]

    if min_loss < global_min_loss:
        global_min_loss = min_loss
        global_min_epoch = min_loss_epoch

    fig.add_shape(type='line',
                  x0=global_min_epoch, y0=0, x1=global_min_epoch, y1=global_min_loss,
                  line=dict(color='Red', width=2, dash='dash'),
                  xref='x', yref='y')

    fig.update_layout(title='Metrics per Validation Loss', xaxis_title='Epoch', yaxis_title='Validation Loss',
                      annotations=[dict(x=global_min_epoch, y=global_min_loss,
                                        text=f"Epoch: {global_min_epoch}",
                                        showarrow=True, arrowhead=1)])

    fig.show()

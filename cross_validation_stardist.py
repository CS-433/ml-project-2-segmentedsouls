from stardist_model import *
from stardist.plot import render_label
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
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
from stardist.plot import render_label



def cross_validate(model, dataset, epochs, learning_rates, epochs_per_step):
    os.makedirs("results_", exist_ok=True)
    if model == "MIP":
        X, Y = load_splice_mip(dataset+"inputs/", dataset + "masks/")

        X_trn, Y_trn, X_val, Y_val = train_test_split(X, Y)

        model = StarDist("_","mip", 32, True, (2,2))
        history_mip = model.CV(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epochs, epochs_per_step)

        history = [history.history for history in history_mip]

        with open("results_/history_mip.json", "w") as file:
            json.dump(history, file, default=default_converter)

            
        Y_val_pred = [model.model.predict_instances(x, n_tiles=model.model._guess_n_tiles(x))[0] for x in tqdm(X_val)]
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scores = []
        for Y_true, Y_pred in zip(Y_val, Y_val_pred):
            stats = [matching(Y_true.astype(int), Y_pred.astype(int), thresh=t) for t in taus]
            stats = [s._asdict() for s in stats]
            scores.append(stats)
            
            numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                            'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores

        thresh_values = sorted(set(item['thresh'] for sublist in data for item in sublist))
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}

        for sublist in data:
            for item in sublist:
                thresh = item['thresh']
                for key in numeric_keys:
                    sums[thresh][key] += item[key]
                counts[thresh] += 1

        averages = {thresh: {key: sums[thresh][key] / counts[thresh] for key in sums[thresh]} for thresh in thresh_values}

        result = []
        for thresh in thresh_values:
            avg_dict = averages[thresh]
            avg_dict['thresh'] = thresh  # Adding the 'thresh' value back
            result.append(avg_dict)

        with open("results_/stats_mip.json", "w") as file:
            json.dump(result, file, default=default_converter)

    if model == "MIP_FOCUS":
        X, Y = load_splice(dataset+"inputs/", dataset + "masks/")

        X_trn, Y_trn, X_val, Y_val = train_test_split(X, Y)

        model = StarDist("_","mipfocus", 32, True, (2,2))
        history_mip = model.CV(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epochs, epochs_per_step)

        history = [history.history for history in history_mip]

        with open("results_/history_mipfocus.json", "w") as file:
            json.dump(history, file, default=default_converter)
            
        Y_val_pred = [model.model.predict_instances(x, n_tiles=model.model._guess_n_tiles(x))[0] for x in tqdm(X_val)]
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scores = []
        for Y_true, Y_pred in zip(Y_val, Y_val_pred):
            stats = [matching(Y_true.astype(int), Y_pred.astype(int), thresh=t) for t in taus]
            stats = [s._asdict() for s in stats]
            scores.append(stats)
            
            numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                            'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores

        thresh_values = sorted(set(item['thresh'] for sublist in data for item in sublist))
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}

        for sublist in data:
            for item in sublist:
                thresh = item['thresh']
                for key in numeric_keys:
                    sums[thresh][key] += item[key]
                counts[thresh] += 1

        averages = {thresh: {key: sums[thresh][key] / counts[thresh] for key in sums[thresh]} for thresh in thresh_values}

        result = []
        for thresh in thresh_values:
            avg_dict = averages[thresh]
            avg_dict['thresh'] = thresh  # Adding the 'thresh' value back
            result.append(avg_dict)

        with open("results_/stats_mipfocus.json", "w") as file:
            json.dump(result, file, default=default_converter)

    if model == "FINETUNE":
        X, Y = load_splice_mip(dataset+"inputs/", dataset+"masks/")

        X_trn, Y_trn, X_val, Y_val = train_test_split(X, Y)

        model = StarDist("pretrained","finetuned", 32, True, (2,2))

        history = model.CV(X_trn, Y_trn, X_val, Y_val, augmenter, learning_rates, epochs, epochs_per_step)
        history = [history.history for history in history]

        with open("results_/history_finetuned.json", "w") as file:
            json.dump(history, file, default=default_converter)

        Y_val_pred = [model.model.predict_instances(x, n_tiles=model.model._guess_n_tiles(x))[0] for x in tqdm(X_val)]
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scores = []
        for Y_true, Y_pred in zip(Y_val, Y_val_pred):
            stats = [matching(Y_true.astype(int), Y_pred.astype(int), thresh=t) for t in taus]
            stats = [s._asdict() for s in stats]
            scores.append(stats)
            
        numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                        'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores

        thresh_values = sorted(set(item['thresh'] for sublist in data for item in sublist))
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}

        for sublist in data:
            for item in sublist:
                thresh = item['thresh']
                for key in numeric_keys:
                    sums[thresh][key] += item[key]
                counts[thresh] += 1

        averages = {thresh: {key: sums[thresh][key] / counts[thresh] for key in sums[thresh]} for thresh in thresh_values}

        result = []
        for thresh in thresh_values:
            avg_dict = averages[thresh]
            avg_dict['thresh'] = thresh  # Adding the 'thresh' value back
            result.append(avg_dict)

        with open("results_/stats_finetuned.json", "w") as file:
            json.dump(result, file, default=default_converter)
    
    else:
        print("Your model is not supported")

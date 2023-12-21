import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stardist_model import *
import json



def visualisation(datafolder, type_):
    if type_ == "FINETUNED":
        historyfile = datafolder+"history_finetuned.json"
        with open(historyfile, 'r') as file:
            data = json.load(file)
            
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Validation Loss', 'Training Loss'))

        for item, lr in zip(data, [0.0003, 0.0002]):
            lr_formatted = "{:.1e}".format(lr)
            fig.add_trace(go.Scatter(x=list(range(len(item['loss']))), y=item['loss'], mode='lines', name=f'LR: {lr_formatted} - Training Loss'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(len(item['val_loss']))), y=item['val_loss'], mode='lines', name=f'LR: {lr_formatted} - Validation Loss'), row=2, col=1)
        fig.update_layout(title='Metrics per Learning Rate - Finetuned', xaxis_title='Epoch', yaxis_title='Value', width= 800)
        fig.show()

        statsfile= datafolder+"stats_finetuned.json"
        with open(datafolder+"stats_finetuned.json", "r") as file:
            scores = json.load(file)

        numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                        'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores
        thresh_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}


        for item in data:
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

        fig2 = plot_result(result)
        fig2.show()



    if type_ == "MIPFOCUS":
        historyfile = datafolder+"history_mipfocus.json"

        with open(historyfile, 'r') as file:
            data = json.load(file)
            
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Validation Loss', 'Training Loss'))

        for item, lr in zip(data, [0.0003, 0.0002]):
            lr_formatted = "{:.1e}".format(lr)
            fig.add_trace(go.Scatter(x=list(range(len(item['loss']))), y=item['loss'], mode='lines', name=f'LR: {lr_formatted} - Training Loss'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(len(item['val_loss']))), y=item['val_loss'], mode='lines', name=f'LR: {lr_formatted} - Validation Loss'), row=2, col=1)
        fig.update_layout(title='Metrics per Learning Rate', xaxis_title='Epoch', yaxis_title='Value', width= 800)
        fig.show()

        statsfile= datafolder+"stats_mipfocus.json"
        with open(statsfile, "r") as file:
            scores = json.load(file)

        numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                        'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores
        thresh_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}


        for item in data:
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


        fig2 = plot_result(result)
        fig2.show()

    if type_ == "MIP":

        historyfile = datafolder+"history_mip.json"
        with open(historyfile, 'r') as file:
            data = json.load(file)
            
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Validation Loss', 'Training Loss'))
        for item, lr in zip(data, [0.0003, 0.0002]):
            lr_formatted = "{:.1e}".format(lr)
            fig.add_trace(go.Scatter(x=list(range(len(item['loss']))), y=item['loss'], mode='lines', name=f'LR: {lr_formatted} - Training Loss'), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(len(item['val_loss']))), y=item['val_loss'], mode='lines', name=f'LR: {lr_formatted} - Validation Loss'), row=2, col=1)

        fig.update_layout(title='Validation and Training Loss per Learning Rate', xaxis_title='Epoch', yaxis_title='', width= 800)

        fig.show()

        statsfile= datafolder+"stats_mip.json"
        with open(statsfile, "r") as file:
            scores = json.load(file)

        numeric_keys = ['fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 
                        'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality']

        data = scores
        thresh_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        sums = {thresh: {key: 0 for key in numeric_keys} for thresh in thresh_values}
        counts = {thresh: 0 for thresh in thresh_values}


        for item in data:
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


        fig2 = plot_result(result)
        fig2.show()

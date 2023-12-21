# Performing a 2D Nucleus Segmentation With Cellpose and StarDist on Xenium Data

## Overview
This document provides a detailed overview of the processes and resources utilized for performing 2D nucleus segmentation using Cellpose and StarDist on Xenium data.


## Stardist: 

### Download weights and resources here and add the project folder inside the repo (same level as all the other folders eg;inputs, masks): https://drive.google.com/file/d/1CpCZ1GZamGl3kcfPDySYkWi4fgI56g3O/view?usp=sharing
## Files Description

### `cross_validation_stardist`
This script is used for cross-validation of the StarDist model. It contains routines to systematically evaluate the model's performance across different subsets of the data.

### `visualisation_stardist`
This script is dedicated to visualizing the data in the context of the StarDist model. It provides functions to plot the data, model predictions, and potentially compare these with ground truth annotations.

### `stardist_model`
This script includes the code for training the StarDist model. It sets up the model architecture, compiles it with the necessary configurations, and trains it on the provided dataset.

### `stardist.ipynb`
This Jupyter notebook serves as a comprehensive guide and demonstration of the entire workflow. It walks through cross-validation, data visualization, model training, and evaluation, providing a hands-on experience with the StarDist model.
You can choose what you whant to try by changing the global variable in the beggining of the notebook


## Cellpose: Resources and Files
Below is a comprehensive list of the key folders and files integral to the Cellpose training and evaluation:

### Notebooks
- **`CellposeTrainingArena.ipynb`**: Contains code for training models and collecting data. It's pivotal for cross-validation across various learning rates and epochs.
- **`cellpose.ipynb`**: Dedicated to the evaluation of our trained Cellpose model, this notebook includes several visualizations.

### Training Directory: `cellposetraining/`
This directory encompasses all the necessary data for training and validating the Cellpose model.

#### Subdirectories and Content
- **`cellposetraining/control/`**: 
  - Contains images and masks for validating the trained Cellpose model.
  - `final_segmentation/`: Includes the Cellpose-produced masks for `mip_2.tif`.
  - `images/`: Contains the `mip_2.tif` image.
  - `masks/`: Houses manually segmented/ground truth masks for `mip_2.tif`.

- **Validation Data**: 
  - `cellposetraining/{morphology}_validate_mask` and `cellposetraining/{morphology}_validate`: These folders contain the validation data used during Cellpose training.

- **Human-in-the-Loop Training Data**: 
  - `cellposetraining/traindataHIL` and `cellposetraining/trainingdataHIL_mip`: Includes training data generated from our Human-in-the-Loop (HIL) iterations.


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12918895&assignment_repo_type=AssignmentRepo)

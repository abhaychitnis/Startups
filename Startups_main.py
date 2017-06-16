
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


preprocessing_override = pd.read_csv('Startups_override.csv')

# Importing the dataset
dataset_X = pd.read_csv('Startups.csv')
dataset_X_verify = dataset_X
dataset_y = dataset_X['Profit']

del dataset_X['Profit']
del preprocessing_override['Profit']

import preprocess_data as prd

preprocessed_data = prd.preprocess_data(dataset_X, dataset_y, preprocessing_override, dataset_X_verify)

X = preprocessed_data["X"]
y = preprocessed_data["y"]

# import get_best_model as bfm
import get_best_regression_model as bfm

best_fit_model = bfm.get_best_model(X,y)

import plot_best_model as pbm

pbm.plot_best_model(X,y,best_fit_model)

#print (best_fit_model)

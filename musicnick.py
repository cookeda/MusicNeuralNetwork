#!/usr/bin/python3

import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.backend import clear_session
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, kullback_leibler_divergence
import tensorflow as tf
import warnings
import matplotlib as mpl
from keras.layers import Normalization

mpl.use('TkAgg')
from helper import *


# Things to do: 
# Condense into a general function, with the file_name, target, and the model as parameters
# Make a dictionary, with the structure:
# prob_dict = {
#   "loudness" = (model, data_file),
#   "spectral" = (model, data_file)
# }

def make_model(target, x_val):
    normalization_layer = Normalization(axis = 1)
    normalization_layer.adapt(x_val)
    
    if (target == "loudness"):
        """
        This begins the definition and description of the model for my loudness data.
        """
        model = Sequential([
            Input(129),
            normalization_layer,
            Dense(512, activation = 'tanh'),
            Dense(256, activation = 'tanh'),
            Dense(128, activation = 'relu'),
            Dense(64, activation = 'relu'),
            Dense(32, activation = 'relu'),
            Dense(16, activation = 'relu'),
            Dense(8, activation = 'relu'),
            
            Dense(1, activation = 'linear')
        ])

    else:
        """
        This begins the definition and description of the model for my spectral data.
        """
        model = Sequential([
            Input(129),
            normalization_layer,
            Dense(512, activation = 'tanh'),
            Dense(256, activation = 'tanh'),
            Dense(128, activation = 'relu'),
            Dense(64, activation = 'relu'),
            Dense(32, activation = 'relu'),
            Dense(16, activation = 'relu'),
            
            Dense(16, activation = 'linear')
        ]) 

    model.compile(loss = "mse", optimizer = "adam")
    model.summary()
    return(model)

def solve(data_f, target):
    output_path = "./output/"
    x_train, y_train = get_xy(target, subset = 'train')
    x_valid, y_valid = get_xy(target, subset = 'valid')
    
    clear_session()

    # Setting up model callbacks
    model_checkpoint = setup_model_checkpoints(output_path)
    early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose = 1)
    
    model = make_model(target, x_valid)
    
    history = model.fit(
        x_train, y_train, 
        validation_data=[x_valid, y_valid],
        callbacks=[model_checkpoint, early_stopping]
    )
    model, best_epoch, best_loss = get_best_model(output_path)
    
    # Plot the history, visualize things, and save the model. 
    plot_history(history=history, output_path=output_path)
    visualize(model, endpoint=target, subset='valid', output_path=output_path)

    model.save(output_path + target + ".h5")


if __name__ == '__main__':
    prob_dict = {
        "loudness" : "./data/music_loudness.npz",
        "spectral" : "./data/music_spectral.npz"
    }

    solve(prob_dict["loudness"], "loudness")
    solve(prob_dict["spectral"], "spectral")

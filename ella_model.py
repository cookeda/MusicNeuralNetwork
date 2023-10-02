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
from keras.src.layers import Normalization

mpl.use('TkAgg')
from helper import *


def loudness(file_name):
    """
    An example applying linear regression to the loudness problem.

    This example uses early stopping on the validation loss.

    :return: None
    """
    endpoint = 'loudness'
    output_path = f'models/music_{endpoint}_linear'

    x_train, y_train = get_xy(endpoint, subset='train')
    x_valid, y_valid = get_xy(endpoint, subset='valid')

    clear_session()

    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    norm_layer = Normalization(axis=1)
    norm_layer.adapt(x_valid)

    # create linear model
    model = Sequential([
        Input(129),
        norm_layer,
        Dense(512, activation='tanh'),
        Dense(256, activation='tanh'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),

        Dense(1, activation='linear')
    ])

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid],
                        callbacks=[model_checkpoint, early_stopping])
    model, best_epoch, best_loss = get_best_model(output_path)

    # plot history
    plot_history(history=history, output_path=output_path)
    visualize(model, endpoint=endpoint, subset='valid', output_path=output_path)

    model.save(file_name)


def chroma(file_name):
    """
    :return: None
    """
    endpoint = 'chroma'
    output_path = f'models/music_{endpoint}_linear'

    x_train, y_train = get_xy(endpoint, subset='train')
    x_valid, y_valid = get_xy(endpoint, subset='valid')

    clear_session()

    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    norm_layer = Normalization(axis=1)
    norm_layer.adapt(x_valid)

    # create linear model
    model = Sequential([
        Input(129),
        norm_layer,
        # Dense(1048, activation='tanh'), did not improve
        Dense(512, activation='tanh'),
        Dense(256, activation='tanh'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),

        Dense(1, activation='linear')
    ])

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid],
                        callbacks=[model_checkpoint, early_stopping])
    model, best_epoch, best_loss = get_best_model(output_path)

    # plot history
    plot_history(history=history, output_path=output_path)
    visualize(model, endpoint=endpoint, subset='valid', output_path=output_path)

    model.save(file_name)


if __name__ == '__main__':
    loudness("loudness.h5")
    chroma("chroma.h5")

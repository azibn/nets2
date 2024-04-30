import os
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
from astropy.table import Table, Column

"""Script is a framework for a neural network."""

def model(layers=None, optimizer='adam', loss='binary_crossentropy', metrics=None, cadences=None):
    ## consider categorical_crossentropy for loss
    ## consider softmax for activation instead of sigmoid
    
    """
    Creates a Tensorflow keras model with either provided layers or default layers.

    Parameters
    ----------
    layers : list, optional
        List of keras.layers for the NN.
    optimizer : str, optional
        Optimizer used to compile the keras model. Default is 'adam'.
    loss : str, optional
        Loss function used to compile keras model. Default is 'binary_crossentropy'.
    metrics: list, optional
        Metrics used to train the keras model. If None, metrics are ['accuracy', 'precision', 'recall'].
    cadences: int, optional
        Number of cadences for input shape.

    Returns
    -------
    model : tensorflow.python.keras.engine.sequential.Sequential
        The created keras model.
    """
    model = keras.models.Sequential()

    if layers is None:
        filter1 = 16
        filter2 = 64
        dense = 32
        dropout = 0.2

        model.add(tf.keras.layers.LSTM(filters=filter1, kernel_size=7,
                                         activation='relu', padding='same',
                                         input_shape=(cadences, 1)))
        #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.LSTM(filters=filter2, kernel_size=3,
                                         activation='relu', padding='same'))
        #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(dropout))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(dense, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        for i in layers:
            model.add(i)

    if metrics is None:
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    else:
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

    return model

def load(model):
    """
    Loads a pre-trained keras model.

    Parameters
    ----------
    modelname : str
        Path and filename of the model to load.

    Returns
    -------
    model : tensorflow.python.keras.engine.sequential.Sequential
        The loaded keras model.
    """
    model = keras.models.load_model(model)
    return model

def train(model, train_data, train_labels, val_data, val_labels, epochs=50, batch_size=64, shuffle=False):
    """
    Trains the keras model.

    Parameters
    ----------
    model : tensorflow.python.keras.engine.sequential.Sequential
        The keras model to train.
    train_data : np.ndarray
        Training data.
    train_labels : np.ndarray
        Labels for training data.
    val_data : np.ndarray
        Validation data.
    val_labels : np.ndarray
        Labels for validation data.
    epochs : int, optional
        Number of epochs to train for. Default is 15.
    batch_size : int, optional
        Batch size for training. Default is 64.
    shuffle : bool, optional
        Whether to shuffle the training data. Default is False.

    Returns
    -------
    history : dict
        The training history of the model.
    """
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                        validation_data=(val_data, val_labels))
    return history.history

def save(model, output_dir='neural-net/', model_fmt):
    """
    Saves the given model to output directory.

    Parameters
    ----------
    model : tensorflow.python.keras.engine.sequential.Sequential
        The keras model to save.
    output_dir : str
        Path to the output directory.
    model_fmt : str
        The format for the model filename.
    """
    model.save(os.path.join(output_dir, model_fmt))

# Define other functions (e.g., predict, cross_validation, etc.) similarly

# Example usage:
# model = create_nn_model(layers=None, optimizer='adam', loss='binary_crossentropy',
#                          metrics=None, cadences=10)
# history = train_nn_model(model, train_data, train_labels, val_data, val_labels, epochs=15, batch_size=64, shuffle=False)
# save_nn_model(model, 'output_directory', 'model_filename.h5')


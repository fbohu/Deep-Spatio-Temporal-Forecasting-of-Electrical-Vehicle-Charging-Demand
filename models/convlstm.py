

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import activations, initializers, constraints, regularizers


def get_conv_lstm(args):
    model = Sequential([
        keras.Input(
            #shape=(None, 5, 5, 1)),  # Variable-length sequence of 5x5x1 frames
            shape=args['input_shape']),
        layers.TimeDistributed(layers.Conv2D(args['filters'],3, activation='relu',padding='same')),
        layers.TimeDistributed(layers.Flatten()),
        layers.BatchNormalization(),
        layers.LSTM(args['lstm_size'], dropout=0.1),
        layers.BatchNormalization(),
        layers.Dense(units=args["forecast_horizon"]*args['input_shape'][1]*args['input_shape'][2]),
        layers.Reshape((args['input_shape'][1],args['input_shape'][2],args["forecast_horizon"]))
    ])

    return model
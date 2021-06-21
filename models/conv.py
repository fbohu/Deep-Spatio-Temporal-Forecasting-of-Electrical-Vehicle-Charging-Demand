
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import activations, initializers, constraints, regularizers


def get_conv_model(args):
    
    model = Sequential([
        layers.Conv2D(args['filters'],3, activation='relu', padding='same', input_shape=(args['input_shape'])),
        layers.BatchNormalization(),
        tf.keras.layers.Dense(units=args['forecast_horizon'])
    ])

    return model


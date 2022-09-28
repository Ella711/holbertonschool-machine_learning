#!/usr/bin/env python3
"""
0. Inception Block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block

    A_prev: output from the previous layer
    filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        F1: number of filters in the 1x1 conv
        F3R: number of filters in the 1x1 conv before the 3x3 conv
        F3: number of filters in the 3x3 conv
        F5R: number of filters in the 1x1 conv before the 5x5 conv
        F5: number of filters in the 5x5 conv
        FPP: number of filters in the 1x1 conv after the max pooling
        (output shape after max pooling:
        outputshape = math.floor((inputshape - 1) / strides) + 1)
    Use a rectified linear activation (ReLU)

    Returns: concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding="same",
                            activation="relu")(A_prev)
    conv3a = K.layers.Conv2D(filters=F3R, kernel_size=1, padding="same",
                             activation="relu")(A_prev)
    conv3b = K.layers.Conv2D(filters=F3, kernel_size=3, padding="same",
                             activation="relu")(conv3a)
    conv5a = K.layers.Conv2D(filters=F5R, kernel_size=1, padding="same",
                             activation="relu")(A_prev)
    conv5b = K.layers.Conv2D(filters=F5, kernel_size=5, padding="same",
                             activation="relu")(conv5a)
    pool = K.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(A_prev)
    convp = K.layers.Conv2D(filters=FPP, kernel_size=1, padding="same",
                            activation="relu")(pool)

    return K.layers.Concatenate(axis=-1)([conv1, conv3b, conv5b, convp])

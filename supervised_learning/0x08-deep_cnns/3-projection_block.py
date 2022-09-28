#!/usr/bin/env python3
"""
 3. Projection Block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
        for Image Recognition (2015)

    A_prev: output from the previous layer
    filters: tuple or list containing F11, F3, F12, respectively:
        F11: number of filters in the first 1x1 conv
        F3: number of filters in the 3x3 conv
        F12: number of filters in the second 1x1 conv as well as the
            1x1 conv in the shortcut connection
    s: stride of the first conv in both main path and the shortcut connection
    All convolutions inside the block should be followed by batch
        normalization along the channels axis and ReLU, respectively.
    All weights should use he normal initialization

    Returns: activated output of the projection block
    """
    F11, F3, F12 = filters

    # Main block
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                            padding="same", activation="linear",
                            kernel_initializer="he_normal")(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding="same",
                            activation="linear",
                            kernel_initializer="he_normal")(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding="same",
                            activation="linear",
                            kernel_initializer="he_normal")(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut block
    shortconv = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s,
                                padding="same", activation="linear",
                                kernel_initializer="he_normal")(A_prev)
    shortbn = K.layers.BatchNormalization(axis=-1)(shortconv)

    # Merge
    add = K.layers.Add()([bn3, shortbn])
    act3 = K.layers.Activation('relu')(add)

    return act3

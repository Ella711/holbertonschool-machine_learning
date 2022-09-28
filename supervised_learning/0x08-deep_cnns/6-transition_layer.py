#!/usr/bin/env python3
"""
 5. Dense Block
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely
        Connected Convolutional Networks

    X: output from the previous layer
    nb_filters: integer representing the number of filters in X
    compression: compression factor for transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization
        and a ReLU activation, respectively

    Returns: Output of the transition layer and the number of filters
        within the output, respectively
    """
    comp_filters = int(nb_filters * compression)
    bn1 = K.layers.BatchNormalization(axis=-1)(X)
    act1 = K.layers.Activation('relu')(bn1)
    conv = K.layers.Conv2D(filters=comp_filters, kernel_size=1,
                           padding="same", activation="linear",
                           kernel_initializer="he_normal")(act1)
    pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return pool, comp_filters

#!/usr/bin/env python3
"""
 5. Dense Block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely
        Connected Convolutional Networks:

    X: output from the previous layer
    nb_filters: integer representing the number of filters in X
    growth_rate: growth rate for the dense block
    layers: number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization
        and a ReLU activation, respectively

    Returns: Concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs, respectively
    """
    for layer in range(layers):
        bn1 = K.layers.BatchNormalization(axis=-1)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(filters=(4 * growth_rate), kernel_size=1,
                                padding="same", activation="linear",
                                kernel_initializer="he_normal")(act1)
        bn2 = K.layers.BatchNormalization(axis=-1)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                padding="same", activation="linear",
                                kernel_initializer="he_normal")(act2)
        X = K.layers.Concatenate(axis=-1)([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters

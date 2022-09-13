#!/usr/bin/env python3
"""
1. Input
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    nx: number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions used for each layer
        of the network
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Returns: keras model
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    layer = K.layers.Dense(layers[0],
                           activation=activations[0],
                           kernel_regularizer=reg)(inputs)

    for i in range(1, len(layers)):
        layer = K.layers.Dropout(rate=(1-keep_prob))(layer)
        layer = K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_regularizer=reg)(layer)

    model = K.Model(inputs=inputs, outputs=layer)
    return model

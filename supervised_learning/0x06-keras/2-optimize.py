#!/usr/bin/env python3
"""
2. Optimize
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical
        crossentropy loss and accuracy metrics

    network: model to optimize
    alpha: learning rate
    beta1: first Adam optimization parameter
    beta2: second Adam optimization parameter

    Returns: None
    """
    adam = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss="categorical_crossentropy",
                    optimizer=adam,
                    metrics=["accuracy"])
    return None

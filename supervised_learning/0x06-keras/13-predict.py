#!/usr/bin/env python3
"""
13. Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    network: model to test
    data: input data to test the model with
    verbose: boolean determines if output should be printed during process

    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)

#!/usr/bin/env python3
"""
4. Forward Propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    X: np.ndarray - shape (nx, m) containing input data
        nx: number of input features
        m: number of data points
    weights: dictionary of the weights and biases
    L: number of layers
    keep_prob: probability that a node will be kept

    Returns: a dictionary containing the outputs of each layer and
        the dropout mask used on each layer
    """
    cache = {}
    A = X
    cache["A0"] = A
    for layer in range(1, L + 1):
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]
        Z = np.matmul(W, A) + b
        if layer == L:
            A_temp = np.exp(Z)
            A = A_temp / np.sum(A_temp, axis=0, keepdims=True)
            cache["A" + str(layer)] = A
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = np.multiply(A, D)
            A = A / keep_prob
            cache["A" + str(layer)] = A
            cache["D" + str(layer)] = D
    return cache

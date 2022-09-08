#!/usr/bin/env python3
"""
5. Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization

    Y: one-hot np.ndarray - shape (classes, m) contains correct labels
        classes: number of classes
        m: number of data points
    weights: dictionary of weights and biases
    cache: dictionary of the outputs of each layer
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache["A{}".format(i)]
        A_prev = cache["A{}".format(i - 1)]
        if i == L:
            dz = (A - Y)
        else:
            dz = da_prev * (1 - (A ** 2))
            dz = (dz * cache["D{}".format(i)]) / keep_prob
        W = weights["W{}".format(i)]
        dw = np.matmul(dz, A_prev.T) / m
        db = (np.sum(dz, axis=1, keepdims=True) / m)
        da_prev = np.matmul(W.T, dz)
        weights["W{}".format(i)] -= (alpha * dw)
        weights["b{}".format(i)] -= (alpha * db)

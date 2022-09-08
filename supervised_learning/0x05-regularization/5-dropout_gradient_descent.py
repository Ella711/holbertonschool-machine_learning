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
    for i in reversed(range(1, L + 1)):
        A = cache["A" + str(i)]
        A_prev = cache["A" + str(i - 1)]
        if i == L:
            dz = A - Y
        else:
            inv = 1 - np.square(A)
            dz = da * inv
            dz = (dz * cache["D" + str(i)]) / keep_prob
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        W = weights["W" + str(i)]
        da = np.matmul(W.T, dz)

        weights["W" + str(i)] -= (alpha * dw)
        weights["b" + str(i)] -= (alpha * db)

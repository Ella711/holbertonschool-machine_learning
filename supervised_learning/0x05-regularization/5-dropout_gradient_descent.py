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
    dz = cache["A" + str(L)] - Y
    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        inv = 1 - np.square(A_prev)
        W = weights["W" + str(i)]
        da = np.matmul(W.T, dz)
        if i > 1:
            da = da * cache["D" + str(i - 1)]
            da = da / keep_prob
        dz = da * inv

        weights["W" + str(i)] -= (alpha * dw)
        weights["b" + str(i)] -= (alpha * db)

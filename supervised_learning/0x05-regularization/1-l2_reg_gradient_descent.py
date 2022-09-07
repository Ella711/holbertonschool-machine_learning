#!/usr/bin/env python3
"""
1. Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization

    Y: one-hot np.ndarray - shape (classes, m) contains correct labels
        classes: number of classes
        m: number of data points
    weights: dictionary of weights and biases
    cache: dictionary of the outputs of each layer
    alpha: learning rate
    lambtha: L2 regularization parameter
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
        dz = np.matmul(W.T, dz) * inv
        l2_reg = 1 - (alpha * lambtha) / m

        weights["W" + str(i)] = l2_reg * weights["W" + str(i)] - (alpha * dw)
        weights["b" + str(i)] -= (alpha * db)

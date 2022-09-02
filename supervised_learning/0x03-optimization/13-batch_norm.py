#!/usr/bin/env python3
"""
13. Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
        using batch normalization


    Z: ndarray - shape (m, n) that should be normalized
        m: number of data points
        n: number of features in Z
    gamma: ndarray - shape (1, n) containing the scales
    beta: ndarray - shape (1, n) containing the offsets
    epsilon: small number used to avoid division by zero

    Returns: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    numerator = (Z - mean) / (np.sqrt(variance + epsilon))
    return gamma * numerator + beta

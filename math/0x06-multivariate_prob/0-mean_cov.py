#!/usr/bin/env python3
"""
0. Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    Args:
        X: np.ndarray - shape (n, d) contains data set

    Returns: mean, cov
    """
    n = X.shape[0]
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = X.sum(axis=0) / n
    deviation = X - mean
    covariant = np.matmul(deviation.T, deviation)
    return mean[np.newaxis, ...], covariant / (n - 1)

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
    n, _ = X.shape
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov

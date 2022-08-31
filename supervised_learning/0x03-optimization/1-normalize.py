#!/usr/bin/env python3
"""
1. Normalize
"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    X: np.ndarray of shape (m, nx)
        m: number of data points
        nx: number of features
    m: np.ndarray of shape (nx,), contains the mean of all features of X
    s: np.ndarray of shape (nx,), contains the stddev of all features of X

    Returns: mean and std deviation of each feature
    """
    return (X - m) / s

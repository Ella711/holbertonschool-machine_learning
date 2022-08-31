#!/usr/bin/env python3
"""
0. Normalization Constants
"""


def normalization_constants(X):
    """
    Calculates the normalization (standardization)
    constants of a matrix

    X: np.ndarray of shape (m, nx)
    m: number of data points
    nx: number of features

    Returns: mean and std deviation of each feature
    """
    return X.mean(axis=0), X.std(axis=0)

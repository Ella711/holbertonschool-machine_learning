#!/usr/bin/env python3
"""
0. Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization)
    constants of a matrix

    X: np.ndarray of shape (m, nx)
    m: number of data points
    nx: number of features

    Returns: mean and std deviation of each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)

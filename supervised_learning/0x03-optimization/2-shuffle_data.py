#!/usr/bin/env python3
"""
2. Shuffle Data
"""
import numpy as np
from copy import deepcopy


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    X: np.ndarray of shape (m, nx)
        m: number of data points
        nx: number of features
    Y: np.ndarray of shape (m, ny)
        m: is the same number of data points as in X
        ny: is the number of features in Y

    Returns: the shuffled X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle], Y[shuffle]

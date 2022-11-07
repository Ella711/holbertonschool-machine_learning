#!/usr/bin/env python3
"""
1. Correlation
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    Args:
        C: np.ndarray - shape (d, d) contains a covariance matrix

    Returns: np.ndarray matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    # Isolate variances from matrix.
    var_values = np.diag(np.diag(C))
    # Get standard deviations
    std_devs = np.sqrt(var_values)
    # Get inverse / standardizing matrix
    inverse = np.linalg.inv(std_devs)

    corr_matrix = (inverse @ C) @ inverse

    return corr_matrix

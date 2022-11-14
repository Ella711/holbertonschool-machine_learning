#!/usr/bin/env python3
"""
2. Variance
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        C: np.ndarray - shape (n, d) contains the centroid
            means for each cluster

    Returns: var, or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray) or \
            len(X.shape) != 2 or len(C.shape) != 2 or \
            X.shape[1] != C.shape[1] or C.shape[1] <= 0 or X.size == 0 or \
            C.size == 0:
        return None

    distances = np.linalg.norm(X - C[:, np.newaxis], axis=2).T
    min_distances = np.min(distances, axis=1)
    var = np.sum(np.square(min_distances))
    return var

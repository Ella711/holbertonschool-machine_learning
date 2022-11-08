#!/usr/bin/env python3
"""
0. PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    Args:
        X: np.ndarray - shape (n, d)
            n: number of data points
            d: number of dimensions in each point
            all dimensions have a mean of 0
        var: fraction of the variance that the PCA transformation
            should maintain

    Returns: weights matrix, W, that maintains var fraction of
        X‘s original variance
    """
    # SVD (Singular Value Decomposition)
    u, s, vh = np.linalg.svd(X)
    V = vh.T
    s_percent = s / np.sum(s)
    variance_cs = np.cumsum(s_percent)
    num = np.argmax(np.where(variance_cs <= var, variance_cs, 0)) + 1
    W = V[:, :(num + 1)]
    return W

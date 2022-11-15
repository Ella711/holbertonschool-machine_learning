#!/usr/bin/env python3
"""
4. Initialize GMM
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model:
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        k: positive integer containing the number of clusters

    Returns: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None
    _, d = X.shape
    pi = np.ones(k) / k
    m = kmeans(X, k)[0]
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S

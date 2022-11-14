#!/usr/bin/env python3
"""
0. Initialize K-means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    Args:
        X: np.ndarray - shape (n, d) contains dataset that will be
            used for K-means clustering
        k: positive integer containing the number of clusters

    Returns: np.ndarray - shape (k, d) contains initialized centroids
        for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0 or \
            len(X.shape) != 2 or k > X.shape[0]:
        return None

    _, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, (k, d))

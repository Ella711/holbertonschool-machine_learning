#!/usr/bin/env python3
"""
1. K-means
"""
import numpy as np


def initialize(X, k, size):
    """
    Initializes cluster centroids for K-means
    """
    k, d = size
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0 or \
            len(X.shape) != 2 or k > X.shape[0]:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, (k, d))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset:
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number
            of iterations that should be performed

    Returns: C, clss, or None, None on failure
    """
    n, d = X.shape
    centroids = initialize(X, k, (k, d))

    for i in range(iterations):
        # Calculate distance between centroids and data points
        difference = (X - centroids[:, None, :])
        distance = np.linalg.norm(difference, axis=2).T
        # Separate into clusters
        clss = np.argmin(distance, axis=1)
        labeled = np.concatenate((X.copy(), np.reshape(clss, (n, 1))), axis=1)

        # Calculate means
        C = np.empty((k, d))
        for j in range(k):
            temp = labeled[labeled[:, -1] == j]
            temp = temp[:, :d]
            if temp.size == 0:
                re_init = initialize(X, k, size=(1, d))
                C[j] = re_init
            else:
                C[j] = np.mean(temp, axis=0)
        # Recalculate clss
        clss = np.argmin(np.linalg.norm((X - C[:, None, :]), axis=2).T, axis=1)

        # Check for change
        if np.array_equal(centroids, C):
            break

        # Assign new centroids
        centroids = C

    return C, clss

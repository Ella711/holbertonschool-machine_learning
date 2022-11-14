#!/usr/bin/env python3
"""
1. K-means
"""
import numpy as np


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
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or \
            k <= 0 or len(X.shape) != 2 or k > X.shape[0] or \
            not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = np.random.uniform(low, high, (k, d))

    for i in range(iterations):
        difference = (X - centroids[:, None, :])
        distance = np.linalg.norm(difference, axis=2).T
        clss = np.argmin(distance, axis=1)
        labeled = np.concatenate((X.copy(), np.reshape(clss, (n, 1))), axis=1)

        C = np.empty((k, d))
        for j in range(k):
            temp = labeled[labeled[:, -1] == j]
            temp = temp[:, :d]
            if temp.size == 0:
                re_init = np.random.uniform(low, high, (1, d))
                C[j] = re_init
            else:
                C[j] = np.mean(temp, axis=0)
        clss = np.argmin(np.linalg.norm((X - C[:, None, :]), axis=2).T, axis=1)

        if np.array_equal(centroids, C):
            break

        centroids = C

    return C, clss

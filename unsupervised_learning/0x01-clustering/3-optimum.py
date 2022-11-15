#!/usr/bin/env python3
"""
3. Optimize k
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        kmin: positive integer containing the minimum number
            of clusters to check for (inclusive)
        kmax: positive integer containing the maximum number
            of clusters to check for (inclusive)
        iterations: positive integer containing the maximum number
            of iterations for K-means

    Returns: results, d_vars, or None, None on failure
    """
    if kmax is not None and not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if not isinstance(X, np.ndarray) or not isinstance(iterations, int) or \
            len(X.shape) != 2 or iterations <= 0:
        return None, None
    if not kmax:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin >= kmax or kmin <= 0:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, C))

    d_vars = [d_vars[0] - x for x in d_vars]

    return results, d_vars

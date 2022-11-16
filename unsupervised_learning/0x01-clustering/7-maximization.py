#!/usr/bin/env python3
"""
7. Maximization
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        g: np.ndarray - shape (k, n) contains the posterior probabilities
            for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or \
            not isinstance(g, np.ndarray) or g.ndim != 2 or \
            X.shape[0] != g.shape[1] or \
            not np.isclose(np.sum(g, axis=0), np.ones(X.shape[0], )).all():
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    pi = np.empty((k))
    m = np.empty((k, d))
    S = np.empty((k, d, d))

    for i in range(k):
        gi_sum = np.sum(g[i], axis=0)
        pi[i] = gi_sum / n
        m[i] = np.sum(g[i, None, ...] @ X, axis=0) / gi_sum
        S[i] = ((g[i, None, ...] * (X - m[i]).T) @ (X - m[i])) / gi_sum

    return pi, m, S

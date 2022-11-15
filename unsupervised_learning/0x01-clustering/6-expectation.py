#!/usr/bin/env python3
"""
4. Initialize GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm
        for a GMM
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        pi: np.ndarray - shape (k,) contains the priors for
            each cluster
        m: np.ndarray - shape (k, d) contains the centroid
            means for each cluster
        S: np.ndarray - shape (k, d, d) contains the covariance
            matrices for each cluster

    Returns: g, l, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray)\
        or not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray)\
        or X.ndim != 2 or pi.ndim != 1 or m.ndim != 2 or S.ndim != 3\
        or pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]\
        or X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]\
        or S.shape[1] != S.shape[2] or np.any(np.linalg.det(S) == 0)\
            or not np.isclose(pi.sum(), 1):
        return None, None

    k = pi.shape[0]
    n, d = X.shape
    g = np.empty((k, n))

    for i in range(k):
        likelihood = pdf(X, m[i], S[i])
        prior = pi[i]  # (1,)
        intersection = prior * likelihood
        g[i] = intersection

    marginal = np.sum(g, axis=0, keepdims=True)
    g /= marginal

    log = np.sum(np.log(np.sum(marginal, axis=0)), axis=0)
    return g, log

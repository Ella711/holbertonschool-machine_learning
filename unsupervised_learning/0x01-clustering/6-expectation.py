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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    pdfs = np.ndarray((m.shape[0], X.shape[0]))

    for cluster in range(m.shape[0]):
        pdfs[cluster] = pdf(X, m[cluster], S[cluster])

    pdfs = pdfs * pi[:, np.newaxis]
    pdfsum = pdfs.sum(axis=0)
    expects = pdfs / pdfsum

    return expects, np.log(pdfsum).sum()

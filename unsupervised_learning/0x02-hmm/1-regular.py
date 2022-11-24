#!/usr/bin/env python3
"""
1. Regular Chains
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    Args:
        P: square 2D np.ndarray - shape (n, n) - transition matrix

    Returns: np.ndarray - shape (n, n) - transition matrix, or None
        on failure
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    n = P.shape[0]
    if not (P > 0).all():
        return None

    # v(P - I) = 0, πQ = 0
    identity = np.eye(n)
    Q = (P - identity)

    # Mπ = b
    M = np.vstack((Q.T[:-1], np.ones(n)))
    b = np.vstack((np.zeros((n - 1, 1)), [1]))

    v = np.linalg.solve(M, b).T
    return v

#!/usr/bin/env python3
"""
2. Absorbing Chains
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    Args:
        P: square 2D np.ndarray - shape (n, n) - transition matrix

    Returns: True if it is absorbing, or False on failure
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]
    diag = np.diag(P)

    # Check if 1 or all are absorbing
    if not (diag == 1).any():
        return False
    if (diag == 1).all():
        return True

    # Put transition matrix in standard form
    # Get I, R, & Q
    I_size = np.where(diag != 1)[0][0]
    I_from_P = np.eye(I_size)
    R = P[I_size:, :I_size]
    Q = P[I_size:, I_size:]

    # Fundamental matrix
    I_for_Q = np.eye(Q.shape[0])
    try:
        F = np.linalg.inv(I_for_Q - Q)
    except Exception:
        return False

    # Use F to find limiting matrix
    FR = F @ R
    P_bar = np.zeros((n, n))
    P_bar[:I_size, :I_size] = I_from_P
    P_bar[I_size:, :I_size] = FR

    if (FR == 0).all():
        return False

    return True

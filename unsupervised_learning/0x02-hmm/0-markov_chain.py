#!/usr/bin/env python3
"""
0. Markov Chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being
        in a particular state after a specified number
        of iterations
    Args:
        P: square 2D np.ndarray - shape (n, n) - transition matrix
        s: np.ndarray - shape (1, n) - probability of starting in each state
        t: number of iterations that the markov chain has been through

    Returns: np.ndarray - shape (1, n) - probability of being
        in a specific state after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or not isinstance(t, int):
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or t <= 0:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    return np.matmul(s, np.linalg.matrix_power(P, t))

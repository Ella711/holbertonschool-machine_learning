#!/usr/bin/env python3
"""
2. Initialize t-SNE
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    Args:
        X: np.ndarray - shape (n, d) contains the dataset to be
            transformed by t-SNE
            n: number of data points
            d: number of dimensions in each point
        tol: maximum tolerance allowed
        perplexity: perplexity that all Gaussian distributions should have

    Returns: P
    """
#!/usr/bin/env python3
"""
2. Initialize t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the
        P affinities in t-SNE
    Args:
        X: np.ndarray - shape (n, d) contains the dataset to be
            transformed by t-SNE
            n: number of data points
            d: number of dimensions in each point
        perplexity: perplexity that all Gaussian distributions should have

    Returns: (D, P, betas, H)
    """
    n = X.shape[0]
    X1 = X[:, :, None]
    D = ((X1 - X1.T) ** 2).sum(1)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H

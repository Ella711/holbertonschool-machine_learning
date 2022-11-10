#!/usr/bin/env python3
"""
2. Initialize t-SNE
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities
        relative to a data point
    Args:
        Di: np.ndarray - shape (n - 1,) contains the pariwise
            distances between a data point and all other
            points except itself
            n: number of data points
        beta: np.ndarray - shape (1,) contains the beta value
            for the Gaussian distribution

    Returns: (Hi, Pi)
    """
    numerator = np.exp(-Di * beta)
    denominator = np.sum(numerator)
    Pi = numerator / denominator
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi

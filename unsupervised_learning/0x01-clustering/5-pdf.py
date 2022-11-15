#!/usr/bin/env python3
"""
4. Initialize GMM
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of
        a Gaussian distribution
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        m: np.ndarray - shape (d,) contains the mean of
            the distribution
        S: np.ndarray - shape (d, d) contains the covariance
            of the distribution

    Returns: P, or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    # Multi-dimensional Gaussian Model
    # P(x) =∑[i, k] ϕ_i N(x∣μ_i, Σ_i)
    # N(x∣μ_i, Σ_i) = (1 / sqrt(2π**K |Σ|))exp(-(1/2)(x-μ_i).T(Σ_i**-1)(x-μ_i)
    # ∑[i, k] ϕ_i = 1

    d = X.shape[1]
    mu = m[None, :]
    cov = S
    pi = np.pi
    X_mu = X - mu
    determinant = np.linalg.det(cov)
    inverse = np.linalg.inv(cov)
    norm = 1 / (np.sqrt((((2 * pi) ** (d)) * (determinant))))
    res = np.exp(-0.5 * np.sum(((X_mu @ inverse) * X_mu), axis=1))
    pdf = (norm * res)
    P = np.maximum(pdf, 1e-300)
    return P

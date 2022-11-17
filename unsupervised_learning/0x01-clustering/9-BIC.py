#!/usr/bin/env python3
"""
9. BIC
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the
        Bayesian Information Criterion
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        kmin: positive integer containing the minimum number
            of clusters to check
        kmax: positive integer containing the maximum number
            of clusters to check
        iterations: positive integer containing the maximum
            number of iterations
        tol: non-negative float containing the tolerance
        verbose: boolean that determines if the EM algorithm
            should print information

    Returns: best_k, best_result, l, b, or None, None, None, None on failure
    """
    if kmax is None and isinstance(X, np.ndarray) and X.ndim == 2:
        kmax = X.shape[0]

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or kmin > X.shape[0] or
            not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin or
            kmax > X.shape[0] or not isinstance(iterations, int) or
            iterations <= 0 or not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape
    logll = np.empty((kmax - kmin + 1))
    bic = np.empty((kmax - kmin + 1))
    results = [()] * (kmax - kmin + 1)

    for k in range(kmin, kmax + 1):
        idx = k - kmin
        pi, m, S, g, log = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        logll[idx] = log
        results[idx] = (pi, m, S)

        p = (k * d) + (k * (d * (d + 1) / 2)) + (k - 1)
        BIC = (p * np.log(n)) - (2 * log)
        bic[idx] = BIC

    best_result = results[np.argmin(bic)]
    best_k = np.argmin(bic) + kmin

    return best_k, best_result, logll, bic

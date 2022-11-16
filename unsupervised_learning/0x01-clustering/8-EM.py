#!/usr/bin/env python3
"""
8. EM
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    Args:
        X:
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum
            number of iterations
        tol: non-negative float containing tolerance of the log
            likelihood, used to determine early stopping
        verbose: boolean that determines if you should print information

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure

    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or \
            not isinstance(k, int) or k <= 0 or k > X.shape[0] or \
            not isinstance(iterations, int) or iterations <= 0 or \
            not isinstance(tol, float) or tol < 0 or \
            not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, logll = expectation(X, pi, m, S)

    for i in range(iterations):
        g, logll = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        g, logll_new = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, logll.round(5)))

        if abs(logll_new - logll) <= tol:
            break

        logll = logll_new
    g, logll = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1, logll.round(5)))

    return pi, m, S, g, logll

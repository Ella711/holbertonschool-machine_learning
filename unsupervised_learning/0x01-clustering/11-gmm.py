#!/usr/bin/env python3
"""
11. GMM
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        k: number of clusters

    Returns: pi, m, S, clss, bic
    """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic

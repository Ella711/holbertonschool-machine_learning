#!/usr/bin/env python3
"""
10. Hello, sklearn!
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset:
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        k: number of clusters

    Returns: C, clss
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss

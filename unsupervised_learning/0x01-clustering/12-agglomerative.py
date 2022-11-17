#!/usr/bin/env python3
"""
12. Agglomerative
"""
from scipy.cluster.hierarchy import dendrogram, ward, fcluster
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset
    Args:
        X: np.ndarray - shape (n, d) contains the dataset
        dist: maximum cophenetic distance for all clusters

    Returns: clss
    """
    Z = ward(X)
    clss = fcluster(Z, t=dist, criterion='distance')
    dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss

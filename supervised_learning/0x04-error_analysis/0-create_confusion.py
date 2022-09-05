#!/usr/bin/env python3
"""
0. Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    labels: one-hot np.ndarray - shape (m, classes) - correct labels for each
    data point
        m: number of data points
        classes: number of classes
    logits: one-hot np.ndarray - shape (m, classes) - predicted labels
    Returns: a confusion np.ndarray - shape (classes, classes) with row indices
        - correct labels and column indices - predicted labels
    """
    return np.matmul(labels.T, logits)

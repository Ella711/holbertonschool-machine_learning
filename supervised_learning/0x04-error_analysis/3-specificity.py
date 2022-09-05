#!/usr/bin/env python3
"""
3. Specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    confusion: np.ndarray - shape (classes, classes) - row indices -
        correct labels and column indices - predicted labels
    classes: number of classes
    Returns: np.ndarray - shape (classes,) - contains sensitivity of each class
    """
    predicted_labels = np.diagonal(confusion)
    false_neg = np.sum(confusion, axis=1) - predicted_labels
    false_pos = np.sum(confusion, axis=0) - predicted_labels
    total_neg = np.sum(confusion) - predicted_labels - false_neg - false_pos
    return total_neg / (false_pos + total_neg)

#!/usr/bin/env python3
"""
2. Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    confusion: np.ndarray - shape (classes, classes) - row indices -
        correct labels and column indices - predicted labels
    classes: number of classes
    Returns: np.ndarray - shape (classes,) - contains sensitivity of each class
    """
    correct_predicted = np.sum(confusion, axis=0)
    predicted_labels = np.diagonal(confusion)
    precision = predicted_labels / correct_predicted
    return precision

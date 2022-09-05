#!/usr/bin/env python3
"""
1. Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    confusion: np.ndarray - shape (classes, classes) - row indices -
        correct labels and column indices - predicted labels
    classes: number of classes
    Returns: np.ndarray - shape (classes,) - contains sensitivity of each class
    """
    correct_labels = np.sum(confusion, axis=1)
    predicted_labels = np.diagonal(confusion)
    recall = predicted_labels / correct_labels
    return recall

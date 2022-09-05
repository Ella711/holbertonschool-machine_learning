#!/usr/bin/env python3
"""
4. F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

    confusion: np.ndarray - shape (classes, classes) - row indices -
        correct labels and column indices - predicted labels
    classes: number of classes
    Returns: np.ndarray - shape (classes,) - contains F1 score of each class
    """
    prec = precision(confusion)
    recall = sensitivity(confusion)
    return 2 * ((prec * recall) / (prec + recall))

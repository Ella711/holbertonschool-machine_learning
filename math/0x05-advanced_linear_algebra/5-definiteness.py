#!/usr/bin/env python3
"""
5. Definiteness
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Args:
        matrix: a np.ndarray - shape (n, n) - calculate definiteness

    Returns: the string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite if the
        matrix is positive definite, positive semi-definite, negative
        semi-definite, negative definite of indefinite, respectively
        If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    mat_len = matrix.shape[0]
    if len(matrix.shape) != 2 or mat_len != matrix.shape[1]:
        return None
    transpose = matrix.T
    if not np.array_equal(transpose, matrix):
        return None
    eigval, _ = np.linalg.eig(matrix)
    if all(eigval > 0):
        return "Positive definite"
    if all(eigval >= 0):
        return "Positive semi-definite"
    if all(eigval < 0):
        return "Negative definite"
    if all(eigval <= 0):
        return "Negative semi-definite"
    return 'Indefinite'

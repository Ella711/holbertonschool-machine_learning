#!/usr/bin/env python3
"""
Module containing the class MultiNormal
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Class constructor
        Args:
            data: np.ndarray - shape (d, n) contains the data set
        """
        n, _ = data.shape
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = self.mean_cov(data.T)

    @staticmethod
    def mean_cov(X):
        """
        Calculates the mean and covariance of a data set
        """
        n, d = X.shape
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(X, axis=0, keepdims=True)
        deviation = X - mean
        covariant = np.matmul(deviation.T, deviation)
        return mean.reshape((d, 1)), covariant / (n - 1)

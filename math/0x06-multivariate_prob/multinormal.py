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
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

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

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        Args:
            x: np.ndarray - shape (d, 1) containing the data points

        Returns: value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        m = self.mean
        cov = self.cov
        denominator = np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))
        inverse = np.linalg.inv(cov)
        exp = (-.5 * np.matmul(np.matmul((x - m).T, inverse), (x - m)))
        pdf = (1 / denominator) * np.exp(exp[0][0])
        return pdf

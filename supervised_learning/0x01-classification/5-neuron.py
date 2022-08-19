#!/usr/bin/env python3
"""
Module for class Neuron
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """ Class constructor """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter function for W """
        return self.__W

    @property
    def b(self):
        """ Getter function for b """
        return self.__b

    @property
    def A(self):
        """ Getter function for A """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = -1 / Y.shape[1]
        cost = m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        costt = self.cost(Y, A)
        return pred, costt

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = 1 / X.shape[1]
        dz = A - Y
        dw = m * np.matmul(dz, X.T)
        db = m * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

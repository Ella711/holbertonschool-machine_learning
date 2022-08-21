#!/usr/bin/env python3
"""
Module for class DeepNeuralNetwork
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing
    binary classification
    """

    def __init__(self, nx, layers):
        """
        Class Constructor

        nx is the number of input features
        layers is a list representing the number of
            nodes in each layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.__L):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                layer_prev = nx
            else:
                layer_prev = layers[layer - 1]
            self.__weights['W' + str(layer + 1)] = np.random.randn(
                layers[layer], layer_prev) * np.sqrt(2 / layer_prev)
            self.__weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        """ Getter for L """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            Z = np.matmul(
                self.__weights["W" + str(layer + 1)],
                self.__cache["A" + str(layer)]) + \
                self.__weights["b" + str(layer + 1)]
            self.__cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = -1 / Y.shape[1]
        cost = m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A = self.forward_prop(X)[0]
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

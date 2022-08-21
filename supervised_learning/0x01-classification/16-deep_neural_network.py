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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            if l == 0:
                layer_prev = nx
            else:
                layer_prev = layers[l - 1]
            self.weights['W' + str(l + 1)] = np.random.randn(
                layers[l], layer_prev) * np.sqrt(2 / layer_prev)
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))

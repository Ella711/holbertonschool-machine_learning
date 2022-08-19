#!/usr/bin/env python3
"""
Module for class NeuralNetwork
"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Class Constructor

        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.randn(nodes, nx)
        self.b1, self.b2 = np.zeros((nodes, 1)), 0
        self.A1, self.A2 = 0, 0
        self.W2 = np.random.randn(1, nodes)

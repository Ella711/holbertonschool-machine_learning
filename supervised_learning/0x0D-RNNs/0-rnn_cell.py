#!/usr/bin/env python3
"""
0. RNN Cell
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class Constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(y):
        """Softmax activation function.
        Args:
            x: np.ndarray - 2D tensor to apply the soft max activation on
        Returns:
            softmax activated version of y
        """
        return np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: np.ndarray - (m, h) - contains previous hidden state
            x_t: np.ndarray - (m, i) - contains data input for the cell

        Returns: h_next, y
        """
        first = np.concatenate((h_prev.T, x_t.T), axis=0)
        second = np.matmul(first.T, self.Wh) + self.bh
        h_next = np.tanh(second)
        output = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(output)
        return h_next, y

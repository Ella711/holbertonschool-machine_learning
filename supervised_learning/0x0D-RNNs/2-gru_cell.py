#!/usr/bin/env python3
"""
2. GRU Cell
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(y):
        """
        Softmax activation function
        """
        return np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))

    @staticmethod
    def sigmoid(y):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-y))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: np.ndarray - (m, h) - contains previous hidden state
            x_t: np.ndarray - (m, i) - contains data input for the cell

        Returns: h_next, y
        """
        cell_input = np.concatenate((h_prev.T, x_t.T), axis=0)

        z_gate = self.sigmoid((cell_input.T @ self.Wz) + self.bz)
        r_gate = self.sigmoid((cell_input.T @ self.Wr) + self.br)

        o_x = np.concatenate(((r_gate * h_prev).T, x_t.T), axis=0)

        t = np.tanh((o_x.T @ self.Wh) + self.bh)

        h_next = (1 - z_gate) * h_prev + z_gate * t

        y = self.softmax((h_next @ self.Wy) + self.by)
        return h_next, y

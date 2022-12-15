#!/usr/bin/env python3
"""
6. Bidirectional Cell Backward
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the
            forward direction for one time step
        Args:
            h_prev: np.ndarray - (m, h) - contains previous hidden state
            x_t: np.ndarray - (m, i) - contains data input for the cell

        Returns: h_next
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(cell_input @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward
            direction for one time step
        Args:
            h_next: np.ndarray - (m, h) - contains previous hidden state
            x_t: np.ndarray - (m, i) - contains data input for the cell

        Returns: h_prev
        """
        cell_input = np.concatenate((h_next.T, x_t.T), axis=0)

        h_prev = np.tanh(cell_input.T @ self.Whb + self.bhb)
        return h_prev

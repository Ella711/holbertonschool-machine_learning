#!/usr/bin/env python3
"""
3. LSTM Cell
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        Args:
            h_prev: np.ndarray - (m, h) - contains previous hidden state
            c_prev: np.ndarray - (m, h) - contains previous cell state
            x_t: np.ndarray - (m, i) - contains data input for the cell

        Returns: h_next, c_next, y
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = self.sigmoid((cell_input @ self.Wf) + self.bf)
        update_gate = self.sigmoid((cell_input @ self.Wu) + self.bu)
        output_gate = self.sigmoid((cell_input @ self.Wo) + self.bo)

        candidate = np.tanh((cell_input @ self.Wc) + self.bc)

        c_next = forget_gate * c_prev + update_gate * candidate
        h_next = output_gate * np.tanh(c_next)
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y

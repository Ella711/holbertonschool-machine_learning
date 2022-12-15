#!/usr/bin/env python3
"""
 1. RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    Args:
        rnn_cell: instance of RNNCell used for the forward propagation
        X: np.ndarray - (t, m, i) - data to be used
        h_0: np.ndarray - (m, h) - initial hidden state

    Returns: H, Y
    """
    H, Y, h_next = [h_0], [], h_0

    for x in X:
        h_next, y = rnn_cell.forward(h_next, x)
        H.append(h_next), Y.append(y)

    H, Y = np.stack(H), np.stack(Y)
    return H, Y

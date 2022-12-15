#!/usr/bin/env python3
"""
4. Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l
            used for forward propagation
        X: np.ndarray - (t, m, i) - data to be used
        h_0: np.ndarray - (l, m, h) - initial hidden state

    Returns: H, Y
    """
    layers = len(rnn_cells)
    H, Y = [h_0], []
    temp_h = h_0.copy()

    for x in X:
        temp_h[0], _ = rnn_cells[0].forward(temp_h[0], x)
        for i in range(layers - 1):
            temp_h[1 + i], out = rnn_cells[1 + i].forward(
                temp_h[1 + i], temp_h[i])
            if i == layers - 2:
                Y.append(out)

        H.append(temp_h.copy())

    H, Y = np.stack(H), np.stack(Y)
    return H, Y

#!/usr/bin/env python3
"""
8. Bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    Args:
        bi_cell: instance of BidirectinalCell that will be
            used for the forward propagation
        X: np.ndarray - (t, m, i) - contains data to be used
        h_0: np.ndarray - (m, h) - initial hidden state - forward direction
        h_t: np.ndarray - (m, h) - initial hidden state - backward direction

    Returns: H, Y
    """
    Hforward, Hback = [], []
    h_next, h_prev = h_t, h_0

    for x, rev_x in zip(X, X[::-1]):
        h_prev = bi_cell.forward(h_prev, x)
        h_next = bi_cell.backward(h_next, rev_x)

        Hforward.append(h_prev)
        Hback = [h_next] + Hback

    H = np.concatenate((np.stack(Hforward), np.stack(Hback)), axis=-1)

    return H, bi_cell.output(H)

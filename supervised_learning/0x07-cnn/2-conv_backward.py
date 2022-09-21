#!/usr/bin/env python3
"""
2. Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    dZ: np.ndarray - shape (m, h_new, w_new, c_new) - partial derivatives with
    respect to the unactivated output of the convolutional layer
        m: number of examples
        h_new: height of the output
        w_new: width of the output
        c_new: number of channels in the output
    A_prev: np.ndarray - shape (m, h_prev, w_prev, c_prev) -
    output of previous layer
        m: number of examples
        h_prev: height of previous layer
        w_prev: width of previous layer
        c_prev: number of channels in previous layer
    W: np.ndarray - shape (kh, kw, c_prev, c_new): kernels for the convolution
        kh: filter height
        kw: filter width
        c_prev: number of channels in previous layer
        c_new: number of channels in output
    b: np.ndarray - shape (1, 1, 1, c_new): biases applied to the convolution
    activation: activation function applied to the convolution
    padding: string (same or valid) indicates type of padding used
    stride - tuple (sh, sw): strides for the convolution
        sh: stride for the height
        sw: stride for the width

    Returns: partial derivatives with respect to the previous layer (dA_prev),
    the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = (((h_prev - 1) * sh + kh - h_prev) // 2) + 1
        pad_w = (((w_prev - 1) * sw + kw - w_prev) // 2) + 1
    else:
        pad_h = 0
        pad_w = 0

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                   'constant')
    dA_pad = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    'constant')

    for ex in range(m):
        for h in range(h_new):
            x0 = h * sh
            for w in range(w_new):
                y0 = w * sw
                for c in range(c_new):
                    x1 = x0 + kh
                    y1 = y0 + kw
                    dZ_slice = dZ[ex, h, w, c]
                    dA_pad[ex, x0:x1, y0:y1, :] += W[:, :, :, c]\
                                                   * dZ_slice
                    dW[:, :, :, c] += A_pad[ex, x0:x1, y0:y1, :] * dZ_slice
    if padding == "same":
        dA_prev = dA_pad[:, pad_h: -pad_h, pad_w: -pad_w, :]
    else:
        dA_prev = dA_pad
    return dA_prev, dW, db

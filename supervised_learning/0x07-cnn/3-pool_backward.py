#!/usr/bin/env python3
"""
3. Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    dA - np.ndarray - shape (m, h_new, w_new, c_new) - partial derivatives with
    respect to the output of the pooling layer
        - m: number of examples
        - h_new: height of the output
        - w_new: width of the output
        - c: number of channels
    A_prev: np.ndarray - shape (m, h_prev, w_prev, c) -
    multiple grayscale images
        - h_prev: height in pixels of previous layer
        - w_prev: width in pixels of previous layer
    kernel_shape: tuple - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width
    stride: tuple of (sh, sw)
        - sh: stride for the height of image
        - sw: stride for the width of image
    mode: type of pooling
        - max: max pooling
        - avg: average pooling

    Returns: partial derivatives with respect to the previous layer (dA_prev)
    """
    # Grab dimensions of images and the kernel
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev
    dA_prev = np.zeros(A_prev.shape)

    # Back propagate
    for ex in range(m):
        for h in range(h_new):
            x0 = h * sh
            for w in range(w_new):
                y0 = w * sw
                for c in range(c_new):
                    x1 = x0 + kh
                    y1 = y0 + kw
                    if mode == "avg":
                        avg_dA = dA[ex, h, w, c] / kh / kw
                        dA_prev[ex, x0:x1, y0:y1, c] += (
                                np.ones((kh, kw)) * avg_dA)
                    elif mode == "max":
                        A_prev_slice = A_prev[ex, x0:x1, y0:y1, c]
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        dA_prev[ex, x0:x1, y0:y1, c] += mask * dA[ex, h, w, c]
    return dA_prev

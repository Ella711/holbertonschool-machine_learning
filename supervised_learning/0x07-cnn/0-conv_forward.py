#!/usr/bin/env python3
"""
0. Convolutional Forward Prop
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

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

    Returns: output of the convolutional layer
    """
    # Grab dimensions of images and the kernel
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc_prev, c_new = W.shape
    sh, sw = stride

    # Pad dimensions
    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        pad_h = 0
        pad_w = 0

    pad_image = np.pad(A_prev,
                       ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                       "constant")

    # Set up what our output convoluted image will be shaped like
    output_h = (h_prev + 2 * pad_h - kh) // sh + 1
    output_w = (w_prev + 2 * pad_w - kw) // sw + 1
    conv_image = np.zeros((m, output_h, output_w, c_new))

    # Convolve
    for z in range(c_new):
        for x in range(output_h):
            for y in range(output_w):
                x0 = x * sh
                x1 = x0 + kh
                y0 = y * sw
                y1 = y0 + kw
                conv_image[:, x, y, z] = np.sum(pad_image[:, x0:x1, y0:y1, :]
                                                * W[..., z],
                                                axis=(1, 2, 3))
    Z = conv_image + b
    return activation(Z)

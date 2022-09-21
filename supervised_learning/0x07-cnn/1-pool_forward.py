#!/usr/bin/env python3
"""
1. Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs pooling on images

    A_prev: np.ndarray - shape (m, h_prev, w_prev, c_prev) -
    multiple grayscale images
        - m: number of images
        - h_prev: height in pixels of previous layer
        - w_prev: width in pixels of previous layer
        - c_prev: number of channels of previous layer
    kernel_shape: tuple - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width
    stride: tuple of (sh, sw)
        - sh: stride for the height of image
        - sw: stride for the width of image
    mode: type of pooling
        - max: max pooling
        - avg: average pooling

    Returns: a np.ndarray - pooled images
    """
    # Grab dimensions of images and the kernel
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Set up what our output convoluted image will be shaped like
    output_h = (h_prev - kw) // sh + 1
    output_w = (w_prev - kw) // sw + 1
    pooled_image = np.zeros((m, output_h, output_w, c_prev))

    if mode == "max":
        pool_func = np.max
    else:
        pool_func = np.average

    # Convolve
    for x in range(output_h):
        for y in range(output_w):
            # start = height * stride
            x0 = x * sh
            # end = start + kernel/filter size
            x1 = x0 + kh
            y0 = y * sw
            y1 = y0 + kw
            pooled_image[:, x, y, :] = pool_func(
                A_prev[:, x0:x1, y0:y1, :], axis=(1, 2)
            )
    return pooled_image

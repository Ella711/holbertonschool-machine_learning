#!/usr/bin/env python3
"""
1. Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images: np.ndarray - shape (m, h, w) - multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    kernel: np.ndarray - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width
    if necessary, the image should be padded with 0’s

    Returns: a np.ndarray - convolved images
    """
    # Grab dimensions of images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Pad dimensions
    pad_h = kh // 2
    pad_w = kw // 2
    pad_image = np.pad(images,
                       ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                       "constant")

    # Set up what our output convoluted image will be shaped like
    conv_image = np.zeros(images.shape)

    # Convolve
    for x in range(h):
        for y in range(w):
            x0 = x + kh
            y0 = y + kw
            conv_image[:, x, y] = np.sum(pad_image[:, x:x0, y:y0] * kernel,
                                         axis=(1, 2))
    return conv_image

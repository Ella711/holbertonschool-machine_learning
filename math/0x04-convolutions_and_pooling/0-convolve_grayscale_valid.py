#!/usr/bin/env python3
"""
0. Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    images: np.ndarray - shape (m, h, w) - multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    kernel: np.ndarray - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width

    Returns: a np.ndarray - convolved images
    """
    # Grab dimensions of images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Set up what our output convoluted image will be shaped like
    output_h, output_w = h - kh + 1, w - kw + 1
    conv_image = np.zeros((m, output_h, output_w))

    # Convolve
    for x in range(output_h):
        for y in range(output_w):
            x0 = x + kh
            y0 = y + kw
            conv_image[:, x, y] = np.sum(images[:, x:x0, y:y0] * kernel,
                                         axis=(1, 2))
    return conv_image

#!/usr/bin/env python3
"""
2. Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    images: np.ndarray - shape (m, h, w) - multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    kernel: np.ndarray - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width
    padding: tuple of (ph, pw)
        - ph: padding for the height
        - pw: padding for the width
        - the image should be padded with 0’s

    Returns: a np.ndarray - convolved images
    """
    # Grab dimensions of images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Pad dimensions
    pad_h, pad_w = padding
    pad_image = np.pad(images,
                       ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                       "constant")

    # Set up what our output convoluted image will be shaped like
    output_h, output_w = h + 2 * pad_h - kh + 1, w + 2 * pad_w - kw + 1
    conv_image = np.zeros((m, output_h, output_w))

    # Convolve
    for x in range(output_h):
        for y in range(output_w):
            x0 = x + kh
            y0 = y + kw
            conv_image[:, x, y] = np.sum(pad_image[:, x:x0, y:y0] * kernel,
                                         axis=(1, 2))
    return conv_image

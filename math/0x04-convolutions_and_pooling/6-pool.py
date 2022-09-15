#!/usr/bin/env python3
"""
6. Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images: np.ndarray - shape (m, h, w, c) - multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
        - c: number of channels
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
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Set up what our output convoluted image will be shaped like
    output_h = (h - kw) // sh + 1
    output_w = (w - kw) // sw + 1
    pooled_image = np.zeros((m, output_h, output_w, c))

    if mode == "max":
        pool_func = np.amax
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
                images[:, x0:x1, y0:y1, :], axis=(1, 2)
            )
    return pooled_image

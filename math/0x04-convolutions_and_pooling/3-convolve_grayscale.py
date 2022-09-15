#!/usr/bin/env python3
"""
3. Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    images: np.ndarray - shape (m, h, w) - multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    kernel: np.ndarray - shape (kh, kw) - kernel for the convolution
        - kh: height
        - kw: width
    padding: either tuple of (ph, pw) or:
        - if "same", performs a same convolution
        - if "valid", performs a valid convolution
        - ph: padding for the height
        - pw: padding for the width
        - the image should be padded with 0’s
    stride: tuple of (sh, sw)
        - sh: stride for the height of image
        - sw: stride for the width of image


    Returns: a np.ndarray - convolved images
    """
    # Grab dimensions of images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Pad dimensions
    if padding == "same":
        pad_h = ((h - 1) * sh + kh - h) // 2 + 1
        pad_w = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    pad_image = np.pad(images,
                       ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                       "constant")

    # Set up what our output convoluted image will be shaped like
    output_h = (h + 2 * pad_h - kh) // sh + 1
    output_w = (w + 2 * pad_w - kw) // sw + 1
    conv_image = np.zeros((m, output_h, output_w))

    # Convolve
    for x in range(output_h):
        for y in range(output_w):
            # start = height * stride
            x0 = x * sh
            # end = start + kernel/filter size
            x1 = x0 + kh
            y0 = y * sw
            y1 = y0 + kw
            conv_image[:, x, y] = np.sum(pad_image[:, x0:x1, y0:y1] * kernel,
                                         axis=(1, 2))
    return conv_image

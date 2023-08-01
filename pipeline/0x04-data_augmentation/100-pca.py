#!/usr/bin/env python3
"""
7. PCA Color Augmentation
"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper

    Args:
        image: 3D tf.Tensor containing the image to change
        alphas: tuple with amount that each channel should change

    Returns: augmented image
    """
    image = image.numpy()
    img = image.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(U, rand * S)
    delta = (delta * alphas * 255.0).astype(
        np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image + delta, 0, 255).astype(np.uint8)
    return img_out

#!/usr/bin/env python3
"""
2. Rotate
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise

    Args:
        image: 3D tf.Tensor containing the image to rotate

    Returns: rotated image
    """
    return tf.image.rot90(image)

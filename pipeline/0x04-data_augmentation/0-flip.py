#!/usr/bin/env python3
"""
0. Flip
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally

    Args:
        image: 3D tf.Tensor containing the image to flip

    Returns: flipped image
    """
    return tf.image.flip_left_right(image)

#!/usr/bin/env python3
"""
4. Brightness
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Args:
        image: 3D tf.Tensor containing the image to change
        max_delta: max amount image should be brightened (or darkened)

    Returns: altered image
    """
    return tf.image.random_brightness(image, max_delta)

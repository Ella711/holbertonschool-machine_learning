#!/usr/bin/env python3
"""
5. Hue
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Args:
        image: 3D tf.Tensor containing the image to change
        delta: amount the hue should change

    Return: altered image
    """
    return tf.image.adjust_hue(image, delta)

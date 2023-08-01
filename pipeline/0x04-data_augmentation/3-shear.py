#!/usr/bin/env python3
"""
3. Shear
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image

    Args:
        image: 3D tf.Tensor containing the image to shear
        intensity: intensity with which the image
                    should be sheared

    Returns: sheared image
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity)

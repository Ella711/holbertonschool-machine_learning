#!/usr/bin/env python3
"""
Create layer
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ Creates a layer """
    heetal = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    L = tf.keras.layers.Dense(n, activation=activation, name="layer", kernel_initializer=heetal)(prev)
    return L

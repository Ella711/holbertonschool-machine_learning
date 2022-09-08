#!/usr/bin/env python3
"""
6. Create a Layer with Dropout
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function that should be used
    keep_prob: probability that a node will be kept
    Returns: the output of the new layer
    """
    drop = tf.layers.Dropout(rate=keep_prob)
    weight = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weight,
                            kernel_regularizer=drop)(prev)
    return layer

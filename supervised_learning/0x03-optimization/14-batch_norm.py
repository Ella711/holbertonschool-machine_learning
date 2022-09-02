#!/usr/bin/env python3
"""
13. Batch Normalization
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    prev: activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function

    Returns: a tensor of the activated output for the layer
    """
    # Variables
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    epsilon = 1e-8

    # Initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer
    densor = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)

    # Batch norm
    mean, variance = tf.nn.moments(densor, axes=[0])
    batch_norm = tf.nn.batch_normalization(densor, mean, variance,
                                           beta, gamma, epsilon)
    return activation(batch_norm)

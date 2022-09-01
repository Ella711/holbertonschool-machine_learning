#!/usr/bin/env python3
"""
10. Adam Upgraded
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network
    in tensorflow using Adam optimization algorithm

    loss: loss of the network
    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)

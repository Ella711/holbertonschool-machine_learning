#!/usr/bin/env python3
"""
6. Momentum Upgraded
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural
        network in tensorflow using the gradient descent
        with momentum optimization algorithm

    loss: loss of the network
    alpha: learning rate
    beta1: the momentum weight
    Returns: the momentum optimization operation
    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1)
    return momentum.minimize(loss)

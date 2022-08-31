#!/usr/bin/env python3
"""
8. RMSProp Upgraded
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm

    loss: loss of the network
    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    rms_prop = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms_prop.minimize(loss)

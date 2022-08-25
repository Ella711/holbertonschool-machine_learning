#!/usr/bin/env python3
"""
Training Operation
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ Creates the training operation for the network """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)

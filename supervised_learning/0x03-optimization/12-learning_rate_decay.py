#!/usr/bin/env python3
"""
12. Learning Rate Decay Upgraded
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in
        tensorflow using inverse time decay

    alpha: original learning rate
    decay_rate: weight used to determine the rate at which alpha will decay
    global_step: number of passes of gradient descent that have elapsed
    decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further

    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)

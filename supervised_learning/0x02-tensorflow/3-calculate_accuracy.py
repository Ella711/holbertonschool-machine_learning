#!/usr/bin/env python3
"""
Accuracy function
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return mean

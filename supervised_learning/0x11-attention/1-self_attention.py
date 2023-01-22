#!/usr/bin/env python3
"""
1. Self Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class that calculate the attention for machine translation
    """

    def __init__(self, units):
        """
        Class constructor
        Args:
            units: int - number of hidden units in the alignment model
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Args:
            s_prev: tensor of shape (batch, units) containing
                the previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder

        Returns: context, weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))

        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum((weights * hidden_states), axis=1)

        return context, weights

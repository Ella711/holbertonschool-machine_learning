#!/usr/bin/env python3
"""
5. Multi Head Attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class that performs multi head attention
    """
    def __init__(self, dm, h):
        """
        Class Constructor
        Args:
            dm: int - dimensionality of the model
            h: int - number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def split_heads(self, x, batch):
        """
        Splits inputs
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Args:
            Q: tensor - (..., seq_len_q, dk) containing the query matrix
            K: tensor - (..., seq_len_v, dk) containing the key matrix
            V: tensor - (..., seq_len_v, dv) containing the value matrix
            mask: None

        Returns: output, weights
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)
        return output, weights

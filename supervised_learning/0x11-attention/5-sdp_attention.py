#!/usr/bin/env python3
"""
4. Scaled Dot Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention
    Args:
        Q: tensor - (..., seq_len_q, dk) containing the query matrix
        K: tensor - (..., seq_len_v, dk) containing the key matrix
        V: tensor - (..., seq_len_v, dv) containing the value matrix
        mask: tensor - (..., seq_len_q, seq_len_v) containing the optional
            mask, or defaulted to None

    Returns: output, weights
    """
    q_k = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attentions = q_k / tf.sqrt(dk)
    if mask is not None:
        scaled_attentions += mask
    weights = tf.nn.softmax(scaled_attentions, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights

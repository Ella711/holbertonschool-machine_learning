#!/usr/bin/env python3
"""
6. Transformer Encoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Creates an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Args:
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            drop_rate: dropout rate
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Args:
            x: tensor - (batch, input_seq_len, dm) - input to the encoder block
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention

        Returns: (batch, input_seq_len, dm)
        """
        # Multihead Attention Block
        attention_output, _ = self.mha(x, x, x, mask)
        dropout_output_1 = self.dropout1(attention_output, training=training)
        norm_output_1 = self.layernorm1(dropout_output_1 + x)

        # Feed Forward Block
        dense_output_1 = self.dense_hidden(norm_output_1)
        dense_output_2 = self.dense_output(dense_output_1)
        dropout_output_2 = self.dropout2(dense_output_2, training=training)
        norm_output_2 = self.layernorm2(dropout_output_2 + norm_output_1)

        return norm_output_2

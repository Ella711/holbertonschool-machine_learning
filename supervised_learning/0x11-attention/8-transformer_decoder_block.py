#!/usr/bin/env python3
"""
7. Transformer Decoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Creates a decoder block for a transformer
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
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args:
            x: tensor - (batch, target_seq_len, dm) -
                input to the decoder block
            encoder_output: tensor - (batch, input_seq_len, dm) -
                output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to mha1 layer
            padding_mask: mask to be applied to mha2 layer

        Returns: (batch, target_seq_len, dm)
        """
        # Output MHA block
        attention_output_1, _ = self.mha1(x, x, x, look_ahead_mask)
        dropout_output_1 = self.dropout1(attention_output_1, training=training)
        norm_output_1 = self.layernorm1(dropout_output_1 + x)

        # Input MHA block
        attention_output_2, _ = self.mha2(norm_output_1, encoder_output,
                                          encoder_output, padding_mask)
        dropout_output_2 = self.dropout2(attention_output_2, training=training)
        norm_output_2 = self.layernorm2(dropout_output_2 + norm_output_1)

        # Feed Forward Block
        dense_output_1 = self.dense_hidden(norm_output_2)
        dense_output_2 = self.dense_output(dense_output_1)
        dropout_output_3 = self.dropout3(dense_output_2, training=training)
        norm_output_3 = self.layernorm3(dropout_output_3 + norm_output_2)

        return norm_output_3

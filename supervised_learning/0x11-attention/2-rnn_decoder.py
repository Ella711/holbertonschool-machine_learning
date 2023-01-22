#!/usr/bin/env python3
"""
2. RNN Decoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class
    """
    def __init__(self, vocab, embedding, units, batch) -> None:
        """
        Class Constructor
        Args:
            vocab: int - size of the input vocabulary
            embedding: int - dimensionality of the embedding vector
            units: int - number of hidden units in the RNN cell
            batch: int - batch size
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Args:
            x: tensor of shape (batch, input_seq_len) containing
                the input to the encoder layer as word indices within
                the vocabulary
            initial: tensor of shape (batch, units) containing the
                initial hidden state

        Returns: y, s
        """
        self_attention = SelfAttention(self.units)
        context, _ = self_attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        inputs = tf.concat([context, x], axis=-1)
        output, s = self.gru(inputs=inputs)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, s

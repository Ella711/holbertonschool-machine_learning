#!/usr/bin/env python3
"""
0. RNN Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class Constructor
        Args:
            vocab: int - size of the input vocabulary
            embedding: int - dimensionality of the embedding vector
            units: int - number of hidden units in the RNN cell
            batch: int - batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes hidden states RNN cell to a tensor of zeros

        Returns: tensor of shape (batch, units)containing
            the initialized hidden states
        """
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        Args:
            x: tensor of shape (batch, input_seq_len) containing
                the input to the encoder layer as word indices within
                the vocabulary
            initial: tensor of shape (batch, units) containing the
                initial hidden state

        Returns: outputs, hidden
        """
        embedded = self.embedding(x)
        return self.gru(inputs=embedded, initial_state=initial)

#!/usr/bin/env python3
"""
9. Transformer Decoder
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Creates the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        Args:
            N: number of blocks in the encoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layer
            target_vocab: size of the target vocabulary
            max_seq_len: maximum sequence length possible
            drop_rate: dropout rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

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
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x

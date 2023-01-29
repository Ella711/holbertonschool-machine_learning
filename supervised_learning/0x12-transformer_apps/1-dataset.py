#!/usr/bin/env python3
"""
1. Encode Tokens
"""
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset formatted as a tuple (pt, en)

        Returns: tokenizer_pt, tokenizer_en
        """
        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (p.numpy() for p, e in data), target_vocab_size=2 ** 15)
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (e.numpy() for p, e in data), target_vocab_size=2 ** 15)
        return pt, en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        Args:
            pt: tf.Tensor containing Portuguese sentence
            en: tf.Tensor containing corresponding English sentence

        Returns: pt_tokens, en_tokens
        """
        pt_vs = self.tokenizer_pt.vocab_size
        en_vs = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vs] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_vs + 1]
        en_tokens = [en_vs] + self.tokenizer_en.encode(
            en.numpy()) + [en_vs + 1]
        return pt_tokens, en_tokens

#!/usr/bin/env python3
"""
0. Unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence

    Returns: unigram BLEU score
    """
    total, unigrams, bleu = 0, len(sentence), 1
    sen = sentence.copy()

    min_ref = min([len(ref) for ref in references])

    if unigrams <= min_ref:
        bleu = np.exp(1 - min_ref / unigrams)

    while len(sen) > 0:
        word = sen[0]
        word_count = sen.count(word)
        for i in range(word_count):
            sen.pop(sen.index(word))

        max_ref = max([ref.count(word) for ref in references])

        total += word_count if word_count <= max_ref else max_ref

    return bleu * (total / unigrams)

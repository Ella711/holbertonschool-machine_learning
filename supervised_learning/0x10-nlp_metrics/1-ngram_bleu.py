#!/usr/bin/env python3
"""
0. Unigram BLEU score
"""
import numpy as np


def gram(sen, start, n):
    """Calculates an n-gram token of size n"""

    return [sen[start+i] for i in range(n)]


def n_grams(sen, n):
    """Calculates an n-gram with tokens of size n."""
    return [gram(sen, i, n) for i in range(len(sen)-n+1)]


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns: n-gram BLEU score
    """
    ngrams = n_grams(sentence, n)
    total, bleu, num_tokens = 0, 1, len(ngrams)

    min_ref = min([len(ref) for ref in references])

    if len(sentence) <= min_ref:
        bleu = np.exp(1 - min_ref / len(sentence))

    while len(ngrams) > 0:
        token = ngrams[0]
        count_tokens = ngrams.count(token)
        for i in range(count_tokens):
            ngrams.pop(ngrams.index(token))

        max_ref = max([n_grams(ref, n).count(token) for ref in references])

        total += count_tokens if count_tokens <= max_ref else max_ref

    return bleu * (total / num_tokens)

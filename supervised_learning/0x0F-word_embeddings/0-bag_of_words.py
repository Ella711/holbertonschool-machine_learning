#!/usr/bin/env python3
"""
0. Bag Of Words
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis

    Returns: embeddings, features
    """
    count_vector = CountVectorizer(vocabulary=vocab)
    embeddings = count_vector.fit_transform(sentences).toarray()
    features = count_vector.get_feature_names_out()

    return embeddings, features

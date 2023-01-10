#!/usr/bin/env python3
"""
1. TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis

    Returns: embeddings, features
    """
    count_vector = TfidfVectorizer(vocabulary=vocab)
    embeddings = count_vector.fit_transform(sentences).toarray()
    features = count_vector.get_feature_names()

    return embeddings, features

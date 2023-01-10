#!/usr/bin/env python3
"""
3. Extract Word2Vec
"""


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    Args:
        model: trained gensim word2vec models

    Returns: trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)

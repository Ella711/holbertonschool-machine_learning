#!/usr/bin/env python3
"""
2. Train Word2Vec
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """
    Creates and trains a gensim word2vec model
    Args:
        sentences: list of sentences to be trained on
        size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word in training
        window: maximum distance between the current and predicted word
        negative: size of negative sampling
        cbow: boolean to determine the training type
        iterations: number of iterations to train over
        seed: seed for the random number generator
        workers: umber of worker threads to train the model

    Returns: trained model
    """
    model = Word2Vec(
        sentences=sentences, min_count=min_count, window=window,
        negative=negative, workers=workers, seed=seed,
        sg=(not cbow))

    model.train(
        sentences, epochs=iterations,
        total_examples=model.corpus_count)

    return model

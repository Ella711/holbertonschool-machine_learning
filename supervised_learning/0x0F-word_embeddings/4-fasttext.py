#!/usr/bin/env python3
"""
4. FastText
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a genism fastText model
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
    model = FastText(
        sentences=sentences, sg=(not cbow), negative=negative,
        window=window, min_count=min_count, workers=workers,
        seed=seed, size=size)

    model.build_vocab(sentences, update=True)

    model.train(
        sentences, epochs=iterations,
        total_examples=model.corpus_count)

    return model

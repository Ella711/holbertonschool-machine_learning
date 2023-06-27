#!/usr/bin/env python3
"""
3. Semantic Search
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents

    Args:
        corpus_path: path to the corpus of reference documents on
        which to perform semantic search
        sentence: sentence from which to perform semantic search

    Returns: reference text of the document most similar to sentence
    """
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)

    document = [sentence]

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(corpus_path + '/' + filename, encoding="utf-8") as f:
                document.append(f.read())

    embeddings = model(document)

    sentence_corr = np.inner(embeddings, embeddings)[0, 1:]

    similar = np.argmax(sentence_corr)

    return document[1 + similar]

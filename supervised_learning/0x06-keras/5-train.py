#!/usr/bin/env python3
"""
5. Validate
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    network: model to train
    data: np.ndarray - shape (m, nx) - input data
    labels: one-hot npy.ndarray - shape (m, classes) - labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data
    validation_data: data to validate the model with, if not None
    verbose: boolean determines if output should be printed during training
    shuffle: boolean determines whether to shuffle the batches every epoch

    Returns: the History object generated after training the model
    """
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size, verbose=verbose,
                          validation_data=validation_data, shuffle=shuffle)
    return history

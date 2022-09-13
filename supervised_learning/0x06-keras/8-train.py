#!/usr/bin/env python3
"""
8. Save Only the Best
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    network: model to train
    data: np.ndarray - shape (m, nx) - input data
    labels: one-hot npy.ndarray - shape (m, classes) - labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data
    validation_data: data to validate the model with, if not None
    early_stopping: boolean indicates whether early stopping should be used
    patience: patience used for early stopping
    learning_rate_decay: boolean indicates whether lrd should be used
    alpha: initial learning rate
    decay_rate: decay rate
    save_best: boolean indicating whether to save the model
        after each epoch if it is the best
    filepath: file path where the model should be saved
    verbose: boolean determines if output should be printed during training
    shuffle: boolean determines whether to shuffle the batches every epoch

    Returns: the History object generated after training the model
    """
    callback = []
    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(
            filepath=filepath, save_best_only=save_best,
        ))

    if validation_data:
        if early_stopping:
            stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=patience, mode="min")
            callback.append(stop)

        if learning_rate_decay:
            def dlr(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr_decay = K.callbacks.LearningRateScheduler(
                schedule=dlr, verbose=1
            )
            callback.append(lr_decay)

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, callbacks=callback,
                          validation_data=validation_data, shuffle=shuffle)
    return history

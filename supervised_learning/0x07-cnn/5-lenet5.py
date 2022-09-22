#!/usr/bin/env python3
"""
5. LeNet-5 (Keras)
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    x: K.Input - shape (m, 28, 28, 1): input images for the network
        m: number of images

    Returns:
    K.Model compiled to use Adam optimization & accuracy metrics
    """
    # Defining Architecture
    init = K.initializers.he_normal(seed=None)
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding="same",
                            activation="relu", kernel_initializer=init)(X)
    pool1 = K.layers.MaxPool2D(pool_size=2, strides=2)(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding="valid",
                            kernel_initializer=init, activation="relu")(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=2, strides=2)(conv2)
    f1 = K.layers.Flatten()(pool2)
    d1 = K.layers.Dense(units=120, activation="relu",
                        kernel_initializer=init)(f1)
    d2 = K.layers.Dense(units=84, activation="relu",
                        kernel_initializer=init)(d1)
    output_layer = K.layers.Dense(units=10, activation="softmax",
                                  kernel_initializer=init)(d2)

    # Define model
    model = K.Model(inputs=X, outputs=output_layer)

    # Compile
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model

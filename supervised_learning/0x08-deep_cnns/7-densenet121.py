#!/usr/bin/env python3
"""
7. DenseNet-121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely
        Connected Convolutional Networks

    growth_rate: growth rate
    compression: compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
        and a ReLU, respectively
    All weights should use he normal initialization

    Returns: keras model
    """
    X = K.Input(shape=(224, 224, 3))
    bn1 = K.layers.BatchNormalization(axis=-1)(X)
    act1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=(2 * growth_rate), kernel_size=7,
                            strides=2, padding="same", activation="linear",
                            kernel_initializer="he_normal")(act1)
    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(conv1)
    # Blocks 1
    db1, nb_filters = dense_block(pool1, pool1.shape[-1], growth_rate,
                                  layers=6)
    tb1, nb_filters = transition_layer(db1, nb_filters, compression)
    # Blocks 2
    db2, nb_filters = dense_block(tb1, nb_filters, growth_rate, layers=12)
    tb2, nb_filters = transition_layer(db2, nb_filters, compression)
    # Blocks 3
    db3, nb_filters = dense_block(tb2, nb_filters, growth_rate, layers=24)
    tb3, nb_filters = transition_layer(db3, nb_filters, compression)
    # Block 4
    db4, nb_filters = dense_block(tb3, nb_filters, growth_rate, layers=16)
    # Classification
    pool2 = K.layers.AveragePooling2D(pool_size=7, strides=7)(db4)
    Y = K.layers.Dense(units=1000, activation="softmax",
                       kernel_initializer="he_normal")(pool2)

    model = K.Model(inputs=X, outputs=Y)
    return model

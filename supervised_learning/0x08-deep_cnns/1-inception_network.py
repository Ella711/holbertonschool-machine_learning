#!/usr/bin/env python3
"""
1. Inception Network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network
    """
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            padding="same", activation="relu")(X)
    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(conv1)
    conv2 = K.layers.Conv2D(filters=192, kernel_size=3,
                            padding="same", activation="relu")(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(conv2)
    incept3a = inception_block(pool2, filters=[64, 96, 128, 16, 32, 32])
    incept3b = inception_block(incept3a, filters=[128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPool2D(pool_size=3, strides=2,
                               padding="same")(incept3b)
    incept4a = inception_block(pool3, filters=[192, 96, 208, 16, 48, 64])
    incept4b = inception_block(incept4a, filters=[160, 112, 224, 24, 64, 64])
    incept4c = inception_block(incept4b, filters=[128, 128, 256, 24, 64, 64])
    incept4d = inception_block(incept4c, filters=[112, 144, 288, 32, 64, 64])
    incept4e = inception_block(incept4d, filters=[256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPool2D(pool_size=3, strides=2,
                               padding="same")(incept4e)
    incept5a = inception_block(pool4, filters=[256, 160, 320, 32, 128, 128])
    incept5b = inception_block(incept5a, filters=[384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AveragePooling2D(pool_size=7, strides=1)(incept5b)
    drop1 = K.layers.Dropout(rate=0.4)(pool5)
    Y = K.layers.Dense(units=1000, activation="softmax")(drop1)
    model = K.models.Model(inputs=X, outputs=Y)
    return model

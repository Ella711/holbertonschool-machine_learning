#!/usr/bin/env python3
"""
4. LeNet-5 (Tensorflow 1)
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    x: tf.placeholder - shape (m, 28, 28, 1): input images for the network
        m: number of images
    y: tf.placeholder - shape (m, 10) - one-hot labels for the network

    Returns:
    a tensor for the softmax activated output
    a training operation that utilizes Adam optimization
    a tensor for the loss of the network
    a tensor for the accuracy of the network
    """
    # Defining Architecture
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding="same",
                             activation="relu", kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding="valid",
                             kernel_initializer=init, activation="relu")(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
    f1 = tf.layers.Flatten()(pool2)
    d1 = tf.layers.Dense(units=120, activation="relu",
                         kernel_initializer=init)(f1)
    d2 = tf.layers.Dense(units=84, activation="relu",
                         kernel_initializer=init)(d1)
    output_layer = tf.layers.Dense(units=10, kernel_initializer=init)(d2)
    softmax = tf.nn.softmax(output_layer)

    # Accuracy & Loss
    pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(output_layer, axis=1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, logits=output_layer)

    # Optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return softmax, train_op, loss, accuracy

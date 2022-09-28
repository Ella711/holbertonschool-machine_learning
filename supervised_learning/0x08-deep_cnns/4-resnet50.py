#!/usr/bin/env python3
"""
4. ResNet-50
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
        Deep Residual Learning for Image Recognition (2015)
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by
        batch normalization along the channels axis and a (ReLU), respectively.
    All weights should use he normal initialization

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            padding="same", activation="linear",
                            kernel_initializer="he_normal")(X)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Conv2_x
    pool1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(act1)
    filters = [64, 64, 256]
    pb2_1 = projection_block(pool1, filters, s=1)
    id2_1 = identity_block(pb2_1, filters)
    id2_2 = identity_block(id2_1, filters)

    # Conv3_x
    filters = [128, 128, 512]
    pb3_1 = projection_block(id2_2, filters, s=2)
    id3_1 = identity_block(pb3_1, filters)
    id3_2 = identity_block(id3_1, filters)
    id3_3 = identity_block(id3_2, filters)

    # Conv4_x
    filters = [256, 256, 1024]
    pb4_1 = projection_block(id3_3, filters, s=2)
    id4_1 = identity_block(pb4_1, filters)
    id4_2 = identity_block(id4_1, filters)
    id4_3 = identity_block(id4_2, filters)
    id4_4 = identity_block(id4_3, filters)
    id4_5 = identity_block(id4_4, filters)

    # Conv5_x
    filters = [512, 512, 2048]
    pb5_1 = projection_block(id4_5, filters, s=2)
    id5_1 = identity_block(pb5_1, filters)
    id5_2 = identity_block(id5_1, filters)

    # Final layers
    avgpool = K.layers.AveragePooling2D(pool_size=7, strides=7,
                                        padding="valid")(id5_2)
    Y = K.layers.Dense(units=1000, activation="softmax",
                       kernel_initializer="he_normal")(avgpool)

    model = K.Model(inputs=X, outputs=Y)
    return model

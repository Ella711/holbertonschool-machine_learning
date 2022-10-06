#!/usr/bin/env python3
"""
0. Transfer Knowledge
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for the model
    Args:
        X: np.ndarray - shape (m, 32, 32, 3) - CIFAR 10 data,
            where m is the number of data points
        Y: np.ndarray - shape (m,) - CIFAR 10 labels for X

    Returns: X_p, Y_p
    """
    X_p = K.applications.efficientnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p


if __name__ == "__main__":

    def resize_images(X):
        """
        Resize images to be used in EfficientNetv2
        Args:
            X: images

        Returns: resized image
        """
        return K.backend.resize_images(X, 7, 7,
                                       data_format="channels_last")

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    xtrain, ytrain = preprocess_data(X_train, Y_train)
    xtest, ytest = preprocess_data(X_test, Y_test)

    input = K.Input(shape=(32, 32, 3))
    effnetv2_base = K.applications.EfficientNetV2S(weights="imagenet",
                                                   include_top=False,
                                                   input_shape=(224, 224, 3))

    print(len(effnetv2_base.layers))
    for layer in effnetv2_base.layers[:300]:
        layer.trainable = False

    model = K.models.Sequential([
        K.layers.Lambda(resize_images),
        effnetv2_base,
        K.layers.Flatten(),
        K.layers.BatchNormalization(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.BatchNormalization(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dropout(0.4),
        K.layers.BatchNormalization(),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dropout(0.4),
        K.layers.BatchNormalization(),
        K.layers.Dense(10, activation='softmax'),
    ])

    calls = [K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                         save_best_only=True)]

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(xtrain, ytrain,
              batch_size=32,
              epochs=10,
              validation_data=(xtest, ytest),
              shuffle=True,
              callbacks=calls)

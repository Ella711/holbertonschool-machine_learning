#!/usr/bin/env python3
"""
0. "Vanilla" Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
            hidden layer in the encoder,
        latent_dims: int containing the dimensions of the latent
            space representation

    Returns: encoder, decoder, auto
    """
    k = keras.layers
    input = keras.Input(shape=input_dims)
    decoded_input = keras.Input(shape=latent_dims)

    encoded_layer = k.Dense(hidden_layers[0], activation='relu')(input)
    for node in hidden_layers[1:]:
        encoded_layer1 = k.Dense(node, activation='relu')(encoded_layer)
    encoded_layer2 = k.Dense(latent_dims, activation='relu')(encoded_layer1)
    encoder = keras.Model(input, encoded_layer2)

    decoded_layer = k.Dense(
        hidden_layers[-1], activation='relu')(decoded_input)
    for dim in hidden_layers[-2::-1]:
        decoded_layer1 = k.Dense(dim, activation='relu')(decoded_layer)
    decoded_layer2 = k.Dense(input_dims, activation='sigmoid')(decoded_layer1)
    decoder = keras.Model(decoded_input, decoded_layer2)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto

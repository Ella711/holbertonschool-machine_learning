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
    coded_input = keras.Input(shape=latent_dims)

    encoded_layer = k.Dense(hidden_layers[0], activation='relu')(input)
    for node in hidden_layers[1:]+[latent_dims]:
        encoded_layer = k.Dense(node, activation='relu')(encoded_layer)

    decoded_layer = k.Dense(
        hidden_layers[-1], activation='relu')(coded_input)
    for x, nodes in enumerate(list(reversed(hidden_layers[:-1])) + [input_dims]):
        if x == len(hidden_layers) - 1:
            decoded_layer = k.Dense(nodes, activation='sigmoid')(decoded_layer)
        else:
            decoded_layer = k.Dense(nodes, activation='relu')(decoded_layer)

    encoder = keras.Model(input, encoded_layer)
    decoder = keras.Model(coded_input, decoded_layer)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto

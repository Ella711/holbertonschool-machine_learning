#!/usr/bin/env python3
"""
3. Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
            hidden layer in the encoder,
        latent_dims: int containing the dimensions of the latent
            space representation

    Returns: encoder, decoder, auto
    """
    k = keras.layers

    def sampling(args):
        mean, log = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mean)[0], latent_dims),
            mean=0,
            stddev=1
        )
        return mean + keras.backend.exp(log / 2) * epsilon

    input = keras.Input(shape=input_dims)
    encode = k.Dense(hidden_layers[0], activation='relu')(input)
    for layer in hidden_layers[1:]:
        encode = k.Dense(layer, activation='relu')(encode)
    encode_mean = k.Dense(latent_dims)(encode)
    encode_log = k.Dense(latent_dims)(encode)
    encoding = k.Lambda(sampling)([encode_mean, encode_log])
    encoder = keras.Model(input, [encode_mean, encode_log, encoding])

    coded_input = keras.Input(shape=latent_dims)
    decode = k.Dense(hidden_layers[-1], activation='relu')(coded_input)
    for layer in hidden_layers[-2::-1]:
        decode = k.Dense(layer, activation='relu')(decode)
    decode = k.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(coded_input, decode)

    auto = keras.Model(input, decoder(encoder(input)[-1]))

    def vae_loss(inputs, outputs):
        r_loss = keras.losses.binary_crossentropy(inputs, outputs)
        r_loss *= input_dims
        k_exp = keras.backend.exp(encode_log)
        k_square = keras.backend.square(encode_mean)
        lat_loss = -0.5 * keras.backend.sum(1 + encode_log - k_exp
                                            - k_square, axis=-1)
        return keras.backend.mean(lat_loss + r_loss)

    auto.compile(loss=vae_loss, optimizer='adam')

    return encoder, decoder, auto

#!/usr/bin/env python3
"""
11. Save and Load Configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format

    network: model whose weights should be saved
    filename: path of the file that the weights should be saved to

    Return: None
    """
    json_config = network.to_json()
    with open(filename, "w") as f:
        f.write(json_config)


def load_config(filename):
    """
    Loads a model with a specific configuration

    filename: path of the file that the weights should be saved to

    Returns: the loaded model
    """
    with open(filename, "r") as f:
        json_config = f.read()
    model = K.models.model_from_json(json_config)
    return model

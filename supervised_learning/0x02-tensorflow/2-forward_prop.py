#!/usr/bin/env python3
"""
Creates forward prop
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Creates the forward propagation graph for the neural network """
    for i in range(len(layer_sizes)):
        if i == 0:
            prev = create_layer(x, layer_sizes[i], activations[i])
        else:
            prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev

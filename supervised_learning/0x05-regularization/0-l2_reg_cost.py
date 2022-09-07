#!/usr/bin/env python3
"""
0. L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the neural network
    m: number of data points used

    Returns: the cost of the network accounting for L2 regularization
    """
    l2_reg_cost = 0
    for layer in range(1, L + 1):
        l2_reg_cost += np.sum(np.square(weights["W" + str(layer)]))
    l2_cost = (1/m) * (lambtha/2) * l2_reg_cost
    return cost + l2_cost

#!/usr/bin/env python3
"""
7. RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid division by zero
    var: np.ndarray containing the variable to be updated
    grad: np.ndarray containing the gradient of var
    s: previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """
    dwvar = beta2 * s + (1 - beta2) * np.square(grad)
    var -= alpha * (grad / (epsilon + np.sqrt(dwvar)))
    return var, dwvar

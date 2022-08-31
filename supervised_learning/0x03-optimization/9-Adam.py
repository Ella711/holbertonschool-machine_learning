#!/usr/bin/env python3
"""
9. Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid division by zero
    var: np.ndarray containing the variable to be updated
    grad: np.ndarray containing the gradient of var
    v: previous first moment of var
    s: previous second moment of var
    t: time step used for bias correction
    Returns: the updated variable, the new first moment, and the new
        second moment, respectively
    """
    dwvar = beta1 * v + (1 - beta1) * grad
    dws = beta2 * s + (1 - beta2) * np.square(grad)
    dwvar_c = dwvar / (1 - beta1 ** t)
    dws_c = dws / (1 - beta2 ** t)
    var -= alpha * dwvar_c / (epsilon + np.sqrt(dws_c))
    return var, dwvar, dws

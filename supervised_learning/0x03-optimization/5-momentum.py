#!/usr/bin/env python3
"""
5. Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent
        with momentum optimization algorithm

    alpha: learning rate
    beta1: momentum weight
    var: np.ndarray containing the variable to be updated
    grad: np.ndarray containing the gradient of var
    v: previous first moment of var
    Returns: the updated variable and the new moment, respectively
    """
    dvar_prev = beta1 * v + (1 - beta1) * grad
    var -= alpha * dvar_prev
    return var, dvar_prev

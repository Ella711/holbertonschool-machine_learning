#!/usr/bin/env python3
"""
Module with various policy functions
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes to policy with a weight of a matrix
    """
    product = matrix @ weight
    exp = np.exp(product)
    policy = exp / np.sum(exp)
    return policy


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based by state and weight matrix

    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight

    Return: action and gradient
    """
    Policy = policy(state, weight)
    action = np.random.choice(len(Policy[0]), p=Policy[0])
    softmax = Policy.reshape(-1, 1)
    grad_soft = (np.diagflat(softmax) - softmax @ softmax.T)[action, :]
    dlog = grad_soft / Policy[0, action]
    gradient = state.T @ dlog[None, :]
    return action, gradient

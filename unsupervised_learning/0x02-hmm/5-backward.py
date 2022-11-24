#!/usr/bin/env python3
"""
5. The Backward Algorithm
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Args:
        Observation: np.ndarray - (T,) - contains index of the observation
            T: number of observations
        Emission: np.ndarray - (N, M) - contains emission probability of
            a specific observation given a hidden state
            N is the number of hidden states
            M is the number of all possible observations
        Transition: 2D np.ndarray - (N, N) - contains the transition
            probabilities
        Initial: np.ndarray - (N, 1) - contains the probability of starting
            in a particular hidden state

    Returns: P, B, or None, None on failure
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[0] != Emission.shape[0] or Initial.shape[1] != 1:
        return None, None

    N, M = Emission.shape
    T = Observation.size

    B = np.ones((N, T), dtype="float")

    for t in range(T - 2, -1, -1):
        mat = (Emission[:, Observation[t + 1]] * Transition.reshape(N, 1, N))
        mat = (B[:, t + 1] * mat).reshape(N, N).sum(axis=1)

        B[:, t] = mat

    P = (Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P.sum(axis=1)[0], B

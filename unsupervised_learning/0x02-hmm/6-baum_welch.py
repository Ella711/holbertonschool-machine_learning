#!/usr/bin/env python3
"""
6. The Baum-Welch Algorithm
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations: np.ndarray - (T,) - contains index of the observation
            T: number of observations
        Emission: np.ndarray - (M, N) - contains the initialized emission
            probabilities
            N is the number of output states
        Transition: 2D np.ndarray - (M, M) - contains the initialized
            transition probabilities
            M is the number of hidden states
        Initial: np.ndarray - (M, 1) - contains the initialized starting
            probabilities
        iterations: number of times expectation-maximization should be done
    Returns: P, B, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if Observations.shape[0] == 0:
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

    return None, None

#!/usr/bin/env python3
"""
4. The Viretbi Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """

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

    Returns: path, P, or None, None on failure
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

    N, _ = Emission.shape
    T = Observation.size

    mu = Initial * Emission[:, Observation[0]][..., np.newaxis]
    aux = np.zeros((N, T))

    for t in range(1, T):
        mat = (Emission[:, Observation[t]] * Transition.reshape(N, 1, N))
        mat = (mat.reshape(N, N) * mu[:, t - 1].reshape(N, 1))

        mx = np.max(mat, axis=0).reshape(N, 1)
        mu = np.concatenate((mu, mx), axis=1)
        aux[:, t] = np.argmax(mat, axis=0).T

    P = np.max(mu[:, T - 1])
    link = np.argmax(mu[:, T - 1])
    path = [link]

    for t in range(T - 1, 0, -1):
        idx = int(aux[link, t])
        path.append(idx)
        link = idx

    return path[::-1], P

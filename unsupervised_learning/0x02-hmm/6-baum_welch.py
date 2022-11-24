#!/usr/bin/env python3
"""
6. The Baum-Welch Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
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

    F = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        aux = F[:, t - 1].T * Transition.T.reshape(N, 1, N)
        aux = aux * Emission[:, Observation[t]].reshape(N, 1, 1)
        F = np.concatenate((F, aux.sum(-1)), axis=1)

    return np.sum(F[:, -1]), F


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
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


def EM(Observation, Transition, Emission, Initial):
    """
    Expectation Maximization algorithm for updating transition,
    emission, and initial state probabilities
    """

    T = Observation.size
    M, N = Emission.shape
    _, F = forward(Observation, Emission, Transition, Initial)
    _, B = backward(Observation, Emission, Transition, Initial)

    Xi = np.zeros((T, M, M))

    for t in range(T):
        if t == T - 1:
            op = F[:, t].reshape(M, 1) * Transition
            Xi[t, :, :] = op.copy()
            break

        op = F[:, t].reshape(M, 1) * Transition * Emission[:, Observation[t+1]]
        op = op * B[:, t+1]
        Xi[t, :, :] = op.copy()

    Xi = Xi / Xi.sum(axis=(1, 2)).reshape(T, 1, 1)

    Transition = (Xi[:T-1, :, :].sum(axis=0) /
                  Xi[:T-1, :, :].sum(axis=(0, 2)).reshape(M, 1))

    for k in range(N):
        idxs = Observation[:T] == k
        Emission[:, k] = Xi[idxs, :, :].sum(axis=(0, 2))/Xi.sum(axis=(0, 2))

    Initial = Xi[0].sum(axis=0)

    return Transition, Emission, Initial.reshape(M, 1)


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

    for i in range(iterations):
        Transition, Emission, Initial = EM(
            Observations, Transition, Emission, Initial)

    return Transition, Emission

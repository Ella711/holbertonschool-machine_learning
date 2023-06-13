#!/usr/bin/env python3
"""
2. Epsilon Greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Args:
        Q: numpy.ndarray containing the q-table
        state: current state
        epsilon: epsilon to use for the calculation

    Returns: next action index
    """
    exploration_rate = np.random.uniform()
    if exploration_rate > epsilon:
        action_idx = np.argmax(Q[state, :])
    else:
        action_idx = np.argmax(Q.shape[1])
    return action_idx

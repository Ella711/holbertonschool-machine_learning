#!/usr/bin/env python3
"""
1. Initialize Q-table
"""
import numpy as np


def q_init(env):
    """
    initializes the Q-table
    Args:
        env: FrozenLakeEnv instance

    Returns: Q-table as a numpy.ndarray of zeros
    """
    actionspace = env.action_space.n
    statespace = env.observation_space.n
    qtable = np.zeros((statespace, actionspace))
    return qtable

#!/usr/bin/env python3
"""
0. Load the Environment
"""
import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from OpenAI’s gym
    Args:
        desc: None or a list of lists containing a custom description of the map to load
        map_name: None or a string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery

    Returns: environment
    """
    env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env

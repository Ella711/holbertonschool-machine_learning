#!/usr/bin/env python3
"""
3. Q-learning
"""
import numpy as np
import gym


def play(env, Q, max_steps=100):
    """
    The trained agent plays an episode
    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns: total rewards for the episode
    """
    state = env.reset()[0]
    done = False
    env.render()
    total_rewards = 0
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, done, _, _, = env.step(action)
        env.render()
        total_rewards += reward
        if done:
            break
        state = next_state
    return total_rewards

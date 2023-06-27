#!/usr/bin/env python3
"""
1. TD(λ)
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm

    Args:
        env: openAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns next action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns: V, the updated value estimate
    """
    # Initiliaze Eligibility Traces
    ET = [0 for i in range(env.observation_space.n)]
    for ep in range(episodes):
        # Get initial state
        state = env.reset()[0]
        for step in range(max_steps):
            # Update all ETs
            ET = list(np.array(ET) * lambtha * gamma)
            ET[state] += 1

            # Get action and next state
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            # Update Value estimate
            delta = reward + gamma * V[next_state] - V[state]
            V[state] += alpha * delta * ET[state]

            if done:
                break
            state = next_state

    return np.array(V)

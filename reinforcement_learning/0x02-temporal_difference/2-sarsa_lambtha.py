#!/usr/bin/env python3
"""
2. SARSA(λ)
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon greedy
    """
    # determine exploring-exploiting
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ)

    Args:
        env: openAI environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table
    """
    # Save epsilon start value
    max_epsilon = epsilon
    # Initializes Eligibility Traces
    ET = np.zeros((Q.shape))
    for episode in range(episodes):
        # Get initial state
        state = env.reset()[0]

        action = epsilon_greedy(Q, state, epsilon)
        for _ in range(max_steps):
            # Update ET
            ET *= lambtha * gamma
            # increase Et for current state, action
            ET[state, action] += 1

            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, state, epsilon)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1
            # BACKWARD VIEW SARSA(λ)
            # Update Q table
            delta = reward + (
                gamma * Q[next_state, next_action]
            ) - Q[state, action]
            Q[state, action] += (alpha * delta * ET[state, action])
            if done:
                break

            state = next_state
            action = next_action

        # Update epsilon
        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode))

    return Q

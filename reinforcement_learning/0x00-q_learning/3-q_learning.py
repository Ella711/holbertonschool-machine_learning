#!/usr/bin/env python3
"""
3. Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning
    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns: Q, total_rewards
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        current_reward = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            Q[state, action] = Q[state, action] * (1 - alpha) + alpha * \
                (reward + gamma * np.max(Q[new_state, :]))
            current_reward += reward
            if done:
                break
            state = new_state
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        total_rewards.append(current_reward)
    return Q, total_rewards

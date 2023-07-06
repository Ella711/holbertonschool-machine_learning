#!/usr/bin/env python3
"""
Module with various training functions for policy gradient
"""
import numpy as np
from policy_gradient import policy_gradient


def single_episode(env, weight, episode, show_result):
    """
    Play one single episode
    """
    state = env.reset()[None, :]
    return_gradient = []

    while True:
        if show_result and (episode % 1000 == 0):
            env.render()
        action, gradient = policy_gradient(state, weight)
        state, reward, done, _ = env.step(action)
        state = state[None, :]
        return_gradient.append((state, action, reward, gradient))
        if done:
            break
    env.close()
    return return_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Return: all values of the score
            (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    episodes = []

    for episode in range(nb_episodes):
        single = single_episode(env, weight, episode, show_result)
        T = len(single) - 1

        sum_rewards = 0
        for t in range(0, T):
            _, _, reward, gradient = single[t]
            sum_rewards += reward
            G = np.sum([
                gamma**single[k][2] *
                single[k][2] for k in range(t + 1, T + 1)])
            weight += alpha * G * gradient
        episodes.append(sum_rewards)
        print("{}: {}".format(episode, sum_rewards), end="\r", flush=False)
    return episodes

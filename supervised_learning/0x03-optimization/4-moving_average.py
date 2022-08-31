#!/usr/bin/env python3
"""
4. Moving Average
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average

    Returns: a list containing the moving averages of data
    """
    beta_prev = 0
    moving_averagelist = []
    for i in range(len(data)):
        beta_prev = beta_prev * beta + ((1 - beta) * data[i])
        bias_correction = 1 - (beta ** (i + 1))

        moving_averagelist.append(beta_prev / bias_correction)
    return moving_averagelist

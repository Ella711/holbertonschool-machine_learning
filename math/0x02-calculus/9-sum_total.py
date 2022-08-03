#!/usr/bin/env python3
"""
Function that calculates the summation of i squared
"""


def summation_i_squared(n):
    """ Returns summation of i squared """
    if isinstance(n, int) and n > 1:
        return sum(map(lambda n: n**2, range(n + 1)))
    return None

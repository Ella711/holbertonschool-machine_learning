#!/usr/bin/env python3
"""
Contains the class Binomial that represents a binomial distribution
"""


def factorial(x):
    """ Function that calculates the factorial of a given number """
    fact = 1
    for i in range(1, x + 1):
        fact *= i
    return fact


class Binomial:
    """ Represents a normal distribution """

    def __init__(self, data=None, n=1., p=0.5):
        """
        Class constructor

        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a 'success'
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            stddev = sum(((x - mean) ** 2) for x in data) / len(data)
            self.p = 1 - (stddev / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """ Calculates the PMF for a given number of 'successes' """
        if k < 0:
            return 0
        k = int(k)
        nCx = factorial(self.n) / (factorial(self.n - k) * factorial(k))
        return nCx * pow(self.p, k) * pow(1 - self.p, self.n - k)

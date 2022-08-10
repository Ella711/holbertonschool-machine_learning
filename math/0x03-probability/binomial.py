#!/usr/bin/env python3
"""
Contains the class Binomial that represents a binomial distribution
"""


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
            self.n = n
            self.p = float(p)

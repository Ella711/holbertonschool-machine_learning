#!/usr/bin/env python3
"""
Contains the class Exponential that represents an exponential distribution
"""


class Exponential:
    """ Represents an exponential distribution """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurrences in a given time frame
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ Calculates the PDF for a given time period """
        if x < 0:
            return 0
        return self.lambtha * pow(self.e, -(x * self.lambtha))

    def cdf(self, x):
        """ Calculates the CDF for a given time period """
        if x < 0:
            return 0
        return 1 - pow(self.e, -self.lambtha * x)

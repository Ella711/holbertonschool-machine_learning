#!/usr/bin/env python3
"""
Contains the class Normal that represents a normal distribution
"""


class Normal:
    """ Represents a normal distribution """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            length = len(data)
            self.mean = sum(data) / length
            self.stddev = (sum(map(lambda n: pow(n - self.mean, 2),
                                   data)) / length) ** .5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ Calculates the z-score of a given x value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates the PDF for a given x-value """
        numerator = pow(self.e, (-(x - self.mean) ** 2)/(2 * self.stddev ** 2))
        denominator = self.stddev * pow(2 * self.pi, 0.5)
        return numerator / denominator

    def erf(self, x):
        """ Calculates the erf of a given x-value """
        left = 2 / self.pi**0.5
        right = x - (x**3 / 3) + (x**5 / 10) - (x**7 / 42) + (x**9 / 216)
        return left * right

    def cdf(self, x):
        """ Calculates the CDF of a given x-value """
        z = (x - self.mean) / (self.stddev * (2**0.5))
        return (self.erf(z) + 1) / 2

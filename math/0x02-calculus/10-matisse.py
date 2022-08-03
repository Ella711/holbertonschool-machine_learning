#!/usr/bin/env python3
"""
Function that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """ Returns the derivative of a polynomial """
    derivative = []
    if type(poly) == list and len(poly) >= 1:
        if len(poly) == 1:
            return [0]
        for i in range(1, len(poly)):
            derivative.append(i * poly[i])
        return derivative
    return None

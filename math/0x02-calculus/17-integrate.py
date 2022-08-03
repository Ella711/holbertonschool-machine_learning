#!/usr/bin/env python3
"""
Function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """ Returns the integral of a polynomial """
    if poly == [0]:
        return [C]
    if type(poly) == list and len(poly) >= 1 and type(C) == int or float:
        integral = [C]
        for i in range(1, len(poly) + 1):
            integ = poly[i - 1] / i
            if (integ - int(integ)) == 0:
                integral.append(int(integ))
            else:
                integral.append(integ)
        return integral
    return None

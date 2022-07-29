#!/usr/bin/env python3
"""
Function that performs element-wise addition, subtraction,
multiplication and division
"""


def np_elementwise(mat1, mat2):
    """ Function that returns an element-wise operation """
    addition = mat1 + mat2
    subtract = mat1 - mat2
    product = mat1 * mat2
    quotient = mat1 / mat2
    return addition, subtract, product, quotient

#!/usr/bin/env python3
"""
Function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """ Returns the shape of a matrix """
    shape = []
    try:
        while len(matrix) > 0:
            shape.append(len(matrix))
            matrix = matrix[0]
    except TypeError:
        pass
    return shape

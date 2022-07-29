#!/usr/bin/env python3
"""
Function that adds two matrices element-wise
- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If mat1 and mat2 are not the same shape, return None
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


def add_matrices2D(mat1, mat2):
    """ Returns the addition of two matrices """
    if matrix_shape(mat1) == matrix_shape(mat2):
        summ = []
        for i in range(len(mat1)):
            summ1 = []
            for j in range(len(mat1[0])):
                summ1.append(mat1[i][j] + mat2[i][j])
            summ.append(summ1)
        return summ
    return None

#!/usr/bin/env python3
"""
Function that returns the transpose of a 2D matrix
- You must return a new matrix
- You can assume that matrix is never empty
- You can assume all elements in the same dimension are of the same type/shape
"""


def matrix_transpose(matrix):
    """ Returns a matrix transposed """
    rows = len(matrix)
    columns = len(matrix[0])

    matrix_t = []
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        matrix_t.append(row)
    return matrix_t

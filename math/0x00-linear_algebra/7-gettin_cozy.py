#!/usr/bin/env python3
"""
Function that concatenates two matrices along a specific axis
- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be concatenated, return None
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Returns the concatenation of two matrices """
    mat1copy = [row[:] for row in mat1]
    mat2copy = [row[:] for row in mat2]
    try:
        if axis == 0:
            concat = mat1copy + mat2copy
            return concat
        if axis == 1:
            concat = []
            for mats in range(len(mat1copy)):
                concat.append(mat1copy[mats] + mat2copy[mats])
            return concat
    except:
        pass

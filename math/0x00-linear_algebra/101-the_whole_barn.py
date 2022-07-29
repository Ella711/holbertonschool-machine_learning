#!/usr/bin/env python3
"""
Function that adds 2 matrices
"""


def add_matrices(mat1, mat2):
    """ Returns the addition of two matrices """
    import numpy as np
    mat1copy = np.array([row[:] for row in mat1])
    mat2copy = np.array([row[:] for row in mat2])
    if mat1copy.shape != mat2copy.shape:
        return None
    result = mat1copy + mat2copy
    return result

#!/usr/bin/env python3
"""
Function that adds 2 matrices
"""


def add_matrices(mat1, mat2):
    """ Returns the addition of two matrices """
    import numpy as np
    mat1, mat2 = np.array(mat1), np.array(mat2)
    if mat1.shape != mat2.shape:
        return None
    else:
        result = mat1 + mat2
        return result

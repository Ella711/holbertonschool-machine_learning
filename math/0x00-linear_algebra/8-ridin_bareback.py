#!/usr/bin/env python3
"""
Function that performs matrix multiplication
- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- If the two matrices cannot be multiplied, return None
"""


def mat_mul(mat1, mat2):
    """ Multiplies two matrices """
    if len(mat1[0]) == len(mat2):
        product = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]
        for x in range(len(mat1)):
            for y in range(len(mat2[0])):
                for z in range(len(mat2)):
                    product[x][y] += mat1[x][z] * mat2[z][y]
        return product
    return None

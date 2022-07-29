#!/usr/bin/env python3
"""
Function that slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """ Slice matrix based on axis given """
    sliced = (max(axes) + 1) * [slice(None)]
    for k, v in axes.items():
        sliced[k] = slice(*v)
    result = matrix[tuple(sliced)]
    return result

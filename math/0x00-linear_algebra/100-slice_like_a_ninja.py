#!/usr/bin/env python3
"""
Function that slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """ Slice matrix based on axis given """
    slices = []
    for i in range(matrix.ndim):
        val = axes.get(i)
        if val is not None:
            slices.append(slice(*val))
        else:
            slices.append(slice(None, None, None))
    return matrix[slices]


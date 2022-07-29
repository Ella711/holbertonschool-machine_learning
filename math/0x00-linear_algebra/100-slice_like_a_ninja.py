#!/usr/bin/env python3
"""
Function that slices a matrix along specific axes
- axes is a dictionary where the key is an axis to slice along
    and the value is a tuple representing the slice to make along that axis
- You can assume that axes represents a valid slice
"""


def np_slice(matrix, axes={}):
    """ Slice matrix based on axis given """
    slices = []
    for i in range(matrix.ndim):
        values = axes.get(i)
        if values is not None:
            slices.append(slice(*values))
        else:
            slices.append(slice(None, None, None))
    return matrix[slices]

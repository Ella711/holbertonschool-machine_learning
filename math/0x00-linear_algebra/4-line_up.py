#!/usr/bin/env python3
"""
Function that adds two arrays element-wise
- You can assume that arr1 and arr2 are lists of ints/floats
- You must return a new list
- If arr1 and arr2 are not the same shape, return None
"""


def add_arrays(arr1, arr2):
    """ Returns the addition of two arrays"""
    if len(arr1) == len(arr2):
        summ = []
        for i in range(len(arr1)):
            summ.append(arr1[i] + arr2[i])
        return summ
    return None

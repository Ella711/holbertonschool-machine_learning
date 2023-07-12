#!/usr/bin/env python3
"""
0. From Numpy
"""
import pandas as pd
import string


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Args:
        array: np.ndarray from which to create the pd.DataFrame

    Returns: the newly created pd.DataFrame
    """
    upper_ascii = string.ascii_uppercase
    df = pd.DataFrame(array, columns=[upper_ascii[x]
                                      for x in range(len(array[0]))])
    return df

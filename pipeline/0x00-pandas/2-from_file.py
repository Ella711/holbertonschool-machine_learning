#!/usr/bin/env python3
"""
2. From File
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Args:
        filename: file to load from
        delimiter: column separator

    Returns: loaded pd.DataFrame
    """
    df = pd.read_csv(filename, sep=delimiter)
    return df

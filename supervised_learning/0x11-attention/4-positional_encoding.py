#!/usr/bin/env python3
"""
3. Positional Encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    Args:
        max_seq_len: int - maximum sequence length
        dm: model depth

    Returns: np.ndarray - (max_seq_len, dm) - positional encoding vectors
    """
    position = np.arange(max_seq_len)
    depth = np.arange(dm)
    angle = position[:, None] / np.power(10000,
                                         ((2 * (depth[None, :] // 2)) /
                                          np.float32(dm)))
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle

#!/usr/bin/env python3
"""
1. Minor
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1

    for i, mat in enumerate(matrix):
        if len(matrix) != len(mat):
            raise ValueError("matrix must be a square matrix")
        if not isinstance(mat, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    det = 0
    for idx, num in enumerate(matrix):
        aux_mat = []
        product = matrix[0][idx]
        for row in matrix[1:]:
            mat = []
            for j in range(len(matrix)):
                if j != idx:
                    mat.append(row[j])
            aux_mat.append(mat)
        det += product * determinant(aux_mat) * (-1) ** idx
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Args:
        matrix: list of lists whose minor matrix should be calculated

    Returns: the minor matrix of matrix
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor_mat = []
    for i in range(len(matrix)):
        aux = []
        for num in range(len(matrix[0])):
            mat = []
            for row in (matrix[:i] + matrix[i + 1:]):
                mat.append(row[:num] + row[num + 1:])
            aux.append(determinant(mat))
        minor_mat.append(aux)
    return minor_mat

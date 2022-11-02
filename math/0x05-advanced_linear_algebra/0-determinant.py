#!/usr/bin/env python3
"""
0. Determinant
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns: the determinant of matrix
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

#!/usr/bin/env python3
"""
4. Inverse
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


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    cofactor_matrix = minor(matrix)

    sign_start = 1

    for row, _ in enumerate(cofactor_matrix):
        sign = sign_start
        for col, _ in enumerate(cofactor_matrix):
            cofactor_matrix[row][col] *= sign
            sign *= -1
        sign_start *= -1

    return cofactor_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    temp_matrix = cofactor(matrix)
    adjugate_matrix = [[] for x in temp_matrix]

    for row, _ in enumerate(temp_matrix):
        for col, _ in enumerate(temp_matrix):
            adjugate_matrix[col].append(temp_matrix[row][col])

    return adjugate_matrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Args:
        matrix: list of lists whose inverse should be calculated

    Returns: the inverse of matrix, or None if matrix is singular
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    inverse_matrix = adjugate(matrix)
    det = determinant(matrix)

    if det == 0:
        return None

    det = 1 / det

    for row, _ in enumerate(inverse_matrix):
        for col, _ in enumerate(inverse_matrix):
            inverse_matrix[row][col] *= det

    return inverse_matrix

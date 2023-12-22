# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# static_typing.pyx 

import numpy as np
cimport numpy as np

cpdef double array_max(double[:] arr):
    """
    Find the maximum value in a 1D array using static typing.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The maximum value in the array.
    """
    cdef int i
    cdef double max_value = arr[0]
    for i in range(1, arr.shape[0]):
        if arr[i] > max_value:
            max_value = arr[i]
    return max_value

cpdef double array_min(double[:] arr):
    """
    Find the minimum value in a 1D array using static typing.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The minimum value in the array.
    """
    cdef int i
    cdef double min_value = arr[0]
    for i in range(1, arr.shape[0]):
        if arr[i] < min_value:
            min_value = arr[i]
    return min_value

cpdef double array_mean(double[:] arr):
    """
    Calculate the mean of a 1D array using static typing.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The mean of the array elements.
    """
    return sum_array(arr) / arr.shape[0]

cpdef np.ndarray[np.float64_t, ndim=2] transpose_matrix(np.ndarray[np.float64_t, ndim=2] matrix):
    """
    Transpose a 2D numpy array (matrix).

    Parameters
    ----------
    matrix : ndarray
        A 2D numpy array of type np.float64.

    Returns
    -------
    ndarray
        A 2D numpy array representing the transposed matrix.
    """
    cdef int rows = matrix.shape[0]
    cdef int cols = matrix.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] transposed = np.zeros((cols, rows), dtype=np.float64)

    cdef int i, j
    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = matrix[i, j]

    return transposed

cpdef double dot_product(double[:] vec1, double[:] vec2):
    """
    Compute the dot product of two 1D vectors.

    Parameters
    ----------
    vec1 : memoryview
        A 1D memoryview of a double array (first vector).
    vec2 : memoryview
        A 1D memoryview of a double array (second vector).

    Returns
    -------
    double
        The dot product of the two vectors.

    Notes
    -----
    The function assumes both vectors are of the same length.
    """
    cdef int i
    cdef double result = 0.0
    for i in range(vec1.shape[0]):
        result += vec1[i] * vec2[i]
    return result

cpdef np.ndarray[np.float64_t, ndim=2] elementwise_multiply_matrices(np.ndarray[np.float64_t, ndim=2] mat1, 
                                                                     np.ndarray[np.float64_t, ndim=2] mat2):
    """
    Perform element-wise multiplication of two 2D matrices.

    Parameters
    ----------
    mat1 : ndarray
        First input matrix, a 2D numpy array of type np.float64.
    mat2 : ndarray
        Second input matrix, a 2D numpy array of type np.float64.

    Returns
    -------
    ndarray
        A 2D numpy array representing the element-wise multiplication of the two matrices.

    Notes
    -----
    Both matrices must have the same dimensions.
    """
    cdef int rows = mat1.shape[0]
    cdef int cols = mat1.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float64)

    cdef int i, j
    for i in range(rows):
        for j in range(cols):
            result[i, j] = mat1[i, j] * mat2[i, j]

    return result

# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# parallel_processing.pyx 
"""
Created on Fri Dec 22 17:20:17 2023

@author: Daniel
"""
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython.parallel import prange


cpdef double parallel_sum(double[:] arr):
    """
    Compute the sum of a 1D array in parallel using OpenMP.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The sum of the array elements.
    """
    cdef int i
    cdef double total = 0.0
    for i in prange(arr.shape[0], nogil=True):
        total += arr[i]
    return total

cpdef double[:] parallel_multiply_arrays(double[:] arr1, double[:] arr2):
    """
    Perform parallel elementwise multiplication of two 1D arrays using OpenMP.

    Parameters
    ----------
    arr1 : memoryview
        First input array.
    arr2 : memoryview
        Second input array.

    Returns
    -------
    memoryview
        A new array resulting from elementwise multiplication.
    """
    cdef int i
    cdef int n = arr1.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.double)
    for i in prange(n, nogil=True):
        result[i] = arr1[i] * arr2[i]
    return result

cpdef double parallel_max(double[:] arr):
    """
    Find the maximum value in a 1D array in parallel using OpenMP.

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
    for i in prange(1, arr.shape[0], nogil=True):
        if arr[i] > max_value:
            max_value = arr[i]
    return max_value



def parallel_matrix_vector_multiply(np.ndarray[np.float64_t, ndim=2] matrix, 
                                    np.ndarray[np.float64_t, ndim=1] vector):
    """
    Perform parallel matrix-vector multiplication.
    
    This function, parallel_matrix_vector_multiply, takes a 2D NumPy 
    array (matrix) and a 1D NumPy array (vector) as inputs and returns a 
    1D NumPy array as the output. The function assumes that the matrix's 
    number of columns is equal to the vector's length. It uses parallel 
    processing (with prange and nogil) to speed up the computation, 
    especially beneficial for large matrices.

   To compile and use this Cython code, you'll need a setup script or a 
   Jupyter notebook configured to compile Cython code. Also, ensure that 
   the compiler you're using supports OpenMP for parallel processing 
   capabilities. The use of nogil allows the function to release the 
   Global Interpreter Lock (GIL) in Python, enabling true multi-threaded
   parallelism.

    Parameters
    ----------
    matrix : ndarray
        A 2D numpy array (matrix) of shape (m, n).
    vector : ndarray
        A 1D numpy array (vector) of length n.

    Returns
    -------
    ndarray
        A 1D numpy array of length m, which is the result of the 
        matrix-vector multiplication.

    Notes
    -----
    This function assumes that the number of columns in the matrix 
    (n) is the same as the length
    of the vector. If this is not the case, the function will not behave correctly.
    """
    cdef int m = matrix.shape[0]
    cdef int n = matrix.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(m, dtype=np.float64)
    
    cdef int i, j
    with nogil, parallel():
        for i in prange(m):
            for j in range(n):
                result[i] += matrix[i, j] * vector[j]

    return result

# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# memory_view.pyx 

cpdef typed_memoryview_sum(double[:] arr):
    """
    Calculates the sum of a 1D array using typed memoryviews for efficiency.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The sum of the elements in the array.
    """
    cdef double total = 0.0
    cdef int i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

cpdef double sum_array(double[:] arr):
    """
    Calculate the sum of elements in a 1D array using typed memoryviews.

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
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

cpdef double[:] multiply_arrays(double[:] arr1, double[:] arr2):
    """
    Perform elementwise multiplication of two 1D arrays using typed memoryviews.

    Parameters
    ----------
    arr1 : memoryview
        First input array.
    arr2 : memoryview
        Second input array.

    Returns
    -------
    memoryview
        A new array that is the result of elementwise multiplication.
    """
    cdef int i
    cdef int n = arr1.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.double)
    for i in range(n):
        result[i] = arr1[i] * arr2[i]
    return result

cpdef double array_std(double[:] arr):
    """
    Compute the standard deviation of a 1D array using typed memoryviews.

    Parameters
    ----------
    arr : memoryview
        A 1D memoryview of a double array.

    Returns
    -------
    double
        The standard deviation of the array elements.
    """
    cdef int i
    cdef double mean = sum_array(arr) / arr.shape[0]
    cdef double total = 0.0
    for i in range(arr.shape[0]):
        total += (arr[i] - mean) ** 2
    return (total / arr.shape[0]) ** 0.5

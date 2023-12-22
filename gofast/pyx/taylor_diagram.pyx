# taylor_diagram.pyx

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pow

cpdef calculate_statistics(cnp.ndarray[cnp.float64_t, ndim=1] predictions, cnp.ndarray[cnp.float64_t, ndim=1] reference):
    """
    Calculate statistics for Taylor Diagram (standard deviation and correlation).

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions.

    reference : np.ndarray
        Reference data to compare against.

    Returns
    -------
    tuple
        std_dev: Standard deviation of the predictions.
        correlation: Correlation coefficient with the reference data.
    """
    cdef double std_dev = np.std(predictions)
    cdef double mean_pred = np.mean(predictions)
    cdef double mean_ref = np.mean(reference)
    cdef double sum_sq_diff = np.sum((predictions - mean_pred) * (reference - mean_ref))
    cdef double correlation = sum_sq_diff / (std_dev * np.std(reference) * len(predictions))

    return std_dev, correlation

# Function to calculate the dot product between two vectors
cpdef double dot_product(cnp.ndarray[cnp.float64_t, ndim=1] vec1, cnp.ndarray[cnp.float64_t, ndim=1] vec2):
    """
    Calculate the dot product of two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector.

    vec2 : np.ndarray
        Second vector.

    Returns
    -------
    double
        The dot product of the two vectors.
    """
    if vec1.shape[0] != vec2.shape[0]:
        raise ValueError("Vectors must be the same length.")
    return np.dot(vec1, vec2)

# Function to calculate Euclidean distance between two points
cpdef double euclidean_distance(cnp.ndarray[cnp.float64_t, ndim=1] point1, cnp.ndarray[cnp.float64_t, ndim=1] point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : np.ndarray
        First point.

    point2 : np.ndarray
        Second point.

    Returns
    -------
    double
        The Euclidean distance between the two points.
    """
    cdef double dist = 0.0
    cdef int i
    for i in range(point1.shape[0]):
        dist += pow(point1[i] - point2[i], 2)
    return sqrt(dist)

# Element-wise addition of two arrays
cpdef cnp.ndarray[cnp.float64_t, ndim=1] elementwise_add(cnp.ndarray[cnp.float64_t, ndim=1] arr1, cnp.ndarray[cnp.float64_t, ndim=1] arr2):
    """
    Perform element-wise addition of two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First array.

    arr2 : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        The element-wise sum of the two arrays.
    """
    return arr1 + arr2

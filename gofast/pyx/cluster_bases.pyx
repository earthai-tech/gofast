# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#  cluster_bases.pyx 
"""
# python setup.py build_ext --inplace
# cluster_bases.pyx

"""
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef double euclidean_distance(double[:] x, double[:] y):
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    x : ndarray
        A 1D numpy array representing the first vector.
    y : ndarray
        A 1D numpy array representing the second vector.

    Returns
    -------
    double
        The Euclidean distance between the two vectors.
    """
    cdef int n = x.shape[0]
    cdef double sum_sq = 0
    cdef int i
    for i in range(n):
        sum_sq += (x[i] - y[i]) ** 2
    return sqrt(sum_sq)

cdef update_centroids(double[:,:] data, int[:] labels, int n_clusters, double[:,:] centroids):
    """
    Update the centroids based on the current assignment of data points.

    Parameters
    ----------
    data : ndarray
        2D numpy array where each row is a data point.
    labels : ndarray
        1D numpy array of cluster labels for each data point.
    n_clusters : int
        The number of clusters.
    centroids : ndarray
        2D numpy array where each row is a centroid.
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef int[:,:] counts = np.zeros(n_clusters, dtype=np.intc)
    cdef double[:,:] new_centroids = np.zeros((n_clusters, d), dtype=np.float64)
    cdef int i, j

    for i in range(n):
        counts[labels[i]] += 1
        for j in range(d):
            new_centroids[labels[i], j] += data[i, j]

    for i in range(n_clusters):
        if counts[i] > 0:
            for j in range(d):
                centroids[i, j] = new_centroids[i, j] / counts[i]

cpdef naive_k_means(double[:,:] data, int n_clusters, int max_iter=100):
    """
    Perform K-Means clustering on a dataset.

    Parameters
    ----------
    data : ndarray
        2D numpy array where each row is a data point.
    n_clusters : int
        The number of clusters to form.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm (default is 100).

    Returns
    -------
    tuple
        A tuple (labels, centroids) where:
        - labels is a 1D numpy array of cluster labels for each data point.
        - centroids is a 2D numpy array where each row represents a centroid.
        
    Examples 
    ----------
    # Python usage example
    import numpy as np
    from cluster_bases import naive_k_means
    # Example data
    data = np.random.rand(100, 2)
    n_clusters = 3
    
    labels, centroids = k_means_clustering(data, n_clusters)
    print("Labels:", labels)
    print("Centroids:", centroids

    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef double[:,:] centroids = np.random.rand(n_clusters, d)
    cdef int[:] labels = np.zeros(n, dtype=np.intc)
    cdef int iter, i, j
    cdef double min_dist, dist
    cdef int min_index

    for iter in range(max_iter):
        # Assignment step
        for i in range(n):
            min_dist = euclidean_distance(data[i, :], centroids[0, :])
            min_index = 0
            for j in range(1, n_clusters):
                dist = euclidean_distance(data[i, :], centroids[j, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            labels[i] = min_index

        # Update step
        update_centroids(data, labels, n_clusters, centroids)

    return np.asarray(labels), np.asarray(centroids)
